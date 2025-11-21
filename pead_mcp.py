# pead_mcp.py
"""
Yahoo Finance PEAD MCP - Net Income Before Taxes used as 'Sales'

This MCP computes a PEAD score as the average of normalized metrics (all mapped to 0..100):
 - Forward PE (lower is better)
 - Sales YoY (we use Net Income Before Taxes as 'Sales' for bank-like tickers)
 - Sales QoQ
 - NP YoY (Net Profit YoY)
 - NP QoQ
 - EBIDT YoY
 - EBIDT QoQ
 - CE/Profit (Cash & Equivalents / Net Income, lower is better)

Behavior changes from previous version:
 - "Sales" now prefers 'Net Income Before Taxes' (user request). Fallback sequence still available.
 - improved label matching and computation of EBITDA when not directly present.
 - forward PE computed as price / forwardEPS when necessary.

Drop into your MCP folder and run; requires `yfinance`, `pandas`, `numpy`.
"""

import os
import json
import math
import io
import csv
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

pead_server = FastMCP(
    "yfinance_pead",
    instructions="Yahoo Finance PEAD MCP: robust fundamentals + PEAD score for LSE tickers (Sales = Net Income Before Taxes).",
)

# ---------- helpers ----------

def clamp(x, lo=0.0, hi=100.0):
    try:
        xv = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xv))


def map_pct_change_to_0_100(change_pct):
    try:
        v = float(change_pct)
    except Exception:
        return 50.0
    mapped = 50.0 + (v / 2.0)
    return clamp(mapped, 0.0, 100.0)


def map_forward_pe_to_0_100(pe):
    try:
        p = float(pe)
    except Exception:
        return 50.0
    if p <= 0:
        return 100.0
    mapped = (1.0 - (p / 50.0)) * 100.0
    return clamp(mapped, 0.0, 100.0)


def map_ce_profit_to_0_100(ratio):
    try:
        r = float(ratio)
    except Exception:
        return 50.0
    mapped = (1.0 - (r / 50.0)) * 100.0
    return clamp(mapped, 0.0, 100.0)


def try_get_series_pct_change(curr, prev):
    try:
        if curr is None or prev is None:
            return None
        if prev == 0:
            return None
        return (float(curr) - float(prev)) / abs(float(prev)) * 100.0
    except Exception:
        return None


# UPDATED key lists: Sales now prefers Net Income Before Taxes per user request
SALES_KEYS = [
    "Net Income Before Taxes",
    "Profit before tax",
    "Profit Before Tax",
    "Pretax Income",
    # fallback generic revenue fields
    "Total Revenue",
    "Revenue",
    "Net Sales",
    "Sales",
]
NP_KEYS = [
    "Net Income",
    "Net income",
    "NetIncome",
    "Net Income Attributable To Parent",
    "Profit (Loss)",
    "Net (loss) income",
]
EBIDT_KEYS = ["EBITDA", "Ebitda", "EBIT", "Operating Income", "Operating income"]
CAPEX_KEYS = ["Capital Expenditure", "Capital Expenditures", "Capex", "Purchase of property, plant and equipment"]


def find_latest_pair(df: pd.DataFrame, keys: List[str]):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
    for key in keys:
        if key in df.index:
            try:
                series = df.loc[key].dropna()
                if len(series) >= 2:
                    return float(series.iloc[0]), float(series.iloc[1])
                if len(series) == 1:
                    return float(series.iloc[0]), None
            except Exception:
                continue
    for idx in df.index:
        try:
            lab = str(idx).lower()
            for k in keys:
                if k.lower() in lab:
                    series = df.loc[idx].dropna()
                    if len(series) >= 2:
                        return float(series.iloc[0]), float(series.iloc[1])
                    if len(series) == 1:
                        return float(series.iloc[0]), None
        except Exception:
            continue
    return None, None


# compute EBITDA if missing using Operating Income + Depreciation + Amortization when possible
def compute_ebitda_from_components(t: yf.Ticker):
    try:
        fin_q = t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame()
        fin_y = t.financials if hasattr(t, "financials") else pd.DataFrame()
    except Exception:
        return None, None

    # look for depreciation & amortization labels
    da_keys = ["Depreciation", "Depreciation & Amortization", "Depreciation and amortisation", "Depreciation & amortization"]

    def find_latest(df, keys):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        for k in keys:
            if k in df.index:
                s = df.loc[k].dropna()
                if len(s) >= 1:
                    return float(s.iloc[0])
        for idx in df.index:
            lab = str(idx).lower()
            for k in keys:
                if k.lower() in lab:
                    s = df.loc[idx].dropna()
                    if len(s) >= 1:
                        return float(s.iloc[0])
        return None

    # Attempt quarterly then yearly
    op_income_q, _ = find_latest_pair(t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame(), ["Operating Income", "Operating income", "EBIT"]) if hasattr(t, "quarterly_financials") else (None, None)
    dep_q = find_latest(t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame(), da_keys)
    ebitda_q = None
    if op_income_q is not None and dep_q is not None:
        ebitda_q = op_income_q + dep_q

    op_income_y, _ = find_latest_pair(t.financials if hasattr(t, "financials") else pd.DataFrame(), ["Operating Income", "Operating income", "EBIT"]) if hasattr(t, "financials") else (None, None)
    dep_y = find_latest(t.financials if hasattr(t, "financials") else pd.DataFrame(), da_keys)
    ebitda_y = None
    if op_income_y is not None and dep_y is not None:
        ebitda_y = op_income_y + dep_y

    return ebitda_q, ebitda_y


@pead_server.tool(
    name="get_pead_table",
    description="Compute PEAD table for LSE tickers. Parameters: tickers, days (lookback).",
)
async def get_pead_table(tickers: str, days: int = 15) -> str:
    if not tickers or not tickers.strip():
        return json.dumps({"error": "tickers required"})

    symbols = [s.strip().upper() for s in tickers.replace(",", " ").split() if s.strip()]
    if not symbols:
        return json.dumps({"error": "no tickers parsed"})

    rows = []
    idx = 1

    for s in symbols:
        tkr = s if s.endswith('.L') else s + '.L'
        try:
            t = yf.Ticker(tkr)
        except Exception as e:
            rows.append({
                "S.No.": idx,
                "Stock Name": tkr,
                "PEAD Score": None,
                "Result Date": None,
                "Forward PE": None,
                "Sales YoY": None,
                "Sales QoQ": None,
                "NP YoY": None,
                "NP QoQ": None,
                "EBIDT YoY": None,
                "EBIDT QoQ": None,
                "CE/Profit": None,
                "error": f"yfinance error: {e}",
            })
            idx += 1
            continue

        # core info
        info = {}
        try:
            info = t.info if hasattr(t, 'info') else {}
        except Exception:
            info = {}

        # price
        price = None
        try:
            price = float(info.get('currentPrice') or info.get('regularMarketPrice') or (t.history(period='1d').tail(1)['Close'].iloc[0]))
        except Exception:
            price = None

        # forward PE
        forward_pe = info.get('forwardPE') or info.get('forwardPERaw') or None
        if (forward_pe is None or (isinstance(forward_pe, (int, float)) and forward_pe <= 1.0)) and price is not None:
            f_eps = info.get('forwardEps') or info.get('forwardEarningsPerShare') or info.get('epsForward') or None
            try:
                if f_eps and float(f_eps) != 0:
                    forward_pe = round(price / float(f_eps), 4)
            except Exception:
                pass

        # frames
        try:
            fin_q = t.quarterly_financials if hasattr(t, 'quarterly_financials') else pd.DataFrame()
        except Exception:
            fin_q = pd.DataFrame()
        try:
            fin_y = t.financials if hasattr(t, 'financials') else pd.DataFrame()
        except Exception:
            fin_y = pd.DataFrame()
        try:
            cf = t.cashflow if hasattr(t, 'cashflow') else pd.DataFrame()
        except Exception:
            cf = pd.DataFrame()
        try:
            bal = t.balance_sheet if hasattr(t, 'balance_sheet') else pd.DataFrame()
        except Exception:
            bal = pd.DataFrame()

        # sales (Net Income Before Taxes preferred)
        sales_curr, sales_prev = find_latest_pair(fin_q, SALES_KEYS)
        if sales_curr is None:
            sales_curr, sales_prev = find_latest_pair(fin_y, SALES_KEYS)

        # NP
        np_curr, np_prev = find_latest_pair(fin_q, NP_KEYS)
        if np_curr is None:
            np_curr, np_prev = find_latest_pair(fin_y, NP_KEYS)

        # EBITDA
        ebidt_curr, ebidt_prev = find_latest_pair(fin_q, EBIDT_KEYS)
        if ebidt_curr is None:
            ebidt_curr, ebidt_prev = find_latest_pair(fin_y, EBIDT_KEYS)

        # if EBITDA missing, attempt compute
        if (ebidt_curr is None or ebidt_prev is None):
            try:
                e_q, e_y = compute_ebitda_from_components(t)
                if ebidt_curr is None:
                    ebidt_curr = e_q
                if ebidt_prev is None:
                    ebidt_prev = e_y
            except Exception:
                pass

        # capex for CE/Profit
        capex_curr, capex_prev = find_latest_pair(cf, CAPEX_KEYS)
        if capex_curr is None:
            capex_curr, capex_prev = find_latest_pair(fin_q, CAPEX_KEYS)

        # percent changes
        def pct(curr, prev):
            try:
                if curr is None or prev is None:
                    return None
                if prev == 0:
                    return None
                return (float(curr) - float(prev)) / abs(float(prev)) * 100.0
            except Exception:
                return None

        sales_qoq = pct(sales_curr, sales_prev)
        sales_yoy = None
        try:
            for key in SALES_KEYS:
                if isinstance(fin_y, pd.DataFrame) and key in fin_y.index:
                    vals = [v for v in fin_y.loc[key].dropna().tolist()]
                    if len(vals) >= 2:
                        sales_yoy = pct(vals[0], vals[1])
                        break
        except Exception:
            sales_yoy = None
        if sales_yoy is None:
            sales_yoy = sales_qoq

        np_qoq = pct(np_curr, np_prev)
        np_yoy = None
        try:
            for key in NP_KEYS:
                if isinstance(fin_y, pd.DataFrame) and key in fin_y.index:
                    vals = [v for v in fin_y.loc[key].dropna().tolist()]
                    if len(vals) >= 2:
                        np_yoy = pct(vals[0], vals[1])
                        break
        except Exception:
            np_yoy = None
        if np_yoy is None:
            np_yoy = np_qoq

        ebidt_qoq = pct(ebidt_curr, ebidt_prev)
        ebidt_yoy = None
        try:
            for key in EBIDT_KEYS:
                if isinstance(fin_y, pd.DataFrame) and key in fin_y.index:
                    vals = [v for v in fin_y.loc[key].dropna().tolist()]
                    if len(vals) >= 2:
                        ebidt_yoy = pct(vals[0], vals[1])
                        break
        except Exception:
            ebidt_yoy = None
        if ebidt_yoy is None:
            ebidt_yoy = ebidt_qoq

        # CE/Profit
        ce_profit_ratio = None
        try:
            cash_val = None
            if isinstance(bal, pd.DataFrame):
                for candidate in ["Cash and cash equivalents", "Cash", "Cash & equivalents", "Cash and cash equivalents (Total)"]:
                    if candidate in bal.index:
                        series = bal.loc[candidate].dropna()
                        if len(series) >= 1:
                            cash_val = float(series.iloc[0])
                            break
                if cash_val is None:
                    for idx_label in bal.index:
                        try:
                            if "cash" in str(idx_label).lower():
                                series = bal.loc[idx_label].dropna()
                                if len(series) >= 1:
                                    cash_val = float(series.iloc[0])
                                    break
                        except Exception:
                            continue
            ni_val = None
            if isinstance(fin_y, pd.DataFrame):
                for key in NP_KEYS:
                    if key in fin_y.index:
                        vals = [v for v in fin_y.loc[key].dropna().tolist()]
                        if len(vals) >= 1:
                            ni_val = float(vals[0])
                            break
            if ni_val is None and np_curr is not None:
                ni_val = float(np_curr)
            if cash_val is not None and ni_val is not None and ni_val != 0:
                ce_profit_ratio = abs(cash_val) / abs(ni_val)
        except Exception:
            ce_profit_ratio = None

        # rounding
        def r2(x):
            return None if x is None else round(float(x), 2)
        def r4(x):
            return None if x is None else round(float(x), 4)

        sales_qoq = r2(sales_qoq)
        sales_yoy = r2(sales_yoy)
        np_qoq = r2(np_qoq)
        np_yoy = r2(np_yoy)
        ebidt_qoq = r2(ebidt_qoq)
        ebidt_yoy = r2(ebidt_yoy)
        forward_pe = r4(forward_pe)
        ce_profit_ratio = r4(ce_profit_ratio)

        # mapping to 0..100
        mapped_forward_pe = map_forward_pe_to_0_100(forward_pe) if forward_pe is not None else None
        mapped_sales_yoy = map_pct_change_to_0_100(sales_yoy) if sales_yoy is not None else None
        mapped_sales_qoq = map_pct_change_to_0_100(sales_qoq) if sales_qoq is not None else None
        mapped_np_yoy = map_pct_change_to_0_100(np_yoy) if np_yoy is not None else None
        mapped_np_qoq = map_pct_change_to_0_100(np_qoq) if np_qoq is not None else None
        mapped_ebidt_yoy = map_pct_change_to_0_100(ebidt_yoy) if ebidt_yoy is not None else None
        mapped_ebidt_qoq = map_pct_change_to_0_100(ebidt_qoq) if ebidt_qoq is not None else None
        mapped_ce_profit = map_ce_profit_to_0_100(ce_profit_ratio) if ce_profit_ratio is not None else None

        mapped_list = []
        for m in [mapped_forward_pe, mapped_sales_yoy, mapped_sales_qoq, mapped_np_yoy, mapped_np_qoq, mapped_ebidt_yoy, mapped_ebidt_qoq, mapped_ce_profit]:
            if m is not None:
                mapped_list.append(float(m))

        pead_score = None
        if mapped_list:
            pead_score = round(float(sum(mapped_list) / len(mapped_list)), 2)

        # result_date
        result_date = None
        try:
            if info.get('earningsTimestamp'):
                dt = datetime.fromtimestamp(int(info.get('earningsTimestamp')), tz=timezone.utc)
                if (datetime.now(timezone.utc) - dt).days <= days:
                    result_date = dt.isoformat()
        except Exception:
            result_date = None
        if not result_date:
            chosen_dt = None
            try:
                if isinstance(fin_q, pd.DataFrame) and not fin_q.empty:
                    cols = list(fin_q.columns)
                    try:
                        col0 = cols[0]
                        dt0 = pd.to_datetime(col0, utc=True, errors='coerce')
                        if not pd.isna(dt0) and (datetime.now(timezone.utc) - dt0.to_pydatetime()).days <= days:
                            chosen_dt = dt0.to_pydatetime()
                    except Exception:
                        pass
                if not chosen_dt and isinstance(fin_y, pd.DataFrame) and not fin_y.empty:
                    cols = list(fin_y.columns)
                    try:
                        col0 = cols[0]
                        dt0 = pd.to_datetime(col0, utc=True, errors='coerce')
                        if not pd.isna(dt0) and (datetime.now(timezone.utc) - dt0.to_pydatetime()).days <= days:
                            chosen_dt = dt0.to_pydatetime()
                    except Exception:
                        pass
            except Exception:
                chosen_dt = None
            if chosen_dt:
                result_date = chosen_dt.isoformat()

        rows.append({
            "S.No.": idx,
            "Stock Name": f"{tkr} — {info.get('longName') or info.get('shortName') or tkr}",
            "PEAD Score": pead_score,
            "Result Date": result_date,
            "Forward PE": forward_pe,
            "Sales YoY": sales_yoy,
            "Sales QoQ": sales_qoq,
            "NP YoY": np_yoy,
            "NP QoQ": np_qoq,
            "EBIDT YoY": ebidt_yoy,
            "EBIDT QoQ": ebidt_qoq,
            "CE/Profit": ce_profit_ratio,
        })
        idx += 1

    headers = ["S.No.", "Stock Name", "PEAD Score", "Result Date", "Forward PE", "Sales YoY", "Sales QoQ", "NP YoY", "NP QoQ", "EBIDT YoY", "EBIDT QoQ", "CE/Profit"]
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    def trunc(x, n=120):
        if x is None:
            return ""
        s = str(x)
        return (s[:n] + "...") if len(s) > n else s
    for r in rows:
        vals = [trunc(r.get(h, "")) for h in headers]
        vals = [v.replace("|", "\\|") for v in vals]
        md_lines.append("| " + " | ".join(vals) + " |")
    table_markdown = "".join(md_lines)

    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow({h: r.get(h, "") for h in headers})
    csv_text = csv_buf.getvalue()

    return json.dumps({"rows": rows, "table_markdown": table_markdown, "csv": csv_text, "count": len(rows)}, default=str)


if __name__ == "__main__":
    print("Starting Yahoo Finance PEAD MCP server (Sales = Net Income Before Taxes)...")
    pead_server.run(transport="stdio")
