# pead_mcp.py
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

# Create MCP server
pead_server = FastMCP(
    "yfinance_pead",
    instructions="""
Yahoo Finance PEAD MCP: returns tabular PEAD metrics for LSE tickers.
PEAD = average of normalized Forward PE, Sales YoY, Sales QoQ, NP YoY, NP QoQ, EBIDT YoY, EBIDT QoQ, CE/Profit (all 0..100).
""",
)

# ---------- helpers ----------
def clamp(x, lo=0.0, hi=100.0):
    try:
        xv = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xv))

def map_pct_change_to_0_100(change_pct):
    # change_pct is in percent (e.g. -20.0 for -20%)
    # mapping: -100 -> 0, 0 -> 50, +100 -> 100
    try:
        v = float(change_pct)
    except Exception:
        return 50.0  # neutral fallback
    mapped = 50.0 + (v / 2.0)
    return clamp(mapped, 0.0, 100.0)

def map_forward_pe_to_0_100(pe):
    # lower PE -> higher score; map 0..50 linearly to 100..0
    try:
        p = float(pe)
    except Exception:
        return 50.0
    if p <= 0:
        return 100.0
    # linear mapping with upper cap:
    mapped = (1.0 - (p / 50.0)) * 100.0
    return clamp(mapped, 0.0, 100.0)

def map_ce_profit_to_0_100(ratio):
    # lower is better. ratio around 0 -> 100, ratio 50 or more -> 0
    try:
        r = float(ratio)
    except Exception:
        return 50.0
    mapped = (1.0 - (r / 50.0)) * 100.0
    return clamp(mapped, 0.0, 100.0)

def try_get_series_pct_change(curr, prev):
    # Both inputs may be numbers or None. If either missing return None
    try:
        if curr is None or prev is None:
            return None
        if prev == 0:
            return None
        return (float(curr) - float(prev)) / abs(float(prev)) * 100.0
    except Exception:
        return None

# ---------- main tool ----------
@pead_server.tool(
    name="get_pead_table",
    description="""
Compute PEAD table for one or more LSE tickers.
Params:
  tickers: str (comma/space separated, eg 'LLOY,HSBA')
  days: int lookback for "recent results" - used to pick latest reporting column (default 15)
Returns JSON with rows, markdown table and csv.
""",
)
async def get_pead_table(tickers: str, days: int = 15) -> str:
    if not tickers or not tickers.strip():
        return json.dumps({"error": "tickers required"})

    parsed = [p.strip().upper() for p in tickers.replace(",", " ").split() if p.strip()]
    rows = []
    idx = 1

    for tk in parsed:
        # enforce LSE suffix
        tkr = tk if tk.endswith(".L") else tk + ".L"

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
                "error": f"yfinance error: {e}"
            })
            idx += 1
            continue

        # fetch info + financials
        info = t.info if hasattr(t, "info") else {}
        # Try to extract forwardPE from info
        forward_pe = info.get("forwardPE") or info.get("forwardPERaw") or info.get("trailingPE") or None

        # We will try to get YoY / QoQ changes from financial statements if available.
        # Prefer quarterly numbers (quarterly_financials) for QoQ; yearly (financials) for YoY.
        try:
            fin_q = t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame()
            fin_y = t.financials if hasattr(t, "financials") else pd.DataFrame()
            # sometimes these DataFrames have columns as timestamps; convert to numeric dictionaries
        except Exception:
            fin_q = pd.DataFrame()
            fin_y = pd.DataFrame()

        # helper to read metric from financials/balance etc.
        def get_metric_latest_change(metric_keys: list):
            """
            Try to return (latest_val, prev_val) searching metric_keys in quarterly then yearly.
            Keys are strings exactly as in yfinance's financials column indices (may vary by company).
            """
            # Search quarterly (for QoQ)
            for df in (fin_q, fin_y):
                try:
                    for key in metric_keys:
                        if key in df.index:
                            series = df.loc[key]
                            if series is not None and len(series) >= 2:
                                # series is pandas Series with columns as dates (newest first)
                                vals = [v for v in series.tolist() if v is not None]
                                if len(vals) >= 2:
                                    return vals[0], vals[1]
                                elif len(vals) == 1:
                                    return vals[0], None
                except Exception:
                    continue
            return None, None

        # Common metric keys to search (these may vary per ticker)
        sales_keys = ["Total Revenue", "Revenue", "Net Sales", "Sales/Revenue"]
        np_keys = ["Net Income", "Net Income Common Stockholders", "Net Income Available to Common Stockholders", "Net Income Attributable To Parent"]
        ebidt_keys = ["EBITDA", "EBIT", "Ebit", "Operating Income"]  # try a few
        capex_keys = ["Capital Expenditure", "Capital Expenditures", "Capital Expenditure (CapEx)"]

        # Get Sales latest and prev
        sales_curr, sales_prev = get_metric_latest_change(sales_keys)
        np_curr, np_prev = get_metric_latest_change(np_keys)
        ebidt_curr, ebidt_prev = get_metric_latest_change(ebidt_keys)
        capex_curr, capex_prev = get_metric_latest_change(capex_keys)

        # compute percent changes
        sales_yoy = None
        sales_qoq = None
        np_yoy = None
        np_qoq = None
        ebidt_yoy = None
        ebidt_qoq = None
        ce_profit_ratio = None

        # For YoY we try to look into fin_y (yearly) first: compare most recent two years
        # For QoQ we prefer fin_q (quarterly) compare last two quarters
        # Our helper above searched both; here we accept the returned pair as "latest, prev".

        # Map sales / np / ebidt changes
        if sales_curr is not None and sales_prev is not None:
            # Determine whether these are quarter/year values - we don't strictly need to know; we map them directly for YoY and QoQ as same source.
            # We'll set both YoY and QoQ to the percent change if we can't disambiguate (user can change mapping rules later).
            pct = try_get_series_pct_change(sales_curr, sales_prev)
            if pct is not None:
                sales_qoq = pct
                sales_yoy = pct  # fallback; if yearly available separately it will be overwritten later
        # attempt to find yearly sales specifically
        try:
            # check fin_y separately for yearly comparison
            for key in sales_keys:
                if key in fin_y.index:
                    vals = [v for v in fin_y.loc[key].tolist() if v is not None]
                    if len(vals) >= 2:
                        sales_yoy = try_get_series_pct_change(vals[0], vals[1])
                        break
        except Exception:
            pass

        # NP
        if np_curr is not None and np_prev is not None:
            np_qoq = try_get_series_pct_change(np_curr, np_prev)
            np_yoy = np_qoq
        try:
            for key in np_keys:
                if key in fin_y.index:
                    vals = [v for v in fin_y.loc[key].tolist() if v is not None]
                    if len(vals) >= 2:
                        np_yoy = try_get_series_pct_change(vals[0], vals[1])
                        break
        except Exception:
            pass

        # EBIDT
        if ebidt_curr is not None and ebidt_prev is not None:
            ebidt_qoq = try_get_series_pct_change(ebidt_curr, ebidt_prev)
            ebidt_yoy = ebidt_qoq
        try:
            for key in ebidt_keys:
                if key in fin_y.index:
                    vals = [v for v in fin_y.loc[key].tolist() if v is not None]
                    if len(vals) >= 2:
                        ebidt_yoy = try_get_series_pct_change(vals[0], vals[1])
                        break
        except Exception:
            pass

        # CE/Profit ratio: capex / net income (most recent)
        try:
            if capex_curr is not None:
                # use latest net income to compute ratio
                net_for_ratio = None
                # pick yearly net income if available:
                try:
                    for key in np_keys:
                        if key in fin_y.index:
                            vals = [v for v in fin_y.loc[key].tolist() if v is not None]
                            if vals:
                                net_for_ratio = vals[0]
                                break
                except Exception:
                    pass
                if net_for_ratio is None:
                    # fallback to np_curr
                    net_for_ratio = np_curr
                if net_for_ratio and net_for_ratio != 0:
                    ce_profit_ratio = float(abs(capex_curr)) / abs(float(net_for_ratio))
        except Exception:
            ce_profit_ratio = None

        # Map all metrics to 0..100
        mapped_forward_pe = map_forward_pe_to_0_100(forward_pe) if forward_pe is not None else None
        mapped_sales_yoy = map_pct_change_to_0_100(sales_yoy) if sales_yoy is not None else None
        mapped_sales_qoq = map_pct_change_to_0_100(sales_qoq) if sales_qoq is not None else None
        mapped_np_yoy = map_pct_change_to_0_100(np_yoy) if np_yoy is not None else None
        mapped_np_qoq = map_pct_change_to_0_100(np_qoq) if np_qoq is not None else None
        mapped_ebidt_yoy = map_pct_change_to_0_100(ebidt_yoy) if ebidt_yoy is not None else None
        mapped_ebidt_qoq = map_pct_change_to_0_100(ebidt_qoq) if ebidt_qoq is not None else None
        mapped_ce_profit = map_ce_profit_to_0_100(ce_profit_ratio) if ce_profit_ratio is not None else None

        # Build list of available mapped metrics (only include those present)
        mapped_list = []
        for m in [mapped_forward_pe, mapped_sales_yoy, mapped_sales_qoq, mapped_np_yoy, mapped_np_qoq, mapped_ebidt_yoy, mapped_ebidt_qoq, mapped_ce_profit]:
            if m is not None:
                mapped_list.append(float(m))

        pead_score = None
        if mapped_list:
            pead_score = round(float(sum(mapped_list) / len(mapped_list)), 2)

        # pick a "result date" from earningsTimestamp or most recent financial column header if available
        result_date = None
        try:
            # try earningsTimestamp -> convert to iso
            if info.get("earningsTimestamp"):
                dt = datetime.fromtimestamp(int(info.get("earningsTimestamp")), tz=timezone.utc)
                result_date = dt.isoformat()
        except Exception:
            result_date = None

        # fallback: try last column in fin_y or fin_q
        try:
            if not result_date:
                if not fin_y.empty:
                    cols = list(fin_y.columns)
                    if cols:
                        # fin_y columns often are Timestamps (pd.Timestamp)
                        lastcol = cols[0]
                        result_date = str(lastcol)
                elif not fin_q.empty:
                    cols = list(fin_q.columns)
                    if cols:
                        lastcol = cols[0]
                        result_date = str(lastcol)
        except Exception:
            pass

        rows.append({
            "S.No.": idx,
            "Stock Name": f"{tkr} \u2014 {info.get('longName') or info.get('shortName') or tkr}",
            "PEAD Score": pead_score,
            "Result Date": result_date,
            "Forward PE": forward_pe,
            "Sales YoY": sales_yoy,
            "Sales QoQ": sales_qoq,
            "NP YoY": np_yoy,
            "NP QoQ": np_qoq,
            "EBIDT YoY": ebidt_yoy,
            "EBIDT QoQ": ebidt_qoq,
            "CE/Profit": round(ce_profit_ratio, 4) if ce_profit_ratio is not None else None,
        })
        idx += 1

    # Build markdown table and CSV similar to previous tool
    headers = ["S.No.", "Stock Name", "PEAD Score", "Result Date", "Forward PE", "Sales YoY", "Sales QoQ", "NP YoY", "NP QoQ", "EBIDT YoY", "EBIDT QoQ", "CE/Profit"]
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md_vals = [str(r.get(h, "")) if r.get(h, "") is not None else "" for h in headers]
        md_vals = [v.replace("|", "\\|") for v in md_vals]
        md_lines.append("| " + " | ".join(md_vals) + " |")
    table_markdown = "\n".join(md_lines)

    # Build CSV
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow({h: r.get(h, "") for h in headers})
    csv_text = csv_buf.getvalue()

    return json.dumps({
        "rows": rows,
        "table_markdown": table_markdown,
        "csv": csv_text,
        "count": len(rows),
    }, default=str)

if __name__ == "__main__":
    print("Starting Yahoo Finance PEAD MCP server...")
    pead_server.run(transport="stdio")
