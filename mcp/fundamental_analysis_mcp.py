"""
Yahoo Finance Fundamental MCP

Produces a fundamental analysis table with columns:
- S.No., Stock Name, P/E Ratio, P/B Ratio, Market Cap, Profit Margin, Debt/Equity Ratio, Dividend Yield

Features:
- Normalizes tickers to LSE (.L) by default
- Uses yfinance `info` and financial statements (financials, balance_sheet) with robust fallbacks
- Accepts an optional `file_url` parameter pointing to a local JSON file (useful if you pre-fetched Ticker objects). The assistant will pass the file path you uploaded as the `file_url` when testing.

Drop into your MCP folder and run it. Requires `yfinance`, `pandas`, `numpy`.
"""

import os
import json
import math
import io
import csv
from typing import Optional, List
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

fund_server = FastMCP(
    "yfinance_fundamentals",
    instructions="Return fundamentals table: P/E, P/B, Market Cap, Profit Margin, Debt/Equity, Dividend Yield",
)


def normalize_t(ticker: str) -> str:
    t = ticker.strip().upper()
    return t if t.endswith('.L') else t + '.L'


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def format_pct(v):
    if v is None:
        return None
    return f"{round(float(v),2)}%"


def format_num(v):
    if v is None:
        return None
    # large int formatting avoided; return rounded
    try:
        if abs(float(v)) >= 1e9:
            return str(round(float(v)/1e9,2)) + 'B'
        if abs(float(v)) >= 1e6:
            return str(round(float(v)/1e6,2)) + 'M'
        return str(round(float(v),2))
    except Exception:
        return str(v)


def load_json_file_if_exists(path: str) -> Optional[dict]:
    try:
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None
    return None


@fund_server.tool(
    name='get_fundamentals_table',
    description='Return fundamentals table for tickers. Optional file_url to load pre-fetched JSON.'
)
async def get_fundamentals_table(tickers: str, file_url: Optional[str] = None) -> str:
    """
    tickers: comma or space separated tickers (e.g. "LLOY,HSBA")
    file_url: optional local file path to a JSON containing yfinance info for a ticker (for testing)
    """
    if not tickers or not tickers.strip():
        return json.dumps({'error':'tickers required'})

    symbols = [s.strip() for s in tickers.replace(',', ' ').split() if s.strip()]
    rows = []
    idx = 1

    # if file_url provided and exists, attempt to load
    prefetch = None
    if file_url:
        prefetch = load_json_file_if_exists(file_url)

    for s in symbols:
        tkr = normalize_t(s)
        # attempt to use prefetch if it matches ticker string
        info = {}
        fin_y = pd.DataFrame()
        bal = pd.DataFrame()
        try:
            if prefetch and (str(prefetch.get('ticker') or '').upper() in (tkr, s.upper())):
                # support packed file structures or direct mapping
                # if prefetch is a dict with 'info' key
                if isinstance(prefetch.get('info'), dict):
                    info = prefetch.get('info')
                else:
                    info = prefetch
                # try to use raw frames if provided
                for k in ('financials', 'quarterly_financials', 'balance_sheet', 'cashflow'):
                    if k in prefetch and isinstance(prefetch[k], dict):
                        try:
                            df = pd.DataFrame(prefetch[k])
                            if 'financials' == k:
                                fin_y = df
                            if 'balance_sheet' == k:
                                bal = df
                        except Exception:
                            pass
            else:
                t = yf.Ticker(tkr)
                try:
                    info = t.info if hasattr(t, 'info') else {}
                except Exception:
                    info = {}
                try:
                    fin_y = t.financials if hasattr(t, 'financials') else pd.DataFrame()
                except Exception:
                    fin_y = pd.DataFrame()
                try:
                    bal = t.balance_sheet if hasattr(t, 'balance_sheet') else pd.DataFrame()
                except Exception:
                    bal = pd.DataFrame()
        except Exception as e:
            info = {}
            fin_y = pd.DataFrame()
            bal = pd.DataFrame()

        # P/E
        pe = None
        try:
            pe = info.get('trailingPE') or info.get('forwardPE') or info.get('pe') or info.get('priceToEarnings')
            pe = safe_float(pe)
            # fallback compute price / epsTrailing
            if pe is None:
                price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
                eps = safe_float(info.get('trailingEps') or info.get('epsTrailingTwelveMonths') or info.get('eps'))
                if price is not None and eps not in (None, 0):
                    pe = price / eps
        except Exception:
            pe = None

        # P/B
        pb = None
        try:
            pb = safe_float(info.get('priceToBook') or info.get('pb'))
            # fallback compute price / bookValue
            if pb is None:
                price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
                bv = safe_float(info.get('bookValue'))
                if price is not None and bv not in (None, 0):
                    pb = price / bv
        except Exception:
            pb = None

        # Market cap
        mcap = None
        try:
            mcap = safe_float(info.get('marketCap') or info.get('market_cap') or info.get('MarketCap'))
        except Exception:
            mcap = None

        # Profit margin = Net Income / Total Revenue *100
        profit_margin = None
        try:
            # try info
            pm = info.get('profitMargins') or info.get('profitMargin')
            if pm is not None:
                profit_margin = safe_float(pm) * 100.0
            else:
                # try compute from financials
                if isinstance(fin_y, pd.DataFrame) and not fin_y.empty:
                    # look for Net Income and Total Revenue rows
                    def get_latest_from_df(df, keynames):
                        for k in keynames:
                            if k in df.index:
                                vals = [v for v in df.loc[k].dropna().tolist()]
                                if vals:
                                    return safe_float(vals[0])
                        # fuzzy match
                        for idx_label in df.index:
                            lab = str(idx_label).lower()
                            for k in keynames:
                                if k.lower() in lab:
                                    vals = [v for v in df.loc[idx_label].dropna().tolist()]
                                    if vals:
                                        return safe_float(vals[0])
                        return None

                    net_income = get_latest_from_df(fin_y, ['Net Income', 'Net income', 'NetIncome', 'Net Income Attributable To Parent', 'Profit (Loss)'])
                    revenue = get_latest_from_df(fin_y, ['Total Revenue', 'Revenue', 'Net Sales', 'Sales'])
                    if net_income is not None and revenue not in (None, 0):
                        profit_margin = (net_income / revenue) * 100.0
        except Exception:
            profit_margin = None

        # Debt/Equity = Total Liab / Total Stockholder Equity
        debt_equity = None
        try:
            if isinstance(bal, pd.DataFrame) and not bal.empty:
                def get_one_from_bal(df, keys):
                    for k in keys:
                        if k in df.index:
                            vals = [v for v in df.loc[k].dropna().tolist()]
                            if vals:
                                return safe_float(vals[0])
                    for idx_label in df.index:
                        lab = str(idx_label).lower()
                        for k in keys:
                            if k.lower() in lab:
                                vals = [v for v in df.loc[idx_label].dropna().tolist()]
                                if vals:
                                    return safe_float(vals[0])
                    return None

                total_liab = get_one_from_bal(bal, ['Total Liab', 'Total liabilities', 'Total Liabilities', 'Total non-current liabilities'])
                total_equity = get_one_from_bal(bal, ['Total Stockholders Equity', 'Total shareholders equity', 'Total Equity', 'Stockholders Equity'])
                if total_liab is not None and total_equity not in (None, 0):
                    debt_equity = total_liab / total_equity
            # fallback: use info
            if debt_equity is None:
                debt_equity = safe_float(info.get('debtToEquity') or info.get('debtToEquityRatio') or info.get('totalDebt') and info.get('totalAssets'))
        except Exception:
            debt_equity = None

        # Dividend yield
        div_yield = None
        try:
            div_yield = safe_float(info.get('dividendYield') or info.get('yield'))
            if div_yield is not None:
                div_yield = div_yield * 100.0
            else:
                # try compute from trailing annual dividend
                trailing_div = safe_float(info.get('trailingAnnualDividendRate') or info.get('dividendRate'))
                price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
                if trailing_div is not None and price not in (None, 0):
                    div_yield = (trailing_div / price) * 100.0
        except Exception:
            div_yield = None

        # rounding and formatting
        pe_fmt = None if pe is None else round(float(pe),2)
        pb_fmt = None if pb is None else round(float(pb),2)
        mcap_fmt = format_num(mcap)
        pm_fmt = None if profit_margin is None else round(float(profit_margin),2)
        de_fmt = None if debt_equity is None else round(float(debt_equity),2)
        div_fmt = None if div_yield is None else round(float(div_yield),2)

        rows.append({
            'S.No.': idx,
            'Stock Name': s.upper(),
            'P/E Ratio': pe_fmt,
            'P/B Ratio': pb_fmt,
            'Market Cap': mcap_fmt,
            'Profit Margin': None if pm_fmt is None else f"{pm_fmt}%",
            'Debt/Equity Ratio': de_fmt,
            'Dividend Yield': None if div_fmt is None else f"{div_fmt}%",
        })
        idx += 1

    # build markdown and csv
    headers = ['S.No.', 'Stock Name', 'P/E Ratio', 'P/B Ratio', 'Market Cap', 'Profit Margin', 'Debt/Equity Ratio', 'Dividend Yield']
    md_lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for r in rows:
        vals = [str(r.get(h, '') or '') for h in headers]
        vals = [v.replace('|', '\\|') for v in vals]
        md_lines.append('| ' + ' | '.join(vals) + ' |')
    table_markdown = '\n'.join(md_lines)

    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow({h: r.get(h, '') for h in headers})
    csv_text = csv_buf.getvalue()

    return json.dumps({'rows': rows, 'table_markdown': table_markdown, 'csv': csv_text, 'count': len(rows)}, default=str)


if __name__ == '__main__':
    print('Starting Yahoo Finance Fundamentals MCP...')
    fund_server.run(transport='stdio')
