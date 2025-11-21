"""
Yahoo Finance Technical MCP

Produces a table with columns:
- S.No., Stock Name, Current Price, Price Change (%), 52W H/L, RSI, MACD, Overall Tech Single

Heuristics for Overall Tech Single (simple, tunable):
- "Greed": MACD > 0 and RSI >= 55
- "Neutral": MACD >= 0 and 45 <= RSI < 55, or small price move
- "Fear": MACD < 0 and RSI <= 45

Drop this file into your MCP folder and run it. Uses yfinance + pandas + numpy.
"""

import os
import json
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

tech_server = FastMCP(
    "yfinance_tech",
    instructions="Return technical indicators table: Current Price, Price Change, 52W H/L, RSI, MACD, Overall Tech Single",
)


# --- Technical indicator helpers (no external TA dependency) ---

def compute_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    try:
        delta = series.diff().dropna()
        up = delta.clip(lower=0.0)
        down = -1.0 * delta.clip(upper=0.0)
        ma_up = up.ewm(alpha=1.0/period, adjust=False).mean()
        ma_down = down.ewm(alpha=1.0/period, adjust=False).mean()
        rs = ma_up / (ma_down.replace(0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        if rsi.empty:
            return None
        return float(rsi.iloc[-1])
    except Exception:
        return None


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[float]:
    try:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line.iloc[-1] - signal_line.iloc[-1]
        return float(macd_hist)
    except Exception:
        return None


def normalize_ticker_for_lse(ticker: str) -> str:
    t = ticker.strip().upper()
    return t if t.endswith('.L') else t + '.L'


@tech_server.tool(
    name='get_technical_table',
    description='Return technical indicators table for tickers (comma or space separated).',
)
async def get_technical_table(tickers: str, period_days: int = 365) -> str:
    if not tickers or not tickers.strip():
        return json.dumps({'error': 'tickers required'})

    symbols = [s.strip() for s in tickers.replace(',', ' ').split() if s.strip()]
    rows = []
    idx = 1

    for s in symbols:
        tkr = normalize_ticker_for_lse(s)
        try:
            t = yf.Ticker(tkr)
        except Exception as e:
            rows.append({'S.No.': idx, 'Stock Name': tkr, 'Current Price': None, 'Price Change': None, '52W H/L': None, 'RSI': None, 'MACD': None, 'Overall, Tech Single': f'ERROR: {e}'})
            idx += 1
            continue

        # fetch history for indicators
        try:
            hist = t.history(period=f'{period_days}d', interval='1d', actions=False)
        except Exception:
            hist = pd.DataFrame()

        # current price & previous close
        try:
            current_price = None
            prev_close = None
            # use info if available
            info = t.info if hasattr(t, 'info') else {}
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price is None and not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
            if not hist.empty and len(hist['Close']) >= 2:
                prev_close = float(hist['Close'].iloc[-2])
            elif info.get('previousClose'):
                prev_close = info.get('previousClose')
        except Exception:
            current_price = None
            prev_close = None

        # Price change % vs prev_close
        price_change_pct = None
        try:
            if current_price is not None and prev_close is not None:
                price_change_pct = round(((float(current_price) - float(prev_close)) / abs(float(prev_close))) * 100.0, 2)
        except Exception:
            price_change_pct = None

        # 52-week high / low
        high52 = low52 = None
        try:
            if not hist.empty:
                # use last 252 trading days as ~1yr
                last_year = hist['Close'].dropna()
                if len(last_year) > 0:
                    high52 = round(float(last_year.max()), 2)
                    low52 = round(float(last_year.min()), 2)
            # fallback to info
            if (high52 is None or low52 is None) and info:
                high52 = high52 or info.get('fiftyTwoWeekHigh')
                low52 = low52 or info.get('fiftyTwoWeekLow')
        except Exception:
            high52 = low52 = None

        highlow_text = None
        if high52 is not None and low52 is not None:
            highlow_text = f"{high52}/{low52}"
        elif high52 is not None:
            highlow_text = f"{high52}/N/A"
        elif low52 is not None:
            highlow_text = f"N/A/{low52}"

        # RSI and MACD
        rsi_v = None
        macd_v = None
        try:
            if not hist.empty and 'Close' in hist:
                # ensure enough points
                closes = hist['Close'].dropna()
                if len(closes) >= 15:
                    rsi_v = compute_rsi(closes, period=14)
                    macd_v = compute_macd(closes)
        except Exception:
            rsi_v = macd_v = None

        # Overall technical single label
        overall = 'Neutral'
        try:
            if macd_v is None and rsi_v is None:
                overall = 'Unknown'
            else:
                # heuristics (tunable)
                r = rsi_v if rsi_v is not None else 50.0
                m = macd_v if macd_v is not None else 0.0
                if m > 0 and r >= 55:
                    overall = 'Greed'
                elif m < 0 and r <= 45:
                    overall = 'Fear'
                else:
                    overall = 'Neutral'
        except Exception:
            overall = 'Neutral'

        rows.append({
            'S.No.': idx,
            'Stock Name': s.upper(),
            'Current Price': None if current_price is None else round(float(current_price), 2),
            'Price Change': None if price_change_pct is None else f"{price_change_pct}%",
            '52W H/L': highlow_text,
            'RSI': None if rsi_v is None else round(float(rsi_v), 2),
            'MACD': None if macd_v is None else round(float(macd_v), 2),
            'Overall, Tech Single': overall,
        })
        idx += 1

    # Build Markdown table and CSV
    headers = ['S.No.', 'Stock Name', 'Current Price', 'Price Change', '52W H/L', 'RSI', 'MACD', 'Overall, Tech Single']
    md_lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for r in rows:
        vals = [str(r.get(h, '') or '') for h in headers]
        vals = [v.replace('|', '\\|') for v in vals]
        md_lines.append('| ' + ' | '.join(vals) + ' |')
    table_markdown = '\n'.join(md_lines)

    import io, csv as _csv
    csv_buf = io.StringIO()
    writer = _csv.DictWriter(csv_buf, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow({h: r.get(h, '') for h in headers})
    csv_text = csv_buf.getvalue()

    return json.dumps({'rows': rows, 'table_markdown': table_markdown, 'csv': csv_text, 'count': len(rows)}, default=str)


if __name__ == '__main__':
    print('Starting Yahoo Finance Technical MCP server...')
    tech_server.run(transport='stdio')
