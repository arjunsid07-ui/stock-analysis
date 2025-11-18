# fundamental_analysis_mcp_with_table.py
import json
from enum import Enum
import os
import yfinance as yf
import pandas as pd
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta

# -------------------------
# Config / helpers
# -------------------------
os.makedirs('data', exist_ok=True)

def save_to_json(filename, data):
    filepath = os.path.join('data', filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)
    return filepath

def save_to_file(filename, content):
    filepath = os.path.join('data', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

def pct_change(new, old):
    try:
        if old is None or pd.isna(old) or old == 0:
            return None
        return float((float(new) - float(old)) / abs(float(old)) * 100.0)
    except Exception:
        return None

def find_index_value_like(df, col, substrings):
    """
    Try to find a row in df whose index label contains any of substrings (case-insensitive).
    Returns value or None.
    """
    if df is None or df.empty:
        return None
    for s in substrings:
        s_lower = s.lower()
        for idx in df.index:
            if idx is None:
                continue
            label = str(idx).lower()
            if s_lower in label:
                val = df.iloc[df.index.get_loc(idx), col]
                return None if pd.isna(val) else val
    return None

def series_to_ordered_columns(df):
    """
    Convert a DataFrame of quarterly (or other) statements into a list of columns ordered newest->oldest.
    Each column returned as dict with 'date' and rows for each metric (index label -> value).
    """
    out = []
    if df is None or df.empty:
        return out
    # yfinance usually stores columns as timestamps with newest first, but not guaranteed.
    # We'll preserve order of columns as-is.
    for col in df.columns:
        date_str = col.strftime("%Y-%m-%d") if isinstance(col, pd.Timestamp) else str(col)
        rows = {}
        for idx, val in df[col].items():
            rows[str(idx)] = None if pd.isna(val) else val
        out.append({"date": date_str, **rows})
    return out

# -------------------------
# Enums & MCP init
# -------------------------
class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"

class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"

fundamental_analysis_server = FastMCP(
    "fundamental_analysis",
    instructions="""
# Fundamental Analysis MCP Server

Tools:
- get_stock_info
- get_financial_statement
- get_holder_info
- calculate_pead_score
- get_summary_table  <-- new: fetches all requested columns for a list of tickers and returns JSON + HTML table
""",
)

# -------------------------
# Existing tools (kept mostly intact)
# -------------------------
@fundamental_analysis_server.tool(
    name="get_stock_info",
    description="Get stock information for a given ticker symbol from yahoo finance."
)
async def get_stock_info(ticker: str) -> str:
    company = yf.Ticker(ticker)
    try:
        # check existence
        if getattr(company, "isin", None) is None and not company.info:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting stock information for {ticker}: {e}"
    info = company.info
    save_to_json(f"{ticker}_stock_info.json", info)
    return json.dumps(info, default=str)

@fundamental_analysis_server.tool(
    name="get_financial_statement",
    description="Get financial statement for a given ticker symbol from yahoo finance."
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    company = yf.Ticker(ticker)
    try:
        if getattr(company, "isin", None) is None and not company.info:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting financial statement for {ticker}: {e}"

    try:
        if financial_type == FinancialType.income_stmt:
            financial_statement = company.income_stmt
        elif financial_type == FinancialType.quarterly_income_stmt:
            financial_statement = company.quarterly_income_stmt
        elif financial_type == FinancialType.balance_sheet:
            financial_statement = company.balance_sheet
        elif financial_type == FinancialType.quarterly_balance_sheet:
            financial_statement = company.quarterly_balance_sheet
        elif financial_type == FinancialType.cashflow:
            financial_statement = company.cashflow
        elif financial_type == FinancialType.quarterly_cashflow:
            financial_statement = company.quarterly_cashflow
        else:
            return f"Error: invalid financial type {financial_type}."
    except Exception as e:
        return f"Error: retrieving statement for {ticker}: {e}"

    result = series_to_ordered_columns(financial_statement)
    save_to_json(f"{ticker}_{financial_type}_financial_statement.json", result)
    return json.dumps(result, default=str)

@fundamental_analysis_server.tool(
    name="get_holder_info",
    description="Get holder information for a given ticker symbol from yahoo finance."
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    company = yf.Ticker(ticker)
    try:
        if getattr(company, "isin", None) is None and not company.info:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting holder info for {ticker}: {e}"

    try:
        if holder_type == HolderType.major_holders:
            return company.major_holders.reset_index(names="metric").to_json(orient="records", default_handler=str)
        elif holder_type == HolderType.institutional_holders:
            return company.institutional_holders.to_json(orient="records", default_handler=str)
        elif holder_type == HolderType.mutualfund_holders:
            return company.mutualfund_holders.to_json(orient="records", date_format="iso", default_handler=str)
        elif holder_type == HolderType.insider_transactions:
            return company.insider_transactions.to_json(orient="records", date_format="iso", default_handler=str)
        elif holder_type == HolderType.insider_purchases:
            return company.insider_purchases.to_json(orient="records", date_format="iso", default_handler=str)
        elif holder_type == HolderType.insider_roster_holders:
            return company.insider_roster_holders.to_json(orient="records", date_format="iso", default_handler=str)
        else:
            return f"Error: invalid holder type {holder_type}."
    except Exception as e:
        return f"Error: fetching holder info for {ticker}: {e}"

# -------------------------
# PEAD calculation (improved)
# -------------------------
@fundamental_analysis_server.tool(
    name="calculate_pead_score",
    description="Calculate the Post-Earnings Announcement Drift score for a given ticker symbol."
)
async def calculate_pead_score(ticker: str, days_after: int = 30) -> str:
    """
    Returns JSON array of earnings dates with:
    - epsActual
    - epsEstimate (if available)
    - surprise (actual - estimate)
    - CAR_pct (cumulative abnormal return vs SPY over `days_after` trading days)
    """
    company = yf.Ticker(ticker)
    try:
        if getattr(company, "isin", None) is None and not company.info:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: calculating PEAD score for {ticker}: {e}"

    try:
        earnings_dates = getattr(company, "earnings_dates", None)
        if earnings_dates is None or (isinstance(earnings_dates, pd.DataFrame) and earnings_dates.empty):
            return json.dumps([])

        hist = company.history(period="2y", interval="1d")
        if hist is None or hist.empty:
            return json.dumps([])

        # SPY as benchmark
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="2y", interval="1d")

        pead_list = []
        # earnings_dates is a DataFrame: index are dates of the events
        # iterate over rows (sorted newest -> oldest)
        for idx in sorted(list(earnings_dates.index), reverse=True):
            # fallback: some entries have fields with estimate/actual nested - try to access
            row = earnings_dates.loc[idx]
            # Try to find epsActual and epsEstimate in several possible places
            epsActual = None
            epsEstimate = None
            # yfinance's earnings_dates rows often include 'epsActual' / 'epsEstimate' as columns
            for col in earnings_dates.columns:
                # pick eps columns heuristically
                lc = str(col).lower()
                val = row[col]
                if 'actual' in lc and val is not None and not pd.isna(val):
                    epsActual = val
                if 'estimate' in lc and val is not None and not pd.isna(val):
                    epsEstimate = val

            # if not found in earnings_dates, skip estimate but still compute price CAR (we'll return surprise as null)
            # find nearest trading day >= idx
            try:
                start_date = pd.to_datetime(idx).normalize()
            except Exception:
                continue

            # find position in hist
            # get first trading day on/after start_date
            try:
                hist_index = hist.index
                # hist index are timestamps; find nearest index >= start_date
                traded_dates = hist_index[hist_index >= start_date]
                if traded_dates.empty:
                    # no trading day after earnings in history slice; skip
                    continue
                start_trading = traded_dates[0]
                # build returns for days_after window
                end_idx_pos = hist_index.get_loc(start_trading) + days_after
                # safe bound
                if isinstance(end_idx_pos, slice):
                    end_idx_pos = end_idx_pos.stop
                end_idx = min(len(hist_index) - 1, end_idx_pos)
                # compute daily returns
                car = 0.0
                matched_days = 0
                for i in range(hist_index.get_loc(start_trading) + 1, end_idx + 1):
                    ticker_ret = (hist['Close'].iat[i] - hist['Close'].iat[i - 1]) / hist['Close'].iat[i - 1]
                    # find same date in spy_hist
                    dt = hist_index[i].normalize()
                    if dt in spy_hist.index:
                        spy_i = spy_hist.index.get_loc(dt)
                        spy_ret = (spy_hist['Close'].iat[spy_i] - spy_hist['Close'].iat[spy_i - 1]) / spy_hist['Close'].iat[spy_i - 1] if spy_i > 0 else 0.0
                    else:
                        # find nearest earlier spy date
                        spy_dates_after = spy_hist.index[spy_hist.index <= dt]
                        if spy_dates_after.empty or len(spy_dates_after) < 2:
                            continue
                        spy_i = spy_dates_after.get_loc(spy_dates_after[-1])
                        if spy_i == 0:
                            continue
                        spy_ret = (spy_hist['Close'].iat[spy_i] - spy_hist['Close'].iat[spy_i - 1]) / spy_hist['Close'].iat[spy_i - 1]
                    car += (ticker_ret - spy_ret)
                    matched_days += 1
                car_pct = car * 100.0 if matched_days > 0 else None
            except Exception:
                car_pct = None

            pead_item = {
                "date": pd.to_datetime(idx).strftime("%Y-%m-%d"),
                "epsActual": None if epsActual is None else float(epsActual),
                "epsEstimate": None if epsEstimate is None else float(epsEstimate),
                "surprise": None if (epsActual is None or epsEstimate is None) else float(epsActual) - float(epsEstimate),
                "CAR_pct": car_pct
            }
            pead_list.append(pead_item)

        save_to_json(f"{ticker}_pead_scores.json", pead_list)
        return json.dumps(pead_list, default=str)
    except Exception as e:
        return f"Error: calculating PEAD score for {ticker}: {e}"

# -------------------------
# NEW: Summary table tool (core of your request)
# -------------------------
@fundamental_analysis_server.tool(
    name="get_summary_table",
    description="""
Fetch a summary for multiple tickers and return JSON + an HTML table saved to data/.
Args:
    symbols: str -> comma-separated tickers e.g. "AAPL,MSFT,TSLA"
    pead_days: int -> days for PEAD CAR window (default 30)
""",
)
async def get_summary_table(symbols: str, pead_days: int = 30) -> str:
    """
    For each ticker returns:
    - symbol
    - companyName
    - peadScore (object: epsActual, epsEstimate, surprise, CAR_pct for most recent earning)
    - resultDate
    - forwardPE
    - salesYoY, salesQoQ
    - npYoY, npQoQ
    - ebitdaYoY, ebitdaQoQ
    - operatingCF (most recent quarterly)
    - freeCashFlow (from info if available)
    """

    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    summary_rows = []

    for t in tickers:
        try:
            comp = yf.Ticker(t)
            # simple existence check
            if not comp.info:
                summary_rows.append({"symbol": t, "error": "ticker not found / no info"})
                continue

            info = comp.info
            companyName = info.get("longName") or info.get("shortName") or t
            forwardPE = info.get("forwardPE") or info.get("forwardPE")  # may be None

            # Earnings/result date: try comp.earnings_dates
            resultDate = None
            try:
                ed = getattr(comp, "earnings_dates", None)
                if isinstance(ed, pd.DataFrame) and not ed.empty:
                    # pick the most recent date (max)
                    resultDate = pd.to_datetime(ed.index.max()).strftime("%Y-%m-%d")
                else:
                    # fallback: comp.calendar or comp.earnings
                    if getattr(comp, "calendar", None) is not None and hasattr(comp.calendar, 'T'):
                        try:
                            cal = comp.calendar
                            if not cal.empty:
                                # calendar may include "Earnings Date" column
                                # attempt to parse first entry
                                possible = list(cal.iloc[:, 0])
                                if possible:
                                    # pick first and format
                                    first = possible[0]
                                    if isinstance(first, (pd.Timestamp, datetime)):
                                        resultDate = pd.to_datetime(first).strftime("%Y-%m-%d")
                                    else:
                                        try:
                                            resultDate = pd.to_datetime(str(first)).strftime("%Y-%m-%d")
                                        except Exception:
                                            resultDate = None
                        except Exception:
                            resultDate = None
            except Exception:
                resultDate = None

            # Quarterly income statement parsing
            q_income = getattr(comp, "quarterly_income_stmt", None)
            sales_qoq = sales_yoy = np_qoq = np_yoy = ebitda_qoq = ebitda_yoy = None

            if q_income is not None and not q_income.empty:
                # convert columns to ordered list newest->oldest (yfinance typically newest first)
                cols = list(q_income.columns)
                # ensure we have numeric values by locating specific metrics
                # revenue candidates:
                revenue_candidates = ["total revenue", "totalRevenue", "revenue", "totalRevenueNetOfInterestExpense"]
                net_candidates = ["net income", "netincome", "net income common stockholders", "net income available to common shareholders"]
                ebitda_candidates = ["ebitda", "ebit", "ebitda (earnings before interest, taxes, depreciation and amortization)"]

                # helper to get value given column index and candidate keys
                def get_val_for_col(col_idx, candidates):
                    # attempt by exact df index matching first
                    try:
                        for cand in candidates:
                            # check direct label equality
                            for idxlabel in q_income.index:
                                if str(idxlabel).lower() == cand.lower():
                                    v = q_income.iloc[q_income.index.get_loc(idxlabel), col_idx]
                                    return None if pd.isna(v) else v
                        # fallback: contains
                        for cand in candidates:
                            found = find_index_value_like(q_income, col_idx, [cand])
                            if found is not None:
                                return found
                    except Exception:
                        return None
                    return None

                # newest column index 0, previous 1, 4-quarters-ago index 3 (if present)
                try:
                    newest_idx = 0
                    prev_idx = 1 if len(cols) > 1 else None
                    last_year_idx = 3 if len(cols) > 3 else None

                    rev_new = get_val_for_col(newest_idx, revenue_candidates)
                    rev_prev = get_val_for_col(prev_idx, revenue_candidates) if prev_idx is not None else None
                    rev_last_year = get_val_for_col(last_year_idx, revenue_candidates) if last_year_idx is not None else None

                    ni_new = get_val_for_col(newest_idx, net_candidates)
                    ni_prev = get_val_for_col(prev_idx, net_candidates) if prev_idx is not None else None
                    ni_last_year = get_val_for_col(last_year_idx, net_candidates) if last_year_idx is not None else None

                    eb_new = get_val_for_col(newest_idx, ebitda_candidates)
                    eb_prev = get_val_for_col(prev_idx, ebitda_candidates) if prev_idx is not None else None
                    eb_last_year = get_val_for_col(last_year_idx, ebitda_candidates) if last_year_idx is not None else None

                    sales_qoq = pct_change(rev_new, rev_prev)
                    sales_yoy = pct_change(rev_new, rev_last_year)
                    np_qoq = pct_change(ni_new, ni_prev)
                    np_yoy = pct_change(ni_new, ni_last_year)
                    ebitda_qoq = pct_change(eb_new, eb_prev)
                    ebitda_yoy = pct_change(eb_new, eb_last_year)
                except Exception:
                    pass

            # Operating cashflow: from quarterly_cashflow or cashflow
            operating_cf = None
            q_cf = getattr(comp, "quarterly_cashflow", None)
            if q_cf is not None and not q_cf.empty:
                # try to find label containing 'operat' and 'cash'
                operating_cf = None
                for idxlabel in q_cf.index:
                    label = str(idxlabel).lower()
                    if 'operat' in label and 'cash' in label:
                        val = q_cf.iloc[q_cf.index.get_loc(idxlabel), 0] if len(q_cf.columns) > 0 else None
                        operating_cf = None if pd.isna(val) else val
                        break
            # fallback to cashflow['Free Cash Flow'] or info
            free_cashflow_val = info.get("freeCashflow") or info.get("freeCashFlow") or None

            # PEAD: call the calculate_pead_score tool logic inline to avoid additional MCP call overhead
            try:
                # use same approach as calculate_pead_score but only for most recent earnings row
                pead_obj = None
                ed = getattr(comp, "earnings_dates", None)
                if isinstance(ed, pd.DataFrame) and not ed.empty:
                    # pick the most recent
                    most_recent_date = sorted(list(ed.index))[-1]
                    # find epsActual/estimate if present
                    row = ed.loc[most_recent_date]
                    epsActual = None
                    epsEstimate = None
                    for coln in ed.columns:
                        lc = str(coln).lower()
                        try:
                            val = row[coln]
                        except Exception:
                            val = None
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            continue
                        if 'actual' in lc:
                            epsActual = val
                        if 'estimate' in lc:
                            epsEstimate = val
                    # compute CAR over pead_days
                    hist = comp.history(period="2y", interval="1d")
                    spy = yf.Ticker("SPY").history(period="2y", interval="1d")
                    car_pct = None
                    if hist is not None and not hist.empty and spy is not None and not spy.empty:
                        # start from first trading day on or after most_recent_date
                        start_date = pd.to_datetime(most_recent_date).normalize()
                        traded = hist.index[hist.index >= start_date]
                        if not traded.empty:
                            start_idx = hist.index.get_loc(traded[0])
                            matched = 0
                            car = 0.0
                            for i in range(start_idx + 1, min(len(hist.index), start_idx + pead_days + 1)):
                                try:
                                    t_ret = (hist['Close'].iat[i] - hist['Close'].iat[i - 1]) / hist['Close'].iat[i - 1]
                                    dt = hist.index[i].normalize()
                                    # find same date in spy
                                    if dt in spy.index:
                                        spy_i = spy.index.get_loc(dt)
                                        if spy_i > 0:
                                            spy_ret = (spy['Close'].iat[spy_i] - spy['Close'].iat[spy_i - 1]) / spy['Close'].iat[spy_i - 1]
                                        else:
                                            spy_ret = 0.0
                                    else:
                                        # fallback: find nearest earlier spy date
                                        spy_dates_le = spy.index[spy.index <= dt]
                                        if spy_dates_le.empty:
                                            continue
                                        last_spy = spy_dates_le[-1]
                                        spy_i = spy.index.get_loc(last_spy)
                                        if spy_i == 0:
                                            continue
                                        spy_ret = (spy['Close'].iat[spy_i] - spy['Close'].iat[spy_i - 1]) / spy['Close'].iat[spy_i - 1]
                                    car += (t_ret - spy_ret)
                                    matched += 1
                                except Exception:
                                    continue
                            if matched > 0:
                                car_pct = car * 100.0
                    pead_obj = {
                        "most_recent_earnings_date": pd.to_datetime(most_recent_date).strftime("%Y-%m-%d"),
                        "epsActual": None if epsActual is None else float(epsActual),
                        "epsEstimate": None if epsEstimate is None else float(epsEstimate),
                        "surprise": None if (epsActual is None or epsEstimate is None) else float(epsActual) - float(epsEstimate),
                        "CAR_pct": car_pct
                    }
                else:
                    pead_obj = None
            except Exception:
                pead_obj = None

            row = {
                "symbol": t,
                "companyName": companyName,
                "peadScore": pead_obj,
                "resultDate": resultDate,
                "forwardPE": forwardPE,
                "salesYoY": sales_yoy,
                "salesQoQ": sales_qoq,
                "npYoY": np_yoy,
                "npQoQ": np_qoq,
                "ebitdaYoY": ebitda_yoy,
                "ebitdaQoQ": ebitda_qoq,
                "operatingCF": operating_cf,
                "freeCashFlow": free_cashflow_val
            }
            summary_rows.append(row)
        except Exception as e:
            summary_rows.append({"symbol": t, "error": str(e)})

    # Save JSON
    json_path = save_to_json(f"summary_{'_'.join(tickers)}.json", summary_rows)

    # Build simple responsive HTML table
    headers = [
        "Symbol", "Company Name", "PEAD CAR % (most recent)", "Result Date", "Forward PE",
        "Sales YoY %", "Sales QoQ %", "NP YoY %", "NP QoQ %", "EBITDA YoY %", "EBITDA QoQ %",
        "Operating CF", "Free Cash Flow"
    ]

    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.2f}"
        return str(x)

    rows_html = []
    for r in summary_rows:
        car_val = "-"
        if r.get("peadScore") and r["peadScore"].get("CAR_pct") is not None:
            try:
                car_val = f"{float(r['peadScore']['CAR_pct']):.2f}%"
            except Exception:
                car_val = str(r['peadScore']['CAR_pct'])
        elif r.get("peadScore") and "surprise" in r["peadScore"] and r["peadScore"]["surprise"] is not None:
            car_val = f"surprise:{r['peadScore']['surprise']}"
        row_vals = [
            r.get("symbol", "-"),
            r.get("companyName", "-"),
            car_val,
            r.get("resultDate", "-"),
            fmt(r.get("forwardPE")),
            fmt(r.get("salesYoY")),
            fmt(r.get("salesQoQ")),
            fmt(r.get("npYoY")),
            fmt(r.get("npQoQ")),
            fmt(r.get("ebitdaYoY")),
            fmt(r.get("ebitdaQoQ")),
            fmt(r.get("operatingCF")),
            fmt(r.get("freeCashFlow"))
        ]
        row_cells = "".join(f"<td>{cell}</td>" for cell in row_vals)
        rows_html.append(f"<tr>{row_cells}</tr>")

    html_table = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Summary Table - {' '.join(tickers)}</title>
      <style>
        body{{font-family:Arial,Helvetica,sans-serif;margin:20px}}
        table{{border-collapse:collapse;width:100%}}
        th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
        th{{background:#f4f4f4}}
        tr:nth-child(even){{background:#fbfbfb}}
        .mono{{font-family:monospace}}
      </style>
    </head>
    <body>
      <h2>Summary Table for {', '.join(tickers)}</h2>
      <table>
        <thead>
          <tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
      <p>Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</p>
    </body>
    </html>
    """

    html_path = save_to_file(f"summary_table_{'_'.join(tickers)}.html", html_table)

    # Return JSON with both paths
    result = {
        "json_path": json_path,
        "html_path": html_path,
        "rows": summary_rows
    }
    # Save a top-level file referencing both
    save_to_json(f"summary_index_{'_'.join(tickers)}.json", result)
    return json.dumps(result, default=str)

# -------------------------
# run server if executed
# -------------------------
if __name__ == "__main__":
    print("Starting Fundamental Analysis MCP server (with summary table)...")
    fundamental_analysis_server.run(transport="stdio")
