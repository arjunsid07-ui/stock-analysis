import json
from enum import Enum
import os
import yfinance as yf
import pandas as pd
from mcp.server.fastmcp import FastMCP

# Define an enum for the type of financial statement
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

# Initialize FastMCP server for fundamental analysis
fundamental_analysis_server = FastMCP(
    "fundamental_analysis",
    instructions="""
# Fundamental Analysis MCP Server

This server is used to get fundamental analysis information about a given ticker symbol from yahoo finance.

Available tools:
- get_stock_info: Get stock information for a given ticker symbol from yahoo finance. Include the following information: Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.
- get_financial_statement: Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
- calculate_pead_score: Calculate the Post-Earnings Announcement Drift score for a given ticker symbol.
- get_holder_info: Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
""",
)

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

def save_to_json(filename, data):
    # CHANGE: Save JSON files in the 'data' directory
    filepath = os.path.join('data', filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

@fundamental_analysis_server.tool(
    name="get_stock_info",
    description="""Get stock information for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get information for, e.g. "AAPL"
""",
)
async def get_stock_info(ticker: str) -> str:
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting stock information for {ticker}: {e}"
    info = company.info
    save_to_json(f"{ticker}_stock_info.json", info)
    return json.dumps(info)

@fundamental_analysis_server.tool(
    name="get_financial_statement",
    description="""Get financial statement for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get financial statement for, e.g. "AAPL"
    financial_type: str
        The type of financial statement to get.
""",
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting financial statement for {ticker}: {e}"

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

    result = []
    for column in financial_statement.columns:
        date_str = column.strftime("%Y-%m-%d") if isinstance(column, pd.Timestamp) else str(column)
        date_obj = {"date": date_str}
        for index, value in financial_statement[column].items():
            date_obj[index] = None if pd.isna(value) else value
        result.append(date_obj)

    save_to_json(f"{ticker}_{financial_type}_financial_statement.json", result)
    return json.dumps(result)

@fundamental_analysis_server.tool(
    name="get_holder_info",
    description="""Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.

Args:
    ticker: str
        The ticker symbol of the stock to get holder information for, e.g. "AAPL"
    holder_type: str
        The type of holder information to get. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
""",
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    """Get holder information for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting holder info for {ticker}: {e}")
        return f"Error: getting holder info for {ticker}: {e}"

    if holder_type == HolderType.major_holders:
        return company.major_holders.reset_index(names="metric").to_json(orient="records")
    elif holder_type == HolderType.institutional_holders:
        return company.institutional_holders.to_json(orient="records")
    elif holder_type == HolderType.mutualfund_holders:
        return company.mutualfund_holders.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_transactions:
        return company.insider_transactions.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_purchases:
        return company.insider_purchases.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_roster_holders:
        return company.insider_roster_holders.to_json(orient="records", date_format="iso")
    else:
        return f"Error: invalid holder type {holder_type}. Please use one of the following: {HolderType.major_holders}, {HolderType.institutional_holders}, {HolderType.mutualfund_holders}, {HolderType.insider_transactions}, {HolderType.insider_purchases}, {HolderType.insider_roster_holders}."



@fundamental_analysis_server.tool(
    name="calculate_pead_score",
    description="""Calculate the Post-Earnings Announcement Drift score for a given ticker symbol.

Args:
    ticker: str
        The ticker symbol of the stock to calculate PEAD score for, e.g. "AAPL"
    days_after: int
        The number of days after the earnings announcement to calculate the drift.
""",
)
async def calculate_pead_score(ticker: str, days_after: int = 5) -> str:
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: calculating PEAD score for {ticker}: {e}"

    try:
        earnings_dates = company.earnings_dates
        if earnings_dates.empty:
            return f"No earnings data found for {ticker}."

        # Get historical prices
        hist_data = company.history(period="1y")
        pead_scores = []

        for date in earnings_dates.index:
            if date in hist_data.index:
                pre_earnings_price = hist_data.loc[date, 'Close']
                post_earnings_date = date + pd.DateOffset(days=days_after)
                if post_earnings_date in hist_data.index:
                    post_earnings_price = hist_data.loc[post_earnings_date, 'Close']
                    pead_score = ((post_earnings_price - pre_earnings_price) / pre_earnings_price) * 100
                    pead_scores.append({"date": date.strftime("%Y-%m-%d"), "pead_score": pead_score})

        save_to_json(f"{ticker}_pead_scores.json", pead_scores)
        return json.dumps(pead_scores)
    except Exception as e:
        return f"Error: calculating PEAD score for {ticker}: {e}"

if __name__ == "__main__":
    print("Starting Fundamental Analysis MCP server...")
    fundamental_analysis_server.run(transport="stdio")