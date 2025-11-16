 # Code generated via "Slingshot" 
import json
from enum import Enum
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# Define an enum for the type of financial statement
class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"

# Initialize FastMCP server for fundamental analysis
fundamental_analysis_server = FastMCP(
    "fundamental_analysis",
    instructions="""
# Fundamental Analysis MCP Server

This server is used to get fundamental analysis information about a given ticker symbol from yahoo finance.

Available tools:
- get_stock_info: Get stock information for a given ticker symbol from yahoo finance.
- get_financial_statement: Get financial statement for a given ticker symbol from yahoo finance.
""",
)

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

    return json.dumps(result)

if __name__ == "__main__":
    print("Starting Fundamental Analysis MCP server...")
    fundamental_analysis_server.run(transport="stdio")