 # Code generated via "Slingshot" 
import yfinance as yf
import pandas as pd
from mcp.server.fastmcp import FastMCP
import json

# Initialize FastMCP server for financial analysis
financial_analysis_server = FastMCP(
    "financial_analysis",
    instructions="""
# Financial Analysis MCP Server

This server is used to get financial analysis information about a given ticker symbol from yahoo finance.

Available tools:
- get_historical_stock_prices: Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
- get_stock_actions: Get stock dividends and stock splits for a given ticker symbol from yahoo finance.
""",
)

def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

@financial_analysis_server.tool(
    name="get_historical_stock_prices",
    description="""Get historical stock prices for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
    period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
""",
)
async def get_historical_stock_prices(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting historical stock prices for {ticker}: {e}"

    hist_data = company.history(period=period, interval=interval)
    hist_data = hist_data.reset_index(names="Date")
    json_data = hist_data.to_json(orient="records", date_format="iso")
    save_to_json(f"{ticker}_historical_prices.json", json.loads(json_data))
    return json_data

@financial_analysis_server.tool(
    name="get_stock_actions",
    description="""Get stock dividends and stock splits for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get stock actions for, e.g. "AAPL"
""",
)
async def get_stock_actions(ticker: str) -> str:
    try:
        company = yf.Ticker(ticker)
    except Exception as e:
        return f"Error: getting stock actions for {ticker}: {e}"
    actions_df = company.actions
    actions_df = actions_df.reset_index(names="Date")
    json_data = actions_df.to_json(orient="records", date_format="iso")
    save_to_json(f"{ticker}_stock_actions.json", json.loads(json_data))
    return json_data

if __name__ == "__main__":
    print("Starting Financial Analysis MCP server...")
    financial_analysis_server.run(transport="stdio")