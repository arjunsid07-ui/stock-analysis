 # Code generated via "Slingshot" 
import json
import yfinance as yf
import pandas as pd
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server for news and recommendations
news_recommendations_server = FastMCP(
    "news_recommendations",
    instructions="""
# News and Recommendations MCP Server

This server is used to get news and recommendations about a given ticker symbol from yahoo finance.

Available tools:
- get_yahoo_finance_news: Get news for a given ticker symbol from yahoo finance.
- get_recommendations: Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.
""",
)

def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

@news_recommendations_server.tool(
    name="get_yahoo_finance_news",
    description="""Get news for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get news for, e.g. "AAPL"
""",
)
async def get_yahoo_finance_news(ticker: str) -> str:
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting news for {ticker}: {e}"

    try:
        news = company.news
    except Exception as e:
        return f"Error: getting news for {ticker}: {e}"

    news_list = []
    for news in company.news:
        if news.get("content", {}).get("contentType", "") == "STORY":
            title = news.get("content", {}).get("title", "")
            summary = news.get("content", {}).get("summary", "")
            description = news.get("content", {}).get("description", "")
            url = news.get("content", {}).get("canonicalUrl", {}).get("url", "")
            news_list.append(
                {"title": title, "summary": summary, "description": description, "url": url}
            )
    if not news_list:
        return f"No news found for company that searched with {ticker} ticker."
    
    save_to_json(f"{ticker}_news.json", news_list)
    return json.dumps(news_list)

@news_recommendations_server.tool(
    name="get_recommendations",
    description="""Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get recommendations for, e.g. "AAPL"
    recommendation_type: str
        The type of recommendation to get.
    months_back: int
        The number of months back to get upgrades/downgrades for, default is 12.
""",
)
async def get_recommendations(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return f"Company ticker {ticker} not found."
    except Exception as e:
        return f"Error: getting recommendations for {ticker}: {e}"
    try:
        if recommendation_type == "recommendations":
            recommendations = company.recommendations.to_json(orient="records")
            save_to_json(f"{ticker}_recommendations.json", json.loads(recommendations))
            return recommendations
        elif recommendation_type == "upgrades_downgrades":
            upgrades_downgrades = company.upgrades_downgrades.reset_index()
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades_downgrades = upgrades_downgrades[
                upgrades_downgrades["GradeDate"] >= cutoff_date
            ]
            upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
            latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
            json_data = latest_by_firm.to_json(orient="records", date_format="iso")
            save_to_json(f"{ticker}_upgrades_downgrades.json", json.loads(json_data))
            return json_data
    except Exception as e:
        return f"Error: getting recommendations for {ticker}: {e}"

if __name__ == "__main__":
    print("Starting News and Recommendations MCP server...")
    news_recommendations_server.run(transport="stdio")