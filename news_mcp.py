# news_mcp.py
import os
import json
import re
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# Optional NLP imports (graceful fallback)
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon")
    except Exception:
        try:
            nltk.download("vader_lexicon")
        except Exception:
            pass
    NLTK_VADER_AVAILABLE = True
except Exception:
    SentimentIntensityAnalyzer = None
    NLTK_VADER_AVAILABLE = False

# Create server object BEFORE decorators
news_server = FastMCP(
    "serp_news",
    instructions="""
SerpAPI (HTTP) based news MCP: fetch news via SerpAPI Google News endpoint using requests,
compute relevance, sentiment, novelty, source credibility, market noise (from yfinance),
and return per-article signal + aggregates. Provide SERPAPI_KEY via env or param.
""",
)


# -------------------------
# Utilities & heuristics
# -------------------------
# Small default credibility map. Replace/extend for production.
SOURCE_CREDIBILITY_MAP = {
    "ft.com": 0.95,
    "reuters.com": 0.95,
    "bloomberg.com": 0.95,
    "wsj.com": 0.92,
    "economist.com": 0.9,
    "theguardian.com": 0.85,
    "bbc.co.uk": 0.85,
    "cnbc.com": 0.85,
    "marketwatch.com": 0.8,
    "yahoo.com": 0.75,
}


def domain_from_url(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"https?://([^/]+)/?", url)
    if not m:
        return url.lower()
    domain = m.group(1).lower()
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def get_source_credibility(url_or_provider: str) -> float:
    dom = domain_from_url(url_or_provider)
    return float(SOURCE_CREDIBILITY_MAP.get(dom, 0.5))


def compute_sentiment_score(text: str) -> float:
    """Return sentiment in 0..1. Use VADER if available; fallback heuristic otherwise."""
    if not text or not text.strip():
        return 0.5
    if NLTK_VADER_AVAILABLE:
        try:
            sia = SentimentIntensityAnalyzer()
            comp = sia.polarity_scores(text).get("compound", 0.0)  # -1..1
            return (comp + 1.0) / 2.0
        except Exception:
            pass

    # fallback heuristic
    text_l = text.lower()
    pos_words = ["gain", "up", "strong", "beat", "positive", "upgrade", "outperform", "record"]
    neg_words = ["loss", "drop", "down", "weak", "miss", "negative", "downgrade", "lawsuit"]
    score = 0.0
    for w in pos_words:
        if w in text_l:
            score += 1.0
    for w in neg_words:
        if w in text_l:
            score -= 1.0
    score = max(-5, min(5, score))
    return (score + 5) / 10.0


def compute_relevance_score(text: str, keywords: List[str]) -> float:
    if not text or not keywords:
        return 0.0
    text_l = text.lower()
    keywords = [k.lower() for k in keywords if k]
    found_cnt = sum(1 for k in keywords if k in text_l)
    presence_ratio = found_cnt / max(1, len(keywords))
    occ = sum(text_l.count(k) for k in keywords)
    density = min(1.0, occ / max(1, len(text.split()) / 20))
    relevance = 0.6 * presence_ratio + 0.4 * density
    return float(max(0.0, min(1.0, relevance)))


def compute_novelty_scores(items: List[Dict]) -> List[float]:
    """
    Basic novelty: 1 - max_similarity. We use TF-IDF if sklearn is available; otherwise
    fallback to duplicate-title heuristic.
    """
    texts = []
    for it in items:
        txt = " ".join(filter(None, [it.get("title", ""), it.get("snippet", ""), it.get("summary", "")]))
        texts.append(txt or "")

    n = len(texts)
    if n == 0:
        return []

    # Try sklearn tfidf/cosine if available
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        tf = TfidfVectorizer(stop_words="english", max_features=3000)
        X = tf.fit_transform(texts)
        sim = cosine_similarity(X)
        nov = []
        for i in range(n):
            sim_row = sim[i].copy()
            sim_row[i] = -1.0
            max_sim = float(np.max(sim_row)) if n > 1 else 0.0
            nov.append(float(max(0.0, min(1.0, 1.0 - max_sim))))
        return nov
    except Exception:
        # fallback: duplicate-title heuristic
        seen = {}
        for t in texts:
            key = t.strip().lower()
            seen[key] = seen.get(key, 0) + 1
        nov = []
        for t in texts:
            key = t.strip().lower()
            score = 1.0 / seen.get(key, 1)
            score = min(1.0, score + min(0.2, len(t.split()) / 100.0))
            nov.append(score)
        return nov


def compute_market_noise(ticker: str, lookback_days: int = 7) -> float:
    epsilon = 1e-4
    try:
        if not ticker:
            return epsilon
        t = yf.Ticker(ticker)
        df = t.history(period=f"{max(lookback_days,3)}d", interval="1d")
        if df is None or df.empty or "Close" not in df:
            return epsilon
        closes = df["Close"].astype(float).dropna()
        if len(closes) < 2:
            return epsilon
        returns = closes.pct_change().dropna()
        noise = float(np.std(returns))
        return max(noise, epsilon)
    except Exception:
        return epsilon


# -------------------------
# SerpAPI (HTTP) helper
# -------------------------
def serpapi_search_news(
    query: str,
    serp_api_key: str,
    num_results: int = 20,
    language: str = "en",
    country: str = "us",
) -> List[Dict]:
    """
    Use SerpAPI HTTP endpoint to fetch Google News results (tbm=nws).
    Returns list of items with keys: title, snippet, url, source, published_at (ISO when parseable).
    """
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google",
        "tbm": "nws",
        "api_key": serp_api_key,
        "hl": language,
        "gl": country,
        "num": int(num_results),
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"SerpAPI HTTP {r.status_code}: {r.text}")
    data = r.json()

    results = []
    # Common key in SerpAPI responses
    for entry in data.get("news_results", [])[:num_results]:
        title = entry.get("title") or ""
        snippet = entry.get("snippet") or entry.get("summary") or ""
        link = entry.get("link") or entry.get("source_url") or ""
        source = entry.get("source") or entry.get("news_source") or ""
        published_at = None
        # SerpAPI sometimes uses 'date' or 'published' or human-friendly strings
        for cand in ("date", "published", "time", "published_at"):
            if entry.get(cand):
                v = entry.get(cand)
                try:
                    dt = pd.to_datetime(v, utc=True, errors="coerce")
                    if not pd.isna(dt):
                        published_at = dt.isoformat()
                    else:
                        published_at = str(v)
                except Exception:
                    published_at = str(v)
                break
        results.append(
            {
                "title": title,
                "snippet": snippet,
                "url": link,
                "source": source,
                "published_at": published_at,
            }
        )
    # Fallback parsing of other fields if 'news_results' not present
    if not results:
        for key in ("organic_results", "results", "news"):
            if key in data and isinstance(data[key], list):
                for entry in data[key][:num_results]:
                    title = entry.get("title", "")
                    snippet = entry.get("snippet", "")
                    link = entry.get("link", "") or entry.get("url", "")
                    source = entry.get("source", "") or entry.get("site", "")
                    published_at = entry.get("date") or entry.get("published")
                    results.append(
                        {"title": title, "snippet": snippet, "url": link, "source": source, "published_at": published_at}
                    )
                break
    return results


# -------------------------
# MCP Tool
# -------------------------
@news_server.tool(
    name="get_serp_news_and_signals",
    description="""
Fetch news via SerpAPI (HTTP) and compute relevance, sentiment, novelty, source credibility and market noise.
Args:
    target: str (ticker like 'VOD' or 'VOD.L' or keywords)
    serp_api_key: str (optional; if omitted uses env SERPAPI_KEY)
    num_results: int (default 25)
    interval_hours: int (default 2)
    lookback_days_for_noise: int (default 7)
""",
)
async def get_serp_news_and_signals(
    target: str,
    serp_api_key: Optional[str] = None,
    num_results: int = 25,
    interval_hours: int = 2,
    lookback_days_for_noise: int = 7,
    country: str = "us",
    language: str = "en",
) -> str:
    if not target or not target.strip():
        return json.dumps({"error": "target required"})

    serp_api_key = serp_api_key or os.getenv("SERPAPI_KEY") or ""
    if not serp_api_key:
        return json.dumps({"error": "SerpAPI key required (env SERPAPI_KEY or serp_api_key argument)."})

    target_raw = target.strip()
    ticker_candidate = target_raw.upper()
    ticker_for_price = ""
    if re.match(r"^[A-Z]{1,5}(\.L)?$", ticker_candidate):
        ticker_for_price = ticker_candidate if ticker_candidate.endswith(".L") else ticker_candidate + ".L"

    query = target_raw + " stock" if ticker_for_price else target_raw

    try:
        articles = serpapi_search_news(query=query, serp_api_key=serp_api_key, num_results=num_results, country=country, language=language)
    except Exception as e:
        return json.dumps({"error": f"SerpAPI search failed: {e}"})

    if not articles:
        return json.dumps({"target": target_raw, "message": "No articles returned from SerpAPI for this target."})

    now = datetime.now(timezone.utc)
    parsed = []
    for a in articles:
        published_parsed = None
        p = a.get("published_at")
        if p:
            try:
                dt = pd.to_datetime(p, utc=True, errors="coerce")
                if not pd.isna(dt):
                    published_parsed = dt.to_pydatetime()
            except Exception:
                published_parsed = None
        parsed.append(
            {
                "title": a.get("title"),
                "snippet": a.get("snippet"),
                "url": a.get("url"),
                "source": a.get("source"),
                "published_at": published_parsed.isoformat() if published_parsed else a.get("published_at"),
                "_published_dt": published_parsed,
            }
        )

    interval_delta = timedelta(hours=max(1, int(interval_hours)))
    recent_threshold = now - interval_delta
    general_news = parsed
    recent_interval_news = [it for it in general_news if it["_published_dt"] and it["_published_dt"] >= recent_threshold]

    keywords = [target_raw]
    if "." in target_raw:
        keywords.append(target_raw.split(".")[0])

    novelty_scores = compute_novelty_scores(general_news)
    market_noise = compute_market_noise(ticker_for_price, lookback_days=lookback_days_for_noise)

    scored = []
    for idx, art in enumerate(general_news):
        text_combined = " ".join(filter(None, [art.get("title", ""), art.get("snippet", "")]))
        relevance = compute_relevance_score(text_combined, keywords)
        sentiment = compute_sentiment_score(text_combined)
        novelty = float(novelty_scores[idx]) if idx < len(novelty_scores) else 0.5
        cred = get_source_credibility(art.get("url") or art.get("source") or "")
        try:
            numerator = float(relevance) * float(sentiment) * float(novelty) * float(cred)
            signal = numerator / float(max(market_noise, 1e-8))
        except Exception:
            signal = 0.0

        scored.append(
            {
                "title": art.get("title"),
                "snippet": art.get("snippet"),
                "url": art.get("url"),
                "source": art.get("source"),
                "published_at": art.get("published_at"),
                "relevance": round(float(relevance), 4),
                "sentiment": round(float(sentiment), 4),
                "novelty": round(float(novelty), 4),
                "source_credibility": round(float(cred), 4),
                "signal_strength": float(signal),
            }
        )

    recent_urls = {it.get("url") for it in recent_interval_news}
    recent_scored = [s for s in scored if (s.get("url") in recent_urls) or (s.get("published_at") in [r.get("published_at") for r in recent_interval_news])]

    signals = [s["signal_strength"] for s in scored if s.get("signal_strength") is not None]
    agg = {"mean": 0.0, "median": 0.0, "count": 0}
    if signals:
        agg["mean"] = float(np.mean(signals))
        agg["median"] = float(np.median(signals))
        agg["count"] = len(signals)

    response = {
        "target": target_raw,
        "ticker_for_price": ticker_for_price,
        "market_noise": float(market_noise),
        "general_news": scored,
        "recent_interval_news": recent_scored,
        "aggregate_signal": agg,
        "meta": {
            "serpapi_used": True,
            "nlp_engine": "nltk_vader" if NLTK_VADER_AVAILABLE else "heuristic",
            "novelty_engine": "tfidf_if_available",
            "num_results_requested": num_results,
        },
    }
    return json.dumps(response)


if __name__ == "__main__":
    print("Starting SerpAPI News MCP server...")
    news_server.run(transport="stdio")
