# news_mcp_full.py
"""
SerpAPI (HTTP) based news MCP - UK/London focused search (gl=uk, hl=en-GB).
Enforces LSE normalization for tickers (adds .L). Computes relevance, sentiment, novelty,
source credibility and market noise (from yfinance). Returns raw signal + percentage (0-100)
and an overall aggregated analysis (weighted by source credibility and recency).

Drop this file into the directory your MCP launcher points to and run it. Provide SERPAPI_KEY
via environment variable or pass as `serp_api_key` param when calling the tool.
"""

import os
import json
import re
import math
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# Optional nltk VADER sentiment fallback
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


# Create server BEFORE decorators
news_server = FastMCP(
    "serp_news",
    instructions="""
SerpAPI (HTTP) based news MCP - UK/London focused search (gl=uk, hl=en-GB).
Enforces LSE normalization for tickers (adds .L). Computes relevance, sentiment, novelty,
source credibility and market noise (from yfinance). Returns raw signal + percentage (0-100)
and an overall aggregated analysis (weighted by source credibility and recency).
""",
)


# -------------------------
# Utilities & heuristics
# -------------------------
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
    texts = []
    for it in items:
        txt = " ".join(filter(None, [it.get("title", ""), it.get("snippet", ""), it.get("summary", "")]))
        texts.append(txt or "")
    n = len(texts)
    if n == 0:
        return []
    # Try sklearn TF-IDF if available
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
# SerpAPI (HTTP) helper - UK focused by default (gl=uk, hl=en-GB)
# -------------------------
def serpapi_search_news(
    query: str,
    serp_api_key: str,
    num_results: int = 20,
    language: str = "en-GB",
    country: str = "uk",
) -> List[Dict]:
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
    for entry in data.get("news_results", [])[:num_results]:
        title = entry.get("title") or ""
        snippet = entry.get("snippet") or entry.get("summary") or ""
        link = entry.get("link") or entry.get("source_url") or ""
        source = entry.get("source") or entry.get("news_source") or ""
        published_at = None
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
    if not results:
        for key in ("organic_results", "results", "news"):
            if key in data and isinstance(data[key], list):
                for entry in data[key][:num_results]:
                    title = entry.get("title", "")
                    snippet = entry.get("snippet", "")
                    link = entry.get("link", "") or entry.get("url", "")
                    source = entry.get("source", "") or entry.get("site", "")
                    published_at = entry.get("date") or entry.get("published")
                    results.append({"title": title, "snippet": snippet, "url": link, "source": source, "published_at": published_at})
                break
    return results


# -------------------------
# MCP Tool - LSE enforcement + UK news focus
# -------------------------
@news_server.tool(
    name="get_serp_news_and_signals",
    description="""
Fetch UK-focused news via SerpAPI for a ticker or free-text sector/keyword,
compute relevance, sentiment, novelty, credibility and market noise, and return per-article
raw signal and percentage (0-100). Ticker-like targets are normalized to .L (LSE).
""",
)
async def get_serp_news_and_signals(
    target: str,
    serp_api_key: Optional[str] = None,
    num_results: int = 25,
    interval_hours: int = 2,
    lookback_days_for_noise: int = 7,
    country: str = "uk",
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
    # Detect simple ticker-like strings and normalize to .L
    if re.match(r"^[A-Z]{1,5}(\\.L)?$", ticker_candidate):
        ticker_for_price = ticker_candidate if ticker_candidate.endswith(".L") else ticker_candidate + ".L"
    else:
        # For free-text sector queries we still keep ticker_for_price empty but will do UK news search
        ticker_for_price = ""

    # Build query - if ticker-like, search "<ticker> stock UK" to bias results to UK coverage
    query = (target_raw + " stock UK") if ticker_for_price else target_raw

    try:
        articles = serpapi_search_news(query=query, serp_api_key=serp_api_key, num_results=num_results, country=country, language=language)
    except Exception as e:
        return json.dumps({"error": f"SerpAPI search failed: {e}"})

    if not articles:
        return json.dumps({"target": target_raw, "message": "No articles returned from SerpAPI for this target."})

    now = datetime.now(timezone.utc)
    parsed_articles = []
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
        parsed_articles.append(
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
    general_news = parsed_articles
    recent_interval_news = [it for it in general_news if it["_published_dt"] and it["_published_dt"] >= recent_threshold]

    keywords = [target_raw]
    if "." in target_raw:
        keywords.append(target_raw.split(".")[0])

    novelty_scores = compute_novelty_scores(general_news)
    market_noise = compute_market_noise(ticker_for_price, lookback_days=lookback_days_for_noise)

    # Compute raw signals
    scored = []
    raw_signals = []
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
        raw_signals.append(signal)
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
                "signal_strength": float(signal),  # raw
                "signal_strength_pct": None,  # fill later
            }
        )

    # Normalize raw_signals -> percentages (0..100)
    signals_arr = np.array(raw_signals, dtype=float)
    pct_list = []
    if signals_arr.size == 0:
        pct_list = []
    else:
        mean = float(np.nanmean(signals_arr))
        std = float(np.nanstd(signals_arr))
        if std > 0:
            # z-score then sigmoid to get stable 0..1 values, then *100
            z = (signals_arr - mean) / std
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            pct_list = (sigmoid * 100.0).tolist()
        else:
            # fallback: absolute logistic mapping (center at 10, k tuned)
            k = 0.15
            center = 10.0
            pct_list = [(1.0 / (1.0 + math.exp(-k * (s - center)))) * 100.0 for s in signals_arr]

    # Attach percentages back
    for i, p in enumerate(pct_list):
        scored[i]["signal_strength_pct"] = round(float(p), 2)

    # recent scored subset
    recent_urls = {it.get("url") for it in recent_interval_news}
    recent_scored = [s for s in scored if (s.get("url") in recent_urls) or (s.get("published_at") in [r.get("published_at") for r in recent_interval_news])]

    # ------------------------------
    # Overall aggregated analysis
    # ------------------------------
    # Compute recency weights (age-based) and credibility weights
    lambda_decay = 0.1  # tune: higher -> faster decay by age
    now_dt = datetime.now(timezone.utc)

    def age_hours_from_published(published_iso):
        if not published_iso:
            return None
        try:
            dt = pd.to_datetime(published_iso, utc=True)
            if pd.isna(dt):
                return None
            delta = now_dt - dt.to_pydatetime()
            return max(0.0, delta.total_seconds() / 3600.0)
        except Exception:
            return None

    # Build arrays for aggregation
    agg_items = []
    recent_items = []
    for item in scored:
        pub = item.get("published_at")
        age_h = age_hours_from_published(pub)
        recency_w = math.exp(-lambda_decay * age_h) if age_h is not None else 1.0
        cred_w = float(item.get("source_credibility") or 0.5)
        weight = cred_w * recency_w
        # Collect numeric fields (fall back to defaults when missing)
        rel = float(item.get("relevance") or 0.0)
        sent = float(item.get("sentiment") or 0.5)
        nov = float(item.get("novelty") or 0.5)
        cred = float(item.get("source_credibility") or 0.5)
        raw_sig = float(item.get("signal_strength") or 0.0)
        pct_sig = float(item.get("signal_strength_pct") or 0.0)

        agg_items.append(
            {
                "weight": weight,
                "relevance": rel,
                "sentiment": sent,
                "novelty": nov,
                "credibility": cred,
                "raw_signal": raw_sig,
                "pct_signal": pct_sig,
                "published_at": pub,
                "age_hours": age_h,
                "title": item.get("title"),
                "url": item.get("url"),
            }
        )

    # identify recent items by matching recent_interval_news (by url or published_at)
    recent_set_urls = {it.get("url") for it in recent_interval_news}
    recent_set_published = {it.get("published_at") for it in recent_interval_news}
    for it in agg_items:
        if (it["url"] in recent_set_urls) or (it["published_at"] in recent_set_published):
            recent_items.append(it)

    def weighted_mean(items, field, weight_field="weight", default=0.0):
        vals = [float(x.get(field) or 0.0) for x in items]
        ws = [float(x.get(weight_field) or 0.0) for x in items]
        if not vals or sum(ws) == 0:
            return float(default)
        return float(np.dot(vals, ws) / float(sum(ws)))

    # Overall aggregated metrics (weighted)
    overall = {
        "coverage": len(agg_items),
        "mean_relevance": round(weighted_mean(agg_items, "relevance"), 4),
        "mean_sentiment": round(weighted_mean(agg_items, "sentiment"), 4),
        "mean_novelty": round(weighted_mean(agg_items, "novelty"), 4),
        "mean_source_credibility": round(weighted_mean(agg_items, "credibility"), 4),
        "overall_signal_raw": round(weighted_mean(agg_items, "raw_signal"), 6),
        "overall_signal_pct": round(weighted_mean(agg_items, "pct_signal"), 2),
        "num_recent": len(recent_items),
    }

    # Recent-interval aggregated metrics (weighted)
    recent_overall = {
        "coverage": len(recent_items),
        "mean_relevance": round(weighted_mean(recent_items, "relevance"), 4),
        "mean_sentiment": round(weighted_mean(recent_items, "sentiment"), 4),
        "mean_novelty": round(weighted_mean(recent_items, "novelty"), 4),
        "mean_source_credibility": round(weighted_mean(recent_items, "credibility"), 4),
        "overall_signal_raw": round(weighted_mean(recent_items, "raw_signal"), 6),
        "overall_signal_pct": round(weighted_mean(recent_items, "pct_signal"), 2),
    }

    # Top N articles that contributed most (by pct)
    top_n = 3
    top_articles = sorted(scored, key=lambda x: (x.get("signal_strength_pct") or 0.0), reverse=True)[:top_n]
    top_articles_simple = [
        {
            "title": t.get("title"),
            "url": t.get("url"),
            "signal_pct": t.get("signal_strength_pct"),
            "relevance": t.get("relevance"),
            "sentiment": t.get("sentiment"),
            "novelty": t.get("novelty"),
            "source_credibility": t.get("source_credibility"),
            "published_at": t.get("published_at"),
        }
        for t in top_articles
    ]

    overall_analysis = {
        "overall": overall,
        "recent_interval": recent_overall,
        "top_articles": top_articles_simple,
        "weighting": {
            "credibility_weight": "source_credibility * recency_weight",
            "recency_lambda_per_hour": lambda_decay,
            "recency_weight_formula": "exp(-lambda * age_hours)",
        },
    }

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
        "overall_analysis": overall_analysis,
        "meta": {
            "serpapi_used": True,
            "nlp_engine": "nltk_vader" if NLTK_VADER_AVAILABLE else "heuristic",
            "novelty_engine": "tfidf_if_available",
            "num_results_requested": num_results,
            "uk_news_focus": True,
        },
    }
    return json.dumps(response)


@news_server.tool(
    name="get_overview_table",
    description="""
Produce a tabular overview for multiple tickers/keywords.
Params:
    tickers: str (comma or space separated, e.g. "VOD,HSBA" or "VOD HSBA")
    serp_api_key: str (optional; falls back to env SERPAPI_KEY)
    num_results: int (per ticker; default 15)
    interval_hours: int (for 'recent' window; default 2)
Returns JSON with keys: rows (list of dicts), table_markdown (string), csv (string)
""",
)
async def get_overview_table(
    tickers: str,
    serp_api_key: Optional[str] = None,
    num_results: int = 15,
    interval_hours: int = 2,
    lookback_days_for_noise: int = 7,
) -> str:
    """
    Build an overview table for a list of tickers/keywords by calling get_serp_news_and_signals
    for each and extracting overall metrics and counts.
    """
    # parse tickers input
    raw = []
    for p in tickers.replace(",", " ").split():
        if p:
            raw.append(p.strip())
    if not raw:
        return json.dumps({"error": "No tickers provided"})

    serp_api_key = serp_api_key or os.getenv("SERPAPI_KEY") or ""
    if not serp_api_key:
        return json.dumps({"error": "SerpAPI key required (env SERPAPI_KEY or serp_api_key argument)."})

    rows = []
    # Sentiment thresholds
    POS_THRESHOLD = 0.55
    NEG_THRESHOLD = 0.45

    # For each ticker, call the existing tool and extract summary
    idx = 1
    for tk in raw:
        try:
            # Reuse existing tool - it returns JSON string
            res_json = await get_serp_news_and_signals(
                target=tk,
                serp_api_key=serp_api_key,
                num_results=num_results,
                interval_hours=interval_hours,
                lookback_days_for_noise=lookback_days_for_noise,
            )
            # res_json may be a json string (from that tool). Ensure dict:
            if isinstance(res_json, str):
                try:
                    res = json.loads(res_json)
                except Exception:
                    # if not decodeable, create error row
                    res = {"error": "invalid response from scoring tool"}
            else:
                res = res_json
        except Exception as e:
            res = {"error": f"tool_call_failed: {e}"}

        if res.get("error"):
            rows.append({
                "S.No.": idx,
                "Stock": tk,
                "Name": tk,
                "Signal Strength (%)": None,
                "Positive Sentiments": 0,
                "Negative Sentiments": 0,
                "Overall Sentiments": None,
                "Context/Timing": None,
                "Content & Source": None,
                "Market Impact": "ERROR: " + str(res.get("error")),
            })
            idx += 1
            continue

        # Extract overall percentage if available
        overall_pct = None
        if res.get("overall_analysis") and res["overall_analysis"].get("overall"):
            overall_pct = res["overall_analysis"]["overall"].get("overall_signal_pct")

        # If not available, fallback to aggregate_signal mean -> map via logistic fallback
        if overall_pct is None:
            agg = res.get("aggregate_signal", {})
            mean_raw = agg.get("mean", None)
            if mean_raw is not None:
                # map mean_raw to percentage via logistic center=10,k=0.15 fallback
                k = 0.15
                center = 10.0
                try:
                    overall_pct = (1.0 / (1.0 + math.exp(-k * (float(mean_raw) - center)))) * 100.0
                    overall_pct = round(overall_pct, 2)
                except Exception:
                    overall_pct = None

        # Count positive / negative sentiments among returned articles
        general_news = res.get("general_news") or []
        pos_count = 0
        neg_count = 0
        sentiments = []
        most_recent_iso = None
        top_contents = []
        for art in general_news:
            sent = art.get("sentiment")
            if sent is None:
                # try compute from title snippet if missing
                try:
                    combined = " ".join(filter(None, [art.get("title",""), art.get("snippet","")]))
                    sent = compute_sentiment_score(combined)
                except Exception:
                    sent = 0.5
            sentiments.append(float(sent))
            if float(sent) >= POS_THRESHOLD:
                pos_count += 1
            if float(sent) <= NEG_THRESHOLD:
                neg_count += 1
            # track most recent
            pa = art.get("published_at")
            if pa:
                try:
                    dt = pd.to_datetime(pa, utc=True, errors="coerce")
                    if not pd.isna(dt):
                        iso = dt.isoformat()
                        if (most_recent_iso is None) or (pd.to_datetime(iso) > pd.to_datetime(most_recent_iso)):
                            most_recent_iso = iso
                except Exception:
                    pass
            # record content & source for top few
            top_contents.append({
                "title": art.get("title"),
                "snippet": art.get("snippet"),
                "source": art.get("source"),
                "url": art.get("url"),
                "pct": art.get("signal_strength_pct"),
            })

        # Overall sentiment = mean sentiment scaled to -1..+1 or 0..1? User asked Overall Sentiments -> we'll give 0..1 and interpret
        overall_sentiment = None
        if sentiments:
            overall_sentiment = round(float(np.mean(sentiments)), 4)

        # Context/Timing: show most_recent_iso and interval
        context_timing = f"Recent window: last {interval_hours} hours; Most recent article: {most_recent_iso or 'N/A'}"

        # Content & Source: pick up to 2 top contributing items (by pct)
        # safe: handle None titles and sources, avoid stray backslashes
        top_contrib = sorted(top_contents, key=lambda x: (x.get("pct") or 0.0), reverse=True)[:2]
        content_source_summary = "; ".join(
            [
                f"{(t.get('title') or '')[:80]}... ({(t.get('source') or '')})"
                for t in top_contrib
            ]
        ) if top_contrib else "N/A"

        # Market Impact (qualitative): simple thresholds on overall_pct
        market_impact = "Neutral"
        try:
            if overall_pct is None:
                market_impact = "Unknown"
            else:
                p = float(overall_pct)
                if p >= 80:
                    market_impact = "High (likely bullish)"
                elif p >= 65:
                    market_impact = "Moderate (bullish)"
                elif p >= 50:
                    market_impact = "Neutral"
                elif p >= 35:
                    market_impact = "Moderate (bearish)"
                else:
                    market_impact = "High (likely bearish)"
        except Exception:
            market_impact = "Unknown"

        rows.append({
            "S.No.": idx,
            "Stock": tk,
            "Name": tk,
            "Signal Strength (%)": overall_pct,
            "Positive Sentiments": pos_count,
            "Negative Sentiments": neg_count,
            "Overall Sentiments": overall_sentiment,
            "Context/Timing": context_timing,
            "Content & Source": content_source_summary,
            "Market Impact": market_impact,
        })
        idx += 1

    # Build Markdown table
    headers = ["S.No.", "Stock", "Name", "Signal Strength (%)", "Positive Sentiments", "Negative Sentiments", "Overall Sentiments", "Context/Timing", "Content & Source", "Market Impact"]
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md_vals = [str(r.get(h, "")) if r.get(h, "") is not None else "" for h in headers]
        # sanitize pipes in text
        md_vals = [v.replace("|", "\\|") for v in md_vals]
        md_lines.append("| " + " | ".join(md_vals) + " |")
    table_markdown = "\n".join(md_lines)

    # Build CSV
    import io, csv
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
    })



if __name__ == "__main__":
    print("Starting SerpAPI News MCP server (UK-focused, LSE enforcement, overall analysis)...")
    news_server.run(transport="stdio")
