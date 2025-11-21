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
    pos_words = ["gain", "up", "strong", "beat", "positive", "upgrade", "outperform", "record", "optimis", "raise", "upgrade"]
    neg_words = ["loss", "drop", "down", "weak", "miss", "negative", "downgrade", "lawsuit", "cut", "warn", "fall"]
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
# SerpAPI (HTTP) helper - UK focused by default (gl=uk, hl=en)
# -------------------------
def serpapi_search_news(
    query: str,
    serp_api_key: str,
    num_results: int = 20,
    language: str = "en",
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


# -------------------------
# Overview table tool — merged stock/name and textual sentiment summaries
# -------------------------
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

    # heuristics/thresholds
    POS_THRESHOLD = 0.60
    NEG_THRESHOLD = 0.40

    # context keywords (same as before)
    CONTEXT_KEYWORDS = {
        "Earnings Release": ["quarter", "quarterly", "q1", "q2", "q3", "q4", "results", "earnings", "revenue", "profit", "loss"],
        "CEO Statement": ["ceo", "chief executive", "said", "statement", "commented", "spokesman", "spokesperson"],
        "M&A": ["acquir", "acquisit", "merger", "buyout", "takeover", "bid for"],
        "Regulatory / Lawsuit": ["investig", "regulator", "fine", "lawsuit", "probe", "court", "regulatory"],
        "Guidance": ["guidance", "outlook", "forecast", "raise", "lowered", "cut guidance"],
        "Dividend": ["dividend", "payout", "special dividend", "dividends"],
        "Restructuring / Layoff": ["layoff", "redund", "restructur", "job cut", "reorg"],
        "Product / Contract": ["contract", "deal", "agreement", "order", "supply", "launch", "product"],
    }

    def detect_context(text: str) -> Optional[str]:
        if not text:
            return None
        txt = text.lower()
        for ctx, kws in CONTEXT_KEYWORDS.items():
            for kw in kws:
                if kw in txt:
                    return ctx
        return None

    # helper to safely parse published ISO to pandas.Timestamp (utc)
    def parse_iso(iso):
        try:
            dt = pd.to_datetime(iso, utc=True, errors="coerce")
            if pd.isna(dt):
                return None
            return dt
        except Exception:
            return None

    idx = 1
    for tk in raw:
        try:
            res_json = await get_serp_news_and_signals(
                target=tk,
                serp_api_key=serp_api_key,
                num_results=num_results,
                interval_hours=interval_hours,
                lookback_days_for_noise=lookback_days_for_noise,
            )
            if isinstance(res_json, str):
                try:
                    res = json.loads(res_json)
                except Exception:
                    res = {"error": "invalid response from scoring tool"}
            else:
                res = res_json
        except Exception as e:
            res = {"error": f"tool_call_failed: {e}"}

        if res.get("error"):
            rows.append({
                "S.No.": idx,
                "Stock": tk,
                "Signal Strength (%)": None,
                "Positive Sentiments": "ERROR: " + str(res.get("error")),
                "Negative Sentiments": "ERROR",
                "Overall Sentiments": "Unknown",
                "Context/Timing": "N/A",
                "Content & Source": "N/A",
                "Market Impact": "ERROR: " + str(res.get("error")),
            })
            idx += 1
            continue

        # overall pct
        overall_pct = None
        if res.get("overall_analysis") and res["overall_analysis"].get("overall"):
            overall_pct = res["overall_analysis"]["overall"].get("overall_signal_pct")

        if overall_pct is None:
            agg = res.get("aggregate_signal", {})
            mean_raw = agg.get("mean", None)
            if mean_raw is not None:
                k = 0.15; center = 10.0
                try:
                    overall_pct = (1.0 / (1.0 + math.exp(-k * (float(mean_raw) - center)))) * 100.0
                    overall_pct = round(overall_pct, 2)
                except Exception:
                    overall_pct = None

        general_news = res.get("general_news") or []
        highlights = []
        positives = []
        negatives = []
        neutrals = []
        all_sentiments = []
        most_recent_dt = None

        # collect per-article info with parsed dates
        for art in general_news:
            title = (art.get("title") or "").strip()
            snippet = (art.get("snippet") or "").strip()
            combined = (title + " " + snippet).strip()
            sent = art.get("sentiment")
            if sent is None:
                sent = compute_sentiment_score(combined)
            sent = float(sent)
            all_sentiments.append(sent)

            # published parsed
            pa = art.get("published_at")
            dt = parse_iso(pa)
            if dt is not None:
                if (most_recent_dt is None) or (dt > most_recent_dt):
                    most_recent_dt = dt

            entry = {
                "title": title,
                "snippet": snippet,
                "source": art.get("source") or "",
                "url": art.get("url") or "",
                "pct": art.get("signal_strength_pct") or 0.0,
                "dt": dt,
                "sent": sent,
            }

            highlights.append(entry)
            if sent >= POS_THRESHOLD:
                positives.append(entry)
            elif sent <= NEG_THRESHOLD:
                negatives.append(entry)
            else:
                neutrals.append(entry)

        # Build textual positive/negative summaries (top 2 each by pct then recent)
        def build_text_summary(items, topk=2):
            if not items:
                return "None"
            # sort by (pct desc, dt desc)
            items_sorted = sorted(items, key=lambda x: (float(x.get("pct") or 0.0), x.get("dt") or pd.Timestamp.min), reverse=True)[:topk]
            parts = []
            for it in items_sorted:
                t = (it.get("title") or "")
                s = (it.get("snippet") or "")
                src = it.get("source") or ""
                t_short = (t[:100] + "...") if len(t) > 100 else t
                s_short = (s[:140] + "...") if len(s) > 140 else s
                parts.append(f"{t_short} — {s_short} ({src})")
            return " || ".join(parts)

        positive_summary = build_text_summary(positives, topk=2)
        negative_summary = build_text_summary(negatives, topk=2)

        # Content & Source: top 3 highlights by (pct desc, dt desc)
        top_highlights = sorted(highlights, key=lambda x: (float(x.get("pct") or 0.0), x.get("dt") or pd.Timestamp.min), reverse=True)[:3]
        ch_texts = []
        for h in top_highlights:
            t = h.get("title") or ""
            s = h.get("snippet") or ""
            src = h.get("source") or ""
            t_short = (t[:120] + "...") if len(t) > 120 else t
            s_short = (s[:160] + "...") if len(s) > 160 else s
            ch_texts.append(f"{t_short} — {s_short} ({src})")
        content_source_summary = " || ".join(ch_texts) if ch_texts else "N/A"

        # Context/Timing: pick the most recent article that matches any context keyword
        context_candidate = None
        context_candidate_dt = None
        for h in highlights:
            txt = (h.get("title") or "") + " " + (h.get("snippet") or "")
            ctx = detect_context(txt)
            if ctx:
                dt = h.get("dt")
                # prefer later dt; if dt missing, keep but lower priority
                if dt is not None:
                    if (context_candidate_dt is None) or (dt > context_candidate_dt):
                        context_candidate = ctx
                        context_candidate_dt = dt
                else:
                    # if we don't have any candidate yet, set it
                    if context_candidate is None:
                        context_candidate = ctx

        primary_context = context_candidate or "General News"
        most_recent_iso = most_recent_dt.isoformat() if most_recent_dt is not None else "N/A"

        # Overall sentiment label (bullish/bearish/neutral)
        overall_label = "Neutral"
        try:
            if overall_pct is not None:
                p = float(overall_pct)
                if p >= 62:
                    overall_label = "Bullish"
                elif p <= 38:
                    overall_label = "Bearish"
                else:
                    overall_label = "Neutral"
            else:
                if all_sentiments:
                    m = float(np.mean(all_sentiments))
                    if m >= 0.62:
                        overall_label = "Bullish"
                    elif m <= 0.38:
                        overall_label = "Bearish"
                    else:
                        overall_label = "Neutral"
        except Exception:
            overall_label = "Neutral"

        # Market impact (same thresholds)
        market_impact = "Unknown" if overall_pct is None else (
            "High (likely bullish)" if overall_pct >= 80 else
            "Moderate (bullish)" if overall_pct >= 65 else
            "Neutral" if overall_pct >= 50 else
            "Moderate (bearish)" if overall_pct >= 35 else
            "High (likely bearish)"
        )

        # Display name resolve
        display_name = tk
        try:
            cand = tk.upper()
            tf = cand if cand.endswith(".L") else cand + ".L"
            info = yf.Ticker(tf).info
            longname = info.get("longName") or info.get("shortName") or None
            if longname:
                display_name = f"{tk} — {longname}"
        except Exception:
            pass

        # Add number of articles and avg novelty (if available) into a small note appended to Context/Timing
        num_articles = len(highlights)
        avg_novelty = None
        try:
            # compute avg novelty if general_news items contain novelty; fallback compute using titles presence
            novs = []
            for art in (res.get("general_news") or []):
                nval = art.get("novelty")
                if nval is not None:
                    novs.append(float(nval))
            if novs:
                avg_novelty = round(float(np.mean(novs)), 4)
        except Exception:
            avg_novelty = None

        context_timing_field = f"{primary_context}; Most recent: {most_recent_iso}; Articles: {num_articles}"
        if avg_novelty is not None:
            context_timing_field += f"; AvgNovelty: {avg_novelty}"

        rows.append({
            "S.No.": idx,
            "Stock": display_name,
            "Signal Strength (%)": overall_pct,
            "Positive Sentiments": positive_summary,
            "Negative Sentiments": negative_summary,
            "Overall Sentiments": overall_label,
            "Context/Timing": context_timing_field,
            "Content & Source": content_source_summary,
            "Market Impact": market_impact,
        })
        idx += 1

    # Build Markdown and CSV (same truncation as before)
    headers = ["S.No.", "Stock", "Signal Strength (%)", "Positive Sentiments", "Negative Sentiments", "Overall Sentiments", "Context/Timing", "Content & Source", "Market Impact"]
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        def trunc(s, n=120):
            if s is None:
                return ""
            s = str(s)
            return (s[:n] + "...") if len(s) > n else s
        md_vals = [trunc(r.get(h, "")) for h in headers]
        md_vals = [v.replace("|", "\\|") for v in md_vals]
        md_lines.append("| " + " | ".join(md_vals) + " |")
    table_markdown = "\n".join(md_lines)

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
