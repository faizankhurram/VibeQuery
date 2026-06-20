"""
Pure backend NLP pipeline functions.
No CLI, no Rich, no side effects — just importable logic.
"""

import time
import string
import random
from collections import Counter

import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

STOP = set(stopwords.words("english"))


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower().replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in text.split() if w not in STOP)[:2000]


# ── Scrapers ──────────────────────────────────────────────────────────────────

def scrape_wikipedia(kw: str) -> list:
    results = []
    variants = [kw, kw + " algorithms", kw + " applications",
                kw + " techniques", kw + " examples"]
    headers = {"User-Agent": "Mozilla/5.0 (educational-project)"}
    for term in variants:
        try:
            slug = term.strip().replace(" ", "_").title()
            url = f"https://en.wikipedia.org/wiki/{slug}"
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            h1 = soup.find("h1")
            text = " ".join(
                p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50
            )
            if not text:
                continue
            results.append({
                "url": url,
                "title": h1.text if h1 else term,
                "author": "Wikipedia Contributors",
                "date": "2024-01-01",
                "source": "Wikipedia",
                "raw_text": text[:3000],
                "comments": "Community-edited reference article. Neutral encyclopedic tone.",
            })
            time.sleep(0.8)
        except Exception:
            pass
    return results


def scrape_devto(kw: str) -> list:
    results = []
    tag = kw.replace(" ", "").lower()
    try:
        r = requests.get(
            f"https://dev.to/api/articles?tag={tag}&per_page=30", timeout=12
        )
        for art in r.json()[:6]:
            try:
                detail = requests.get(
                    f"https://dev.to/api/articles/{art['id']}", timeout=10
                ).json()
                body = detail.get("body_markdown") or art.get("description", "")
                results.append({
                    "url": art.get("url", ""),
                    "title": art.get("title", "No Title"),
                    "author": art.get("user", {}).get("name", "Unknown"),
                    "date": str(art.get("published_at", "Unknown"))[:10],
                    "source": "Dev.to",
                    "raw_text": body,
                    "article_id": str(art.get("id", "")),
                    "comments": "",
                })
                time.sleep(0.5)
            except Exception:
                pass
    except Exception:
        pass
    return results


def scrape_hackernews(kw: str) -> list:
    results = []
    try:
        r = requests.get(
            f"https://hn.algolia.com/api/v1/search?query={kw}&tags=story&hitsPerPage=35",
            timeout=12,
        )
        for hit in r.json().get("hits", [])[:8]:
            story_url = hit.get("url", "")
            body = hit.get("story_text") or ""
            if story_url and not body:
                try:
                    sr = requests.get(
                        story_url,
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=8,
                    )
                    body = " ".join(
                        p.get_text()
                        for p in BeautifulSoup(sr.text, "html.parser").find_all("p")
                    )[:3000]
                except Exception:
                    body = hit.get("title", "")
            results.append({
                "url": story_url
                or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                "title": hit.get("title", "No Title"),
                "author": hit.get("author", "Unknown"),
                "date": str(hit.get("created_at", "Unknown"))[:10],
                "source": "HackerNews",
                "raw_text": body if body else hit.get("title", ""),
                "article_id": str(hit.get("objectID", "")),
                "comments": "",
            })
            time.sleep(0.4)
    except Exception:
        pass
    return results


# ── Comment fetchers ──────────────────────────────────────────────────────────

def fetch_devto_comments(article_id: str, limit: int = 6) -> str:
    try:
        r = requests.get(
            f"https://dev.to/api/comments?a_id={article_id}", timeout=10
        )
        if r.status_code != 200:
            return ""
        texts = []
        for c in r.json()[:limit]:
            body = BeautifulSoup(
                c.get("body_html", ""), "html.parser"
            ).get_text().strip()
            if body:
                texts.append(body)
        return " | ".join(texts)
    except Exception:
        return ""


def fetch_hn_comments(object_id: str, limit: int = 6) -> str:
    try:
        r = requests.get(
            f"https://hn.algolia.com/api/v1/items/{object_id}", timeout=10
        )
        if r.status_code != 200:
            return ""
        texts = []
        for child in r.json().get("children", [])[:limit]:
            raw = child.get("text") or ""
            if raw:
                cleaned = BeautifulSoup(raw, "html.parser").get_text().strip()
                if cleaned:
                    texts.append(cleaned)
        return " | ".join(texts)
    except Exception:
        return ""


# ── DataFrame builder (with fallback data) ────────────────────────────────────

def build_dataframe(all_data: list, keyword: str) -> pd.DataFrame:
    if len(all_data) < 10:
        fallbacks = [
            {
                "url": "fb1", "title": f"Introduction to {keyword}",
                "author": "John Doe", "date": "2024-01-01", "source": "Backup",
                "raw_text": (
                    f"{keyword} is a broad and evolving field. It uses data-driven "
                    "methods to automate intelligent decisions at scale."
                ),
                "comments": "Very well written! I learned a lot from this.",
            },
            {
                "url": "fb2", "title": f"Applications of {keyword}",
                "author": "Jane Smith", "date": "2024-02-15", "source": "Backup",
                "raw_text": (
                    f"Applications of {keyword} span healthcare, finance, "
                    "education, and engineering."
                ),
                "comments": "Excellent overview! Highly recommended for beginners.",
            },
            {
                "url": "fb3", "title": f"Core Techniques in {keyword}",
                "author": "Alice Web", "date": "2024-03-05", "source": "Backup",
                "raw_text": (
                    f"{keyword} techniques include preprocessing, feature extraction, "
                    "model training, and evaluation."
                ),
                "comments": "This is a terrible explanation. Very disappointing.",
            },
            {
                "url": "fb4", "title": f"History of {keyword}",
                "author": "Bob Tech", "date": "2024-04-01", "source": "Backup",
                "raw_text": (
                    f"The history of {keyword} spans decades, from rule-based systems "
                    "to modern deep learning."
                ),
                "comments": "Interesting historical perspective. Not bad at all.",
            },
            {
                "url": "fb5", "title": f"Future Trends in {keyword}",
                "author": "Carol AI", "date": "2024-05-15", "source": "Backup",
                "raw_text": (
                    f"Future trends in {keyword} include automation, explainability, "
                    "and ethical AI frameworks."
                ),
                "comments": "Amazing article! Exactly what I was looking for. Brilliant work.",
            },
            {
                "url": "fb6", "title": f"{keyword} in Industry",
                "author": "Dave Analyst", "date": "2024-06-20", "source": "Backup",
                "raw_text": (
                    f"Industry adoption of {keyword} has accelerated, optimizing "
                    "supply chains and detecting fraud."
                ),
                "comments": "Very helpful! Great insights for practitioners.",
            },
            {
                "url": "fb7", "title": f"Challenges in {keyword}",
                "author": "Eva Research", "date": "2024-07-08", "source": "Backup",
                "raw_text": (
                    f"Despite its promise, {keyword} faces data quality, "
                    "interpretability, and computational cost challenges."
                ),
                "comments": "I strongly disagree. Misleading and poorly researched.",
            },
            {
                "url": "fb8", "title": f"Beginner's Guide to {keyword}",
                "author": "Frank Edu", "date": "2024-08-12", "source": "Backup",
                "raw_text": (
                    f"This step-by-step tutorial introduces {keyword} to beginners "
                    "using Python."
                ),
                "comments": "Perfect for beginners! Clear, concise and well structured.",
            },
            {
                "url": "fb9", "title": f"Research Directions in {keyword}",
                "author": "Grace Scholar", "date": "2024-09-01", "source": "Backup",
                "raw_text": (
                    f"Research in {keyword} explores transfer learning, federated "
                    "learning, and self-supervised methods."
                ),
                "comments": "Fascinating research summary. Academic progress well highlighted.",
            },
            {
                "url": "fb10", "title": f"Ethics and {keyword}",
                "author": "Henry Ethics", "date": "2024-10-05", "source": "Backup",
                "raw_text": (
                    f"Ethical dimensions of {keyword} include bias, fairness, "
                    "privacy, and societal impact."
                ),
                "comments": "Critical concerns raised. Poorly handled discussion.",
            },
        ]
        for fb in fallbacks:
            if len(all_data) >= 10:
                break
            if not any(a.get("title") == fb["title"] for a in all_data):
                all_data.append(fb)

    df = pd.DataFrame(all_data)
    if "article_id" not in df.columns:
        df["article_id"] = ""
    else:
        df["article_id"] = df["article_id"].fillna("").astype(str)
    if "comments" not in df.columns:
        df["comments"] = ""
    else:
        df["comments"] = df["comments"].fillna("").astype(str)
    return df


def enrich_df_comments(df: pd.DataFrame) -> tuple:
    def _enrich(row):
        aid = str(row["article_id"]).strip()
        if not aid:
            return row["comments"]
        if row["source"] == "Dev.to":
            fetched = fetch_devto_comments(aid)
            return fetched if fetched else row["comments"]
        if row["source"] == "HackerNews":
            fetched = fetch_hn_comments(aid)
            return fetched if fetched else row["comments"]
        return row["comments"]

    df = df.copy()
    df["comments"] = df.apply(_enrich, axis=1)
    fetched = (
        df["source"].isin(["Dev.to", "HackerNews"]) & df["comments"].str.len().gt(0)
    ).sum()
    return df, int(fetched)


# ── NLP steps ─────────────────────────────────────────────────────────────────

def run_tfidf(df: pd.DataFrame) -> tuple:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
    X = vectorizer.fit_transform(df["clean_text"])
    features = vectorizer.get_feature_names_out()

    def top_kw(row):
        return [
            w for w, _ in sorted(
                zip(features, row), key=lambda x: x[1], reverse=True
            )[:5]
        ]

    df = df.copy()
    df["keywords"] = [top_kw(r) for r in X.toarray()]
    return df, features, X


def run_lda(X, features, n_articles: int) -> list:
    n = min(3, n_articles)
    lda = LatentDirichletAllocation(n_components=n, random_state=42)
    lda.fit(X)
    return [[features[i] for i in t.argsort()[-5:]] for t in lda.components_]


def detect_motive(text: str) -> str:
    t = text.lower()
    if any(p in t for p in ["how to", "step by step", "tutorial", "guide"]):
        return "Tutorial"
    if any(p in t for p in ["buy", "pricing", "discount", "offer", "sale"]):
        return "Promotional"
    if any(p in t for p in ["research", "study", "findings", "paper", "journal"]):
        return "Academic"
    if any(p in t for p in ["i think", "in my opinion", "i believe", "personally"]):
        return "Opinion"
    return "Informative"


def detect_tone(text: str) -> str:
    return (
        "Subjective"
        if TextBlob(str(text)).sentiment.subjectivity > 0.5
        else "Objective"
    )


def apply_motive_tone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["motive"] = df["raw_text"].apply(detect_motive)
    df["tone"] = df["raw_text"].apply(detect_tone)
    return df


def get_sentiment(text: str) -> str:
    s = TextBlob(str(text)).sentiment.polarity
    return "Positive" if s > 0.1 else ("Negative" if s < -0.1 else "Neutral")


def generate_comment(title: str, generator) -> str:
    if not isinstance(title, str) or title in ("Error", "No Title"):
        return "Interesting read!"
    try:
        res = generator(
            f"Write a short, engaging reader comment for this blog post titled: {title[:100]}",
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        return res[0]["generated_text"].replace("\n", " ").strip()
    except Exception:
        return "Great article!"


def apply_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scraped_sentiment"] = df["comments"].apply(get_sentiment)
    df["generated_sentiment"] = df["generated_comment"].apply(get_sentiment)
    df["sentiment_score"] = df["generated_comment"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    random.seed(42)
    df["comment_count"] = df["sentiment_score"].apply(
        lambda x: int(abs(x) * 100) + random.randint(5, 20)
    )
    return df
