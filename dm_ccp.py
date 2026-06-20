import subprocess, sys

def _pip(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

_pip(
    "requests", "beautifulsoup4", "nltk", "textblob",
    "wordcloud", "scikit-learn", "transformers", "matplotlib", "pandas", "rich"
)

import requests
import time
import string
import random
import os
from collections import Counter
from bs4 import BeautifulSoup

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
from nltk.corpus import stopwords

console = Console()
STOP    = set(stopwords.words("english"))

console.rule("[bold cyan]Webscraping & NLP Pipeline[/bold cyan]")

keyword = console.input("\n[bold]Keyword:[/bold] ").strip()
console.print(f"[dim]Using keyword:[/dim] [bold white]{keyword}[/bold white]\n")


def clean_text(text):
    text = str(text).lower().replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in text.split() if w not in STOP)[:2000]


def scrape_wikipedia(kw):
    results = []
    variants = [kw, kw + " algorithms", kw + " applications", kw + " techniques", kw + " examples"]
    headers  = {"User-Agent": "Mozilla/5.0 (educational-project)"}
    for term in variants:
        try:
            slug = term.strip().replace(" ", "_").title()
            url  = f"https://en.wikipedia.org/wiki/{slug}"
            r    = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            h1   = soup.find("h1")
            text = " ".join(p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50)
            if not text:
                continue
            results.append({
                "url": url, "title": h1.text if h1 else term,
                "author": "Wikipedia Contributors", "date": "2024-01-01",
                "source": "Wikipedia", "raw_text": text[:3000],
                "comments": "Community-edited reference article. Neutral encyclopedic tone."
            })
            time.sleep(0.8)
        except Exception:
            pass
    return results


def scrape_devto(kw):
    results = []
    tag = kw.replace(" ", "").lower()
    try:
        r = requests.get(f"https://dev.to/api/articles?tag={tag}&per_page=30", timeout=12)
        for art in r.json()[:6]:
            try:
                detail = requests.get(f"https://dev.to/api/articles/{art['id']}", timeout=10).json()
                body   = detail.get("body_markdown") or art.get("description", "")
                results.append({
                    "url":        art.get("url", ""),
                    "title":      art.get("title", "No Title"),
                    "author":     art.get("user", {}).get("name", "Unknown"),
                    "date":       str(art.get("published_at", "Unknown"))[:10],
                    "source":     "Dev.to",
                    "raw_text":   body,
                    "article_id": str(art.get("id", "")),
                    "comments":   ""
                })
                time.sleep(0.5)
            except Exception:
                pass
    except Exception:
        pass
    return results


def scrape_hackernews(kw):
    results = []
    try:
        r = requests.get(
            f"https://hn.algolia.com/api/v1/search?query={kw}&tags=story&hitsPerPage=35",
            timeout=12
        )
        for hit in r.json().get("hits", [])[:8]:
            story_url = hit.get("url", "")
            body      = hit.get("story_text") or ""
            if story_url and not body:
                try:
                    sr   = requests.get(story_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                    body = " ".join(p.get_text() for p in BeautifulSoup(sr.text, "html.parser").find_all("p"))[:3000]
                except Exception:
                    body = hit.get("title", "")
            results.append({
                "url":        story_url or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                "title":      hit.get("title", "No Title"),
                "author":     hit.get("author", "Unknown"),
                "date":       str(hit.get("created_at", "Unknown"))[:10],
                "source":     "HackerNews",
                "raw_text":   body if body else hit.get("title", ""),
                "article_id": str(hit.get("objectID", "")),
                "comments":   ""
            })
            time.sleep(0.4)
    except Exception:
        pass
    return results


def fetch_devto_comments(article_id: str, limit: int = 6) -> str:
    try:
        r = requests.get(f"https://dev.to/api/comments?a_id={article_id}", timeout=10)
        if r.status_code != 200:
            return ""
        texts = []
        for c in r.json()[:limit]:
            body = BeautifulSoup(c.get("body_html", ""), "html.parser").get_text().strip()
            if body:
                texts.append(body)
        return " | ".join(texts)
    except Exception:
        return ""


def fetch_hn_comments(object_id: str, limit: int = 6) -> str:
    try:
        r = requests.get(f"https://hn.algolia.com/api/v1/items/{object_id}", timeout=10)
        if r.status_code != 200:
            return ""
        texts = []
        for child in r.json().get("children", [])[:limit]:
            raw = child.get("text") or ""
            if raw:
                clean = BeautifulSoup(raw, "html.parser").get_text().strip()
                if clean:
                    texts.append(clean)
        return " | ".join(texts)
    except Exception:
        return ""


console.rule("[bold]Step 1 — Data Collection[/bold]")

sources = [
    ("Wikipedia",   scrape_wikipedia),
    ("Dev.to",      scrape_devto),
    ("HackerNews",  scrape_hackernews),
]

all_data = []
src_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
src_table.add_column("Source",   style="white")
src_table.add_column("Articles", justify="right", style="green")

for name, fn in sources:
    result = fn(keyword)
    all_data += result
    src_table.add_row(name, str(len(result)))

console.print(src_table)

if len(all_data) < 10:
    fallbacks = [
        {"url": "fb1",  "title": f"Introduction to {keyword}",        "author": "John Doe",     "date": "2024-01-01",
         "source": "Backup", "raw_text": f"{keyword} is a broad and evolving field. It uses data-driven methods to automate intelligent decisions at scale.",
         "comments": "Very well written! I learned a lot from this."},
        {"url": "fb2",  "title": f"Applications of {keyword}",        "author": "Jane Smith",   "date": "2024-02-15",
         "source": "Backup", "raw_text": f"Applications of {keyword} span healthcare, finance, education, and engineering.",
         "comments": "Excellent overview! Highly recommended for beginners."},
        {"url": "fb3",  "title": f"Core Techniques in {keyword}",     "author": "Alice Web",    "date": "2024-03-05",
         "source": "Backup", "raw_text": f"{keyword} techniques include preprocessing, feature extraction, model training, and evaluation.",
         "comments": "This is a terrible explanation. Very disappointing and incomplete."},
        {"url": "fb4",  "title": f"History of {keyword}",             "author": "Bob Tech",     "date": "2024-04-01",
         "source": "Backup", "raw_text": f"The history of {keyword} spans decades, from rule-based systems to modern deep learning.",
         "comments": "Interesting historical perspective. Not bad at all."},
        {"url": "fb5",  "title": f"Future Trends in {keyword}",       "author": "Carol AI",     "date": "2024-05-15",
         "source": "Backup", "raw_text": f"Future trends in {keyword} include automation, explainability, and ethical AI frameworks.",
         "comments": "Amazing article! Exactly what I was looking for. Brilliant work."},
        {"url": "fb6",  "title": f"{keyword} in Industry",            "author": "Dave Analyst", "date": "2024-06-20",
         "source": "Backup", "raw_text": f"Industry adoption of {keyword} has accelerated, optimizing supply chains and detecting fraud.",
         "comments": "Very helpful! Great insights for practitioners."},
        {"url": "fb7",  "title": f"Challenges in {keyword}",          "author": "Eva Research", "date": "2024-07-08",
         "source": "Backup", "raw_text": f"Despite its promise, {keyword} faces data quality, interpretability, and computational cost challenges.",
         "comments": "I strongly disagree with several points. Misleading and poorly researched."},
        {"url": "fb8",  "title": f"Beginner's Guide to {keyword}",    "author": "Frank Edu",    "date": "2024-08-12",
         "source": "Backup", "raw_text": f"This step-by-step tutorial introduces {keyword} to beginners using Python.",
         "comments": "Perfect for beginners! Clear, concise and well structured."},
        {"url": "fb9",  "title": f"Research Directions in {keyword}", "author": "Grace Scholar","date": "2024-09-01",
         "source": "Backup", "raw_text": f"Research in {keyword} explores transfer learning, federated learning, and self-supervised methods.",
         "comments": "Fascinating research summary. Academic progress well highlighted."},
        {"url": "fb10", "title": f"Ethics and {keyword}",             "author": "Henry Ethics", "date": "2024-10-05",
         "source": "Backup", "raw_text": f"Ethical dimensions of {keyword} include bias, fairness, privacy, and societal impact.",
         "comments": "Critical concerns raised. Poorly handled ethical discussion unfortunately."},
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

console.print("[dim]Fetching real user comments from Dev.to and HackerNews...[/dim]")

def enrich_comments(row):
    aid = row["article_id"].strip()
    if not aid:
        return row["comments"]
    if row["source"] == "Dev.to":
        fetched = fetch_devto_comments(aid)
        return fetched if fetched else row["comments"]
    if row["source"] == "HackerNews":
        fetched = fetch_hn_comments(aid)
        return fetched if fetched else row["comments"]
    return row["comments"]

df["comments"] = df.apply(enrich_comments, axis=1)

fetched_count = (df["source"].isin(["Dev.to", "HackerNews"]) & df["comments"].str.len().gt(0)).sum()
console.print(f"[green]Real comments fetched for {fetched_count} article(s)[/green]")

df["clean_text"] = df["raw_text"].apply(clean_text)

art_table = Table(title=f"Collected Articles  [{len(df)} total]", box=box.ROUNDED,
                  show_header=True, header_style="bold cyan")
art_table.add_column("#",      width=3,  justify="right", style="dim")
art_table.add_column("Title",  width=42, no_wrap=True)
art_table.add_column("Author", width=18, no_wrap=True, style="yellow")
art_table.add_column("Date",   width=12)
art_table.add_column("Source", width=12, style="cyan")
for i, row in df.iterrows():
    art_table.add_row(
        str(i + 1),
        row["title"][:42],
        row["author"][:18],
        row["date"],
        row["source"]
    )
console.print(art_table)


console.rule("[bold]Step 2 — NLP Analysis[/bold]")

vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
X          = vectorizer.fit_transform(df["clean_text"])
features   = vectorizer.get_feature_names_out()

def top_keywords(row):
    return [w for w, _ in sorted(zip(features, row), key=lambda x: x[1], reverse=True)[:5]]

df["keywords"] = [top_keywords(row) for row in X.toarray()]

n_topics = min(3, len(df))
lda      = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)
topics = [[features[i] for i in t.argsort()[-5:]] for t in lda.components_]

topic_table = Table(title="LDA Topics", box=box.SIMPLE_HEAD, header_style="bold cyan")
topic_table.add_column("Topic", style="bold white", width=8)
topic_table.add_column("Top Keywords")
for i, t in enumerate(topics):
    topic_table.add_row(f"Topic {i + 1}", ", ".join(t))
console.print(topic_table)

def detect_motive(text):
    t = text.lower()
    if any(p in t for p in ["how to", "step by step", "tutorial", "guide"]): return "Tutorial"
    if any(p in t for p in ["buy", "pricing", "discount", "offer", "sale"]):  return "Promotional"
    if any(p in t for p in ["research", "study", "findings", "paper", "journal"]): return "Academic"
    if any(p in t for p in ["i think", "in my opinion", "i believe", "personally"]): return "Opinion"
    return "Informative"

def detect_tone(text):
    return "Subjective" if TextBlob(str(text)).sentiment.subjectivity > 0.5 else "Objective"

df["motive"] = df["raw_text"].apply(detect_motive)
df["tone"]   = df["raw_text"].apply(detect_tone)

nlp_table = Table(title="NLP Per-Article Results", box=box.ROUNDED, header_style="bold cyan")
nlp_table.add_column("#",        width=3,  justify="right", style="dim")
nlp_table.add_column("Title",    width=36, no_wrap=True)
nlp_table.add_column("Source",   width=12, style="cyan")
nlp_table.add_column("Motive",   width=12)
nlp_table.add_column("Tone",     width=12)
nlp_table.add_column("Keywords", width=38)
for i, row in df.iterrows():
    nlp_table.add_row(
        str(i + 1),
        row["title"][:36],
        row["source"],
        row["motive"],
        row["tone"],
        ", ".join(row["keywords"])
    )
console.print(nlp_table)

tone_by_src = df.groupby("source")["tone"].value_counts()
tone_src_table = Table(title="Tone by Source", box=box.SIMPLE_HEAD, header_style="bold cyan")
tone_src_table.add_column("Source", style="cyan")
tone_src_table.add_column("Tone")
tone_src_table.add_column("Count", justify="right", style="green")
for (src, tone), count in tone_by_src.items():
    tone_src_table.add_row(src, tone, str(count))
console.print(tone_src_table)


console.rule("[bold]Step 3 — LLM Comment Generation (Flan-T5)[/bold]")
console.print("[dim]Loading google/flan-t5-base — first run downloads ~250 MB[/dim]")

generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_comment(title):
    if not isinstance(title, str) or title in ("Error", "No Title"):
        return "Interesting read!"
    try:
        res = generator(
            f"Write a short, engaging reader comment for this blog post titled: {title[:100]}",
            max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9
        )
        return res[0]["generated_text"].replace("\n", " ").strip()
    except Exception:
        return "Great article!"

df["generated_comment"] = df["title"].apply(generate_comment)

cmt_table = Table(title="Generated Comments", box=box.ROUNDED, header_style="bold cyan")
cmt_table.add_column("#",       width=3, justify="right", style="dim")
cmt_table.add_column("Title",   width=36, no_wrap=True)
cmt_table.add_column("Comment", width=55)
for i, row in df.iterrows():
    cmt_table.add_row(str(i + 1), row["title"][:36], row["generated_comment"][:55])
console.print(cmt_table)


console.rule("[bold]Step 4 — Sentiment Analysis[/bold]")

def get_sentiment(text):
    s = TextBlob(str(text)).sentiment.polarity
    return "Positive" if s > 0.1 else ("Negative" if s < -0.1 else "Neutral")

df["scraped_sentiment"]   = df["comments"].apply(get_sentiment)
df["generated_sentiment"] = df["generated_comment"].apply(get_sentiment)
df["sentiment_score"]     = df["generated_comment"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

random.seed(42)
df["comment_count"] = df["sentiment_score"].apply(lambda x: int(abs(x) * 100) + random.randint(5, 20))

SENT_STYLE = {"Positive": "green", "Neutral": "yellow", "Negative": "red"}

sent_table = Table(title="Sentiment Results", box=box.ROUNDED, header_style="bold cyan")
sent_table.add_column("#",               width=3,  justify="right", style="dim")
sent_table.add_column("Title",           width=34, no_wrap=True)
sent_table.add_column("Scraped Sent.",   width=14, justify="center")
sent_table.add_column("Generated Sent.", width=16, justify="center")
sent_table.add_column("Score",           width=7,  justify="right")
sent_table.add_column("Popularity",      width=10, justify="right")
for i, row in df.iterrows():
    ss = row["scraped_sentiment"]
    gs = row["generated_sentiment"]
    sent_table.add_row(
        str(i + 1),
        row["title"][:34],
        f"[{SENT_STYLE.get(ss, 'white')}]{ss}[/]",
        f"[{SENT_STYLE.get(gs, 'white')}]{gs}[/]",
        f"{row['sentiment_score']:.2f}",
        str(row["comment_count"])
    )
console.print(sent_table)

dist_table = Table(title="Sentiment Distribution", box=box.SIMPLE_HEAD, header_style="bold cyan")
dist_table.add_column("Type",      style="white")
dist_table.add_column("Positive",  justify="right", style="green")
dist_table.add_column("Neutral",   justify="right", style="yellow")
dist_table.add_column("Negative",  justify="right", style="red")
for col, label in [("scraped_sentiment", "Scraped"), ("generated_sentiment", "Generated")]:
    vc = df[col].value_counts()
    dist_table.add_row(
        label,
        str(vc.get("Positive", 0)),
        str(vc.get("Neutral",  0)),
        str(vc.get("Negative", 0))
    )
console.print(dist_table)


console.rule("[bold]Step 5 — Visualizations[/bold]")
os.makedirs("charts", exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 6))
counts = df["generated_sentiment"].value_counts()
colors = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
       colors=[colors.get(k, "#999") for k in counts.index], startangle=140)
ax.set_title("Sentiment Distribution of Generated Comments", fontsize=13)
plt.tight_layout()
plt.savefig("charts/chart1_sentiment_pie.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
for sent, grp in df.groupby("generated_sentiment"):
    ax.scatter(grp["sentiment_score"], grp["comment_count"],
               label=sent, color={"Positive": "green", "Neutral": "orange", "Negative": "red"}.get(sent, "gray"),
               s=80, alpha=0.8)
ax.set_xlabel("Sentiment Score")
ax.set_ylabel("Comment Count (Popularity Proxy)")
ax.set_title("Sentiment Score vs Popularity")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("charts/chart2_sentiment_vs_popularity.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
df["motive"].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Blog Motive Distribution")
ax.set_xlabel("Motive"); ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("charts/chart3_motive_bar.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
df["tone"].value_counts().plot(kind="bar", ax=ax, color=["#2196F3", "#FF9800"], edgecolor="white")
ax.set_title("Tone Comparison Across Sources")
ax.set_xlabel("Tone"); ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig("charts/chart4_tone_bar.png", dpi=150)
plt.close()

all_kw = [kw for kws in df["keywords"] for kw in kws]
top_kw = Counter(all_kw).most_common(10)
if top_kw:
    kw_words, kw_freq = zip(*top_kw)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(kw_words, kw_freq, color="mediumpurple")
    ax.set_title("Top 10 Keywords Across All Blogs")
    ax.set_xlabel("Frequency"); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("charts/chart5_top_keywords.png", dpi=150)
    plt.close()

all_text = " ".join(df["clean_text"].dropna())
wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis", max_words=100).generate(all_text)
fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
ax.set_title("Word Cloud — Most Common Blog Keywords", fontsize=14)
plt.tight_layout()
plt.savefig("charts/wordcloud.png", dpi=150)
plt.close()

charts_table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", show_header=True)
charts_table.add_column("File",        style="cyan")
charts_table.add_column("Description")
for fname, desc in [
    ("charts/chart1_sentiment_pie.png",           "Sentiment distribution pie chart"),
    ("charts/chart2_sentiment_vs_popularity.png", "Sentiment score vs popularity scatter"),
    ("charts/chart3_motive_bar.png",              "Blog motive distribution bar chart"),
    ("charts/chart4_tone_bar.png",                "Tone comparison bar chart"),
    ("charts/chart5_top_keywords.png",            "Top 10 keywords horizontal bar chart"),
    ("charts/wordcloud.png",                      "Word cloud of blog keywords"),
]:
    charts_table.add_row(fname, desc)
console.print(charts_table)


console.rule("[bold]Summary[/bold]")

summary = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
summary.add_column("Field",  style="bold cyan", width=22)
summary.add_column("Value",  style="white")

src_dist  = df['source'].value_counts().to_dict()
mot_dist  = df['motive'].value_counts().to_dict()
tone_dist = df['tone'].value_counts().to_dict()
sent_dist = df['generated_sentiment'].value_counts().to_dict()

summary.add_row("Keyword",            keyword)
summary.add_row("Total Articles",     str(len(df)))
summary.add_row("Sources",            "  ".join(f"{k}: {v}" for k, v in src_dist.items()))
summary.add_row("Motives",            "  ".join(f"{k}: {v}" for k, v in mot_dist.items()))
summary.add_row("Tones",              "  ".join(f"{k}: {v}" for k, v in tone_dist.items()))
summary.add_row("Sentiment",          "  ".join(f"{k}: {v}" for k, v in sent_dist.items()))
summary.add_row("Avg Popularity",     f"{df['comment_count'].mean():.1f}  (simulated proxy)")

console.print(Panel(summary, title="[bold]Project Results[/bold]", border_style="cyan"))

console.print("\n[bold green]Sample Comments[/bold green]")
sample_table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan")
sample_table.add_column("Title",   width=38, no_wrap=True)
sample_table.add_column("Comment", width=55)
for _, row in df[["title", "generated_comment"]].head(5).iterrows():
    sample_table.add_row(row["title"][:38], row["generated_comment"][:55])
console.print(sample_table)

console.rule("[bold green]Pipeline Complete[/bold green]")
