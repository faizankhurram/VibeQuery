# 🔍 VibeQuery

This project is an end-to-end keyword-based web scraping and NLP analytics system that collects articles from multiple sources, analyzes their content using classical NLP + transformer models, and generates structured insights with visualizations.

It is executed via `dm_ccp.py`.

---

## ✨ What It Does

Given a user-provided keyword, the system builds a complete analytical pipeline:

### 📡 Data Collection
- Scrapes and aggregates articles from:
  - Wikipedia (knowledge base articles)
  - Dev.to (developer blogs via API)
  - HackerNews (tech discussions + stories)
- Applies fallback generation if data is insufficient

---

### 🧹 Text Processing
- Cleans and normalizes raw text
- Removes stopwords and noise
- Prepares structured corpus for analysis

---

### 🧠 NLP & Topic Modeling
- TF-IDF keyword extraction
- LDA topic modeling for thematic discovery
- Automatic classification of:
  - Content motive (Tutorial / Opinion / Academic / Informative / Promotional)
  - Tone (Objective / Subjective)

---

### 🤖 AI-Generated Content
- Uses Flan-T5 (google/flan-t5-base) to generate:
  - Human-like reader comments for each article

---

### 📊 Sentiment Analysis
- Performs sentiment scoring on:
  - Scraped comments
  - AI-generated comments
- Classifies sentiment as:
  - Positive
  - Neutral
  - Negative

---

### 📈 Visual Analytics
- Sentiment distribution (pie chart)
- Sentiment vs popularity (scatter plot)
- Motive distribution (bar chart)
- Tone comparison across sources
- Top keywords (horizontal bar chart)
- Word cloud of corpus terms

---

## ⚙️ Requirements

- Python 3.10+
- Internet connection (APIs + model download on first run)
- ~250MB download for Flan-T5 model

---

## 🛠️ Setup (for running locally)

```bash
python -m pip install -r requirements.txt
python dm_ccp.py
```

## 📦 Output Structure
- Data source summary (Wikipedia, Dev.to, HackerNews)
- Article collection table
- NLP analysis results (TF-IDF + LDA topics)
- Motive and tone classification tables
- Sentiment analysis results (scraped + generated)
- Final aggregated summary statistics

---

## ⚠️ Notes

- First execution may take longer due to:
  - Downloading `google/flan-t5-base` (~250MB)
  - Initial NLTK resource setup