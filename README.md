This project runs a keyword-driven web scraping and NLP pipeline using `dm_ccp.py`.

## What It Does

- Collects articles from Wikipedia, Dev.to, and HackerNews
- Cleans and preprocesses text
- Runs TF-IDF, LDA topic modeling, motive and tone detection
- Generates reader-style comments using Flan-T5
- Performs sentiment analysis on scraped and generated comments
- Saves charts and a word cloud in the `charts/` folder (created after successful pipeline execution)

## Requirements

- Python 3.10+ (recommended)
- Internet connection (APIs + model download on first run)

## Setup

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python dm_ccp_fixed.py
```

When prompted, enter a keyword (example: `neural networks`).

## Output

- Terminal tables for data collection, NLP, sentiment, and summary
- Image files saved in `charts/`:
  - `chart1_sentiment_pie.png`
  - `chart2_sentiment_vs_popularity.png`
  - `chart3_motive_bar.png`
  - `chart4_tone_bar.png`
  - `chart5_top_keywords.png`
  - `wordcloud.png`

## Notes

- First run may take longer because `google/flan-t5-base` is downloaded.
- The script also attempts package installation internally; using `requirements.txt` first is recommended for cleaner setup.
