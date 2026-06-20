"""
VibeQuery — Streamlit frontend for the keyword NLP analytics pipeline.

Run with:
    source .venv/bin/activate
    streamlit run app.py
"""

import io
import time
import random
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import pipeline as pl

# ─── Page config (must be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="VibeQuery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Session-state initialisation ────────────────────────────────────────────
_DEFAULTS = {
    "app_state": "idle",   # idle | running | done
    "keyword": "",
    "df": None,
    "topics": None,
    "src_counts": {},
    "fetched_count": 0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _reset_analysis():
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default
    if "kw_input" in st.session_state:
        del st.session_state.kw_input

# ─── Design tokens (dark theme only) ─────────────────────────────────────────
BG      = "#0D0D1A"
BG2     = "#13132A"
BG3     = "#1A1A3A"
BORDER  = "rgba(124,58,237,0.22)"
ACC1    = "#7C3AED"
ACC2    = "#3B82F6"
T1      = "#E8E6F0"
T2      = "#9B97B8"
T3      = "#6E6A88"
OK      = "#10B981"
WARN    = "#F59E0B"
DANGER  = "#EF4444"
PTMPL   = "plotly_dark"
PBGC    = "#13132A"
PFONT   = "#E8E6F0"
WC_BG   = "#0D0D1A"
WC_MAP  = "cool"

# ─── CSS injection ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ── Base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, .stApp {{
    background: {BG} !important;
    color: {T1} !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}}
/* ── Remove Streamlit chrome ── */
#MainMenu, header[data-testid="stHeader"], footer,
.stDeployButton, [data-testid="collapsedControl"],
section[data-testid="stSidebar"] {{ display: none !important; }}

/* ── Block container ── */
.main .block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}
/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {ACC1}55; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACC1}; }}

/* ── Keyword input ── */
.stTextInput label {{ display: none !important; }}
.stTextInput > div > div {{
    background: {BG2} !important;
    border: 2px solid {BORDER} !important;
    border-radius: 16px !important;
    transition: border-color .3s ease, box-shadow .3s ease !important;
    padding: 4px 6px !important;
}}
.stTextInput > div > div:focus-within {{
    border-color: {ACC1} !important;
    box-shadow: 0 0 0 4px {ACC1}22 !important;
}}
.stTextInput input {{
    background: transparent !important;
    color: {T1} !important;
    font-size: 1.05rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 14px 18px !important;
    border: none !important;
    outline: none !important;
    caret-color: {ACC1} !important;
}}
.stTextInput input::placeholder {{ color: {T3} !important; }}

/* ── All buttons base ── */
.stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all .2s ease !important;
    cursor: pointer !important;
}}
/* ── Analyse button (primary type) ── */
[data-testid="baseButton-primary"] {{
    background: linear-gradient(135deg, {ACC1}, {ACC2}) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 14px 36px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: .3px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px {ACC1}44 !important;
}}
[data-testid="baseButton-primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px {ACC1}66 !important;
}}
[data-testid="baseButton-primary"]:active {{
    transform: translateY(0) !important;
}}
/* ── Secondary / reset buttons ── */
[data-testid="baseButton-secondary"] {{
    background: {BG3} !important;
    color: {T1} !important;
    border: 1px solid {BORDER} !important;
    font-size: .85rem !important;
    padding: 7px 18px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}}
[data-testid="baseButton-secondary"]:hover {{
    background: {ACC1}22 !important;
    border-color: {ACC1} !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* ── Progress bar ── */
.stProgress > div > div {{
    background: {BG3} !important;
    border-radius: 4px !important;
    height: 6px !important;
}}
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {ACC1}, {ACC2}) !important;
    border-radius: 4px !important;
}}

/* ── Plotly chart wrapper ── */
[data-testid="stPlotlyChart"] > div {{
    border-radius: 16px !important;
    border: 1px solid {BORDER} !important;
    overflow: hidden !important;
    background: {BG2} !important;
}}

/* ── Image ── */
[data-testid="stImage"] img {{
    border-radius: 16px !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Alert / success ── */
.stSuccess, .stInfo, .stWarning, .stError {{ border-radius: 10px !important; }}

/* ── HR ── */
hr {{ border: none !important; border-top: 1px solid {BORDER} !important; margin: 28px 0 !important; }}

/* ═══════════════════════════════════
   CUSTOM COMPONENT STYLES
═══════════════════════════════════ */

/* ── Tooltip ── */
#kw-tip {{
    position: fixed; z-index: 9999;
    background: {BG2};
    border: 1px solid {ACC1}66;
    border-radius: 12px;
    padding: 9px 16px;
    font: 500 .82rem 'Inter', sans-serif;
    color: {T2};
    box-shadow: 0 8px 28px rgba(0,0,0,.28);
    pointer-events: none;
    display: none;
    white-space: nowrap;
    max-width: 340px;
    opacity: 0;
    transition: opacity .4s ease;
}}

/* ── Nav bar ── */
.ie-nav {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 40px;
    background: {BG2};
    border-bottom: 1px solid {BORDER};
    position: sticky; top: 0; z-index: 100;
}}
.ie-logo {{
    font-size: 1.05rem; font-weight: 800; letter-spacing: -.4px;
    background: linear-gradient(90deg, {ACC1}, {ACC2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.ie-nav-right {{
    display: flex; align-items: center; gap: 10px;
}}
.ie-nav-tag {{
    font-size: .72rem; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: {T3};
    padding: 4px 12px;
    border: 1px solid {BORDER};
    border-radius: 20px;
}}

/* ── Hero ── */
.ie-hero {{
    padding: 72px 24px 48px;
    text-align: center;
    background:
        radial-gradient(ellipse 80% 60% at 50% 0%, {ACC1}14 0%, transparent 70%),
        {BG};
}}
.ie-title {{
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 800; line-height: 1.14;
    letter-spacing: -1.5px; color: {T1};
    margin-bottom: 38px;
    animation: fadeInDown .75s ease;
}}
.ie-title span {{
    background: linear-gradient(135deg, {ACC1}, {ACC2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}

/* ── Stage tracker timeline ── */
.ie-timeline {{
    display: flex; align-items: center;
    justify-content: center; flex-wrap: wrap;
    gap: 4px; margin: 0 auto 32px;
    max-width: 520px;
}}
.ie-dot {{
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .68rem; font-weight: 700;
    transition: all .4s ease; flex-shrink: 0;
}}
.ie-dot.pending  {{ background: {BG3}; color: {T3}; border: 2px solid {BORDER}; }}
.ie-dot.running  {{ background: linear-gradient(135deg, {ACC1}, {ACC2}); color: #fff; animation: pulse-dot 1.2s ease-in-out infinite; }}
.ie-dot.complete {{ background: {OK}; color: #fff; }}
.ie-line {{ height: 2px; width: 18px; background: {BORDER}; transition: background .4s ease; flex-shrink: 0; }}
.ie-line.complete {{ background: {OK}44; }}
@keyframes pulse-dot {{
    0%,100% {{ box-shadow: 0 0 0 0 {ACC1}55; }}
    50%      {{ box-shadow: 0 0 0 7px {ACC1}00; }}
}}

/* ── Stage card ── */
.ie-stage {{
    display: flex; align-items: center; gap: 16px;
    background: {BG2}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 16px 20px;
    margin-bottom: 10px; transition: all .4s ease;
}}
.ie-stage.pending {{ opacity: .42; }}
.ie-stage.running {{
    border-color: {ACC1}88; background: {BG3};
    box-shadow: 0 0 0 1px {ACC1}44, 0 4px 24px {ACC1}18;
    animation: stage-glow 2s ease-in-out infinite;
}}
.ie-stage.complete {{ border-left: 3px solid {OK}; }}
@keyframes stage-glow {{
    0%,100% {{ box-shadow: 0 0 0 1px {ACC1}44, 0 4px 24px {ACC1}12; }}
    50%      {{ box-shadow: 0 0 0 1px {ACC1}88, 0 4px 32px {ACC1}28; }}
}}
.ie-stage-icon {{
    width: 42px; height: 42px; border-radius: 11px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 1.15rem;
}}
.ie-stage-icon.pending  {{ background: {BG3}; }}
.ie-stage-icon.running  {{ background: linear-gradient(135deg, {ACC1}33, {ACC2}33); }}
.ie-stage-icon.complete {{ background: {OK}22; }}
.ie-stage-body {{ flex: 1; min-width: 0; }}
.ie-stage-name {{ font-size: .93rem; font-weight: 600; color: {T1}; margin-bottom: 2px; }}
.ie-stage-status {{ font-size: .77rem; color: {T3}; }}
.ie-stage-status.running  {{ color: {ACC1}; }}
.ie-stage-status.complete {{ color: {OK}; }}
.ie-badge-pill {{
    font-size: .71rem; font-weight: 600;
    padding: 3px 12px; border-radius: 20px; flex-shrink: 0;
}}
.ie-badge-pill.pending  {{ background: {BG3}; color: {T3}; }}
.ie-badge-pill.running  {{ background: {ACC1}22; color: {ACC1}; }}
.ie-badge-pill.complete {{ background: {OK}22; color: {OK}; }}
.spin {{ display: inline-block; animation: spinning .75s linear infinite; }}
@keyframes spinning {{ to {{ transform: rotate(360deg); }} }}

/* ── Section header ── */
.ie-section-hdr {{
    font-size: 1.2rem; font-weight: 700; color: {T1};
    display: flex; align-items: center; gap: 10px;
    margin: 34px 0 16px;
}}
.ie-section-hdr::before {{
    content: ''; display: inline-block;
    width: 4px; height: 22px; border-radius: 2px;
    background: linear-gradient(180deg, {ACC1}, {ACC2});
}}

/* ── Article cards ── */
.ie-article {{
    display: flex; align-items: flex-start; gap: 14px;
    background: {BG2}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 14px 18px; margin-bottom: 8px;
    transition: all .2s ease; animation: fadeInUp .35s ease;
}}
.ie-article:hover {{ transform: translateY(-2px); border-color: {ACC1}55; box-shadow: 0 4px 18px {ACC1}18; }}
.ie-art-num {{
    width: 28px; height: 28px; flex-shrink: 0; margin-top: 1px;
    background: linear-gradient(135deg, {ACC1}2A, {ACC2}2A);
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    font-size: .73rem; font-weight: 700; color: {ACC1};
}}
.ie-art-body {{ flex: 1; min-width: 0; }}
.ie-art-title {{
    font-size: .9rem; font-weight: 600; color: {T1}; margin-bottom: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.ie-art-meta {{ font-size: .74rem; color: {T3}; display: flex; gap: 10px; flex-wrap: wrap; }}
.ie-src-pill {{
    display: inline-block; padding: 2px 9px; border-radius: 12px;
    font-size: .71rem; font-weight: 600; flex-shrink: 0;
}}
.src-Wikipedia  {{ background: {ACC2}22; color: {ACC2}; }}
.src-Devto      {{ background: {ACC1}22; color: {ACC1}; }}
.src-HackerNews {{ background: {WARN}22; color: {WARN}; }}
.src-Backup     {{ background: {T3}22;  color: {T3};  }}

/* ── Topic cards ── */
.ie-topic {{
    background: {BG2}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 16px 18px;
    animation: fadeInUp .4s ease;
}}
.ie-topic-label {{
    font-size: .72rem; font-weight: 700; color: {ACC1};
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;
}}
.ie-kw-tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.ie-kw-tag {{
    background: {ACC1}18; border: 1px solid {ACC1}33; color: {ACC1};
    padding: 3px 10px; border-radius: 20px; font-size: .78rem; font-weight: 500;
}}

/* ── Insight cards ── */
.ie-insights {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px; margin-bottom: 28px;
}}
.ie-insight {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 22px 16px; text-align: center;
    transition: all .25s ease; animation: fadeInUp .45s ease;
}}
.ie-insight:hover {{ transform: translateY(-3px); border-color: {ACC1}66; box-shadow: 0 8px 28px {ACC1}18; }}
.ie-insight-val {{
    font-size: 2.1rem; font-weight: 800; margin-bottom: 6px;
    background: linear-gradient(135deg, {ACC1}, {ACC2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.ie-insight-lbl {{
    font-size: .75rem; color: {T2}; font-weight: 500;
    text-transform: uppercase; letter-spacing: .8px;
}}

/* ── Comment cards ── */
.ie-comment {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 20px 14px; margin-bottom: 10px;
    position: relative; overflow: hidden;
    transition: all .2s ease; animation: fadeInUp .4s ease;
}}
.ie-comment:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px {ACC1}14; }}
.ie-comment::before {{
    content: '"'; position: absolute; top: 6px; left: 14px;
    font-size: 3rem; color: {ACC1}1A;
    font-family: Georgia, serif; line-height: 1;
}}
.ie-cmt-src {{ font-size: .73rem; color: {T3}; margin-bottom: 6px; padding-left: 12px; text-transform: uppercase; letter-spacing: .5px; }}
.ie-cmt-text {{ font-size: .9rem; color: {T1}; line-height: 1.55; padding-left: 12px; font-style: italic; }}
.ie-sent-tag {{
    display: inline-block; margin: 10px 0 0 12px;
    padding: 2px 10px; border-radius: 12px;
    font-size: .72rem; font-weight: 600;
}}
.sent-Positive {{ background: {OK}22; color: {OK}; }}
.sent-Neutral  {{ background: {WARN}22; color: {WARN}; }}
.sent-Negative {{ background: {DANGER}22; color: {DANGER}; }}

/* ── Summary KPI bar ── */
.ie-summary-hdr {{
    text-align: center; padding: 48px 0 28px;
    border-top: 1px solid {BORDER};
}}
.ie-summary-hdr h2 {{
    font-size: 1.7rem; font-weight: 700; color: {T1};
    letter-spacing: -.5px; margin-bottom: 6px;
}}
.ie-summary-hdr p {{ color: {T2}; font-size: .93rem; }}

/* ── Pipeline header ── */
.ie-pipeline-hdr {{
    text-align: center; padding: 40px 24px 28px;
    background: radial-gradient(ellipse 60% 40% at 50% 0%, {ACC1}11 0%, transparent 70%),
                {BG};
}}
.ie-pipeline-hdr h2 {{
    font-size: 1.5rem; font-weight: 700; color: {T1}; margin-bottom: 6px;
    letter-spacing: -.4px;
}}
.ie-pipeline-hdr p {{ color: {T2}; font-size: .92rem; }}

/* ── Animations ── */
@keyframes fadeInDown {{
    from {{ opacity: 0; transform: translateY(-18px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(18px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

/* ── Disabled button ── */
[data-testid="baseButton-primary"]:disabled {{
    opacity: .45 !important;
    cursor: not-allowed !important;
}}
</style>
""", unsafe_allow_html=True)


# ─── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_generator():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("text2text-generation", model="google/flan-t5-base")


# ─── HTML helpers ─────────────────────────────────────────────────────────────

def _timeline(states: list) -> str:
    dots = []
    for i, s in enumerate(states):
        dots.append(f'<div class="ie-dot {s}">{i + 1}</div>')
        if i < len(states) - 1:
            lc = "complete" if s == "complete" else ""
            dots.append(f'<div class="ie-line {lc}"></div>')
    return f'<div class="ie-timeline">{"".join(dots)}</div>'


def _stage_card(title: str, icon: str, status: str, detail: str) -> str:
    spinner = '<span class="spin">⟳</span>&nbsp;' if status == "running" else ""
    icon_show = "✅" if status == "complete" else icon
    labels = {"pending": "Waiting", "running": "Running…", "complete": "Done"}
    return f"""
<div class="ie-stage {status}">
  <div class="ie-stage-icon {status}">{icon_show}</div>
  <div class="ie-stage-body">
    <div class="ie-stage-name">{title}</div>
    <div class="ie-stage-status {status}">{spinner}{detail}</div>
  </div>
  <span class="ie-badge-pill {status}">{labels[status]}</span>
</div>"""


def _section_hdr(text: str) -> str:
    return f'<div class="ie-section-hdr">{text}</div>'


def _article_cards(df: pd.DataFrame) -> str:
    html = []
    for i, row in df.head(12).iterrows():
        src_cls = "src-" + row["source"].replace(".", "").replace(" ", "")
        html.append(f"""
<div class="ie-article">
  <div class="ie-art-num">{i + 1}</div>
  <div class="ie-art-body">
    <div class="ie-art-title">{row['title'][:70]}</div>
    <div class="ie-art-meta">
      <span>{row['author'][:24]}</span>
      <span>{row['date']}</span>
    </div>
  </div>
  <span class="ie-src-pill {src_cls}">{row['source']}</span>
</div>""")
    return "\n".join(html)


def _topic_cards(topics: list) -> str:
    parts = []
    for i, kws in enumerate(topics):
        tags = "".join(f'<span class="ie-kw-tag">{w}</span>' for w in kws)
        parts.append(f"""
<div class="ie-topic">
  <div class="ie-topic-label">Topic {i + 1}</div>
  <div class="ie-kw-tags">{tags}</div>
</div>""")
    return "\n".join(parts)


def _insight_cards(df: pd.DataFrame) -> str:
    pos = (df["generated_sentiment"] == "Positive").sum()
    neg = (df["generated_sentiment"] == "Negative").sum()
    avg = df["sentiment_score"].mean()
    top = Counter(kw for kws in df["keywords"] for kw in kws).most_common(1)
    top_kw = top[0][0] if top else "—"
    return f"""
<div class="ie-insights">
  <div class="ie-insight"><div class="ie-insight-val">{len(df)}</div><div class="ie-insight-lbl">Articles</div></div>
  <div class="ie-insight"><div class="ie-insight-val">{pos}</div><div class="ie-insight-lbl">Positive</div></div>
  <div class="ie-insight"><div class="ie-insight-val">{neg}</div><div class="ie-insight-lbl">Negative</div></div>
  <div class="ie-insight"><div class="ie-insight-val">{avg:+.2f}</div><div class="ie-insight-lbl">Avg Score</div></div>
  <div class="ie-insight"><div class="ie-insight-val" style="font-size:1.25rem">{top_kw}</div><div class="ie-insight-lbl">Top Keyword</div></div>
</div>"""


def _comment_cards(df: pd.DataFrame) -> str:
    html = []
    for _, row in df.head(8).iterrows():
        sent = row.get("generated_sentiment", "Neutral")
        html.append(f"""
<div class="ie-comment">
  <div class="ie-cmt-src">{row['title'][:55]}</div>
  <div class="ie-cmt-text">{row['generated_comment'][:160]}</div>
  <span class="ie-sent-tag sent-{sent}">{sent}</span>
</div>""")
    return "\n".join(html)


# ─── Chart factories ──────────────────────────────────────────────────────────
_SENT_COLORS = {"Positive": "#10B981", "Neutral": "#F59E0B", "Negative": "#EF4444"}


def _chart_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["generated_sentiment"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(
            colors=[_SENT_COLORS.get(k, "#94A3B8") for k in counts.index],
            line=dict(color=PBGC, width=3),
        ),
        textinfo="label+percent",
        textfont=dict(color=PFONT, size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        template=PTMPL, paper_bgcolor=PBGC,
        font=dict(color=PFONT),
        title=dict(text="Sentiment Distribution", x=0.5, xanchor="center",
                   font=dict(size=15, color=PFONT)),
        legend=dict(orientation="h", y=-0.08, font=dict(color=PFONT)),
        margin=dict(l=20, r=20, t=60, b=50), height=340,
    )
    return fig


def _chart_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="sentiment_score", y="comment_count",
        color="generated_sentiment", size="comment_count", size_max=28,
        color_discrete_map=_SENT_COLORS,
        hover_data={"title": True, "source": True,
                    "sentiment_score": ":.3f", "comment_count": True},
        template=PTMPL,
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color=PBGC)))
    fig.update_layout(
        paper_bgcolor=PBGC, plot_bgcolor=PBGC,
        font=dict(color=PFONT),
        title=dict(text="Sentiment Score vs Popularity", x=0.5, xanchor="center",
                   font=dict(size=15, color=PFONT)),
        xaxis=dict(title="Sentiment Score", color=PFONT,
                   gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(title="Comment Count", color=PFONT,
                   gridcolor=BORDER),
        legend_title_text="Sentiment",
        legend=dict(orientation="h", y=-0.18, font=dict(color=PFONT)),
        margin=dict(l=50, r=20, t=60, b=80), height=370,
    )
    return fig


def _chart_motive(df: pd.DataFrame) -> go.Figure:
    mc = df["motive"].value_counts().sort_values(ascending=True).reset_index()
    mc.columns = ["motive", "count"]
    fig = px.bar(
        mc, x="count", y="motive", orientation="h",
        color="count", color_continuous_scale="Purples",
        color_continuous_midpoint=mc["count"].median(),
        text="count", template=PTMPL,
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=PFONT, size=13),
        marker=dict(line=dict(width=1, color=BORDER)),
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=PBGC, plot_bgcolor=PBGC,
        font=dict(color=PFONT),
        title=dict(text="Blog Motive Distribution", x=0.5, xanchor="center",
                   font=dict(size=15, color=PFONT)),
        xaxis=dict(title="Count", color=PFONT, gridcolor=BORDER),
        yaxis=dict(title="", color=PFONT),
        coloraxis_showscale=False,
        margin=dict(l=20, r=50, t=60, b=40), height=320,
    )
    return fig


def _chart_tone(df: pd.DataFrame) -> go.Figure:
    tc = df["tone"].value_counts().sort_values(ascending=True).reset_index()
    tc.columns = ["tone", "count"]
    palette = [ACC1, ACC2]
    fig = go.Figure(go.Bar(
        x=tc["count"], y=tc["tone"], orientation="h",
        marker=dict(color=palette[:len(tc)], line=dict(width=1, color=BORDER)),
        text=tc["count"], textposition="outside",
        textfont=dict(color=PFONT, size=13),
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
    ))
    fig.update_layout(
        template=PTMPL, paper_bgcolor=PBGC, plot_bgcolor=PBGC,
        font=dict(color=PFONT),
        title=dict(text="Tone Distribution", x=0.5, xanchor="center",
                   font=dict(size=15, color=PFONT)),
        xaxis=dict(title="Count", color=PFONT, gridcolor=BORDER),
        yaxis=dict(title="", color=PFONT),
        margin=dict(l=20, r=50, t=60, b=40), height=280,
    )
    return fig


def _chart_keywords(df: pd.DataFrame):
    all_kw = [k for kws in df["keywords"] for k in kws]
    top_kw = Counter(all_kw).most_common(10)
    if not top_kw:
        return None
    kdf = pd.DataFrame(top_kw, columns=["keyword", "frequency"]).sort_values(
        "frequency", ascending=True
    )
    fig = px.bar(
        kdf, x="frequency", y="keyword", orientation="h",
        color="frequency", color_continuous_scale="Viridis",
        text="frequency", template=PTMPL,
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=PFONT, size=13),
        marker=dict(line=dict(width=1, color=BORDER)),
        hovertemplate="<b>%{y}</b><br>Frequency: %{x}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=PBGC, plot_bgcolor=PBGC,
        font=dict(color=PFONT),
        title=dict(text="Top 10 Keywords Across All Articles", x=0.5,
                   xanchor="center", font=dict(size=15, color=PFONT)),
        xaxis=dict(title="Frequency", color=PFONT, gridcolor=BORDER),
        yaxis=dict(title="", color=PFONT),
        coloraxis_showscale=False,
        margin=dict(l=20, r=50, t=60, b=40), height=390,
    )
    return fig


def _wordcloud_image(df: pd.DataFrame) -> io.BytesIO | None:
    all_text = " ".join(df["clean_text"].dropna())
    if not all_text.strip():
        return None
    wc = WordCloud(
        width=1200, height=440, background_color=WC_BG,
        colormap=WC_MAP, max_words=120,
    ).generate(all_text)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor(WC_BG)
    ax.set_facecolor(WC_BG)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=WC_BG)
    plt.close()
    buf.seek(0)
    return buf


# ─── Summary section (standalone, used by both pipeline run and rerun) ────────
def _render_summary(df: pd.DataFrame):
    st.markdown("""
<div class="ie-summary-hdr">
  <h2>Pipeline Summary</h2>
  <p>All stages completed successfully</p>
</div>""", unsafe_allow_html=True)

    src_d  = df["source"].value_counts().to_dict()
    mot_d  = df["motive"].value_counts().to_dict()
    tone_d = df["tone"].value_counts().to_dict()
    sent_d = df["generated_sentiment"].value_counts().to_dict()
    avg_p  = df["comment_count"].mean()

    rows = [
        ("Total Articles",      str(len(df))),
        ("Sources",             " · ".join(f"{k} ({v})" for k, v in src_d.items())),
        ("Motives",             " · ".join(f"{k}: {v}" for k, v in mot_d.items())),
        ("Tones",               " · ".join(f"{k}: {v}" for k, v in tone_d.items())),
        ("Sentiment",           " · ".join(f"{k}: {v}" for k, v in sent_d.items())),
        ("Avg Popularity Proxy", f"{avg_p:.1f}"),
    ]
    summary_html = "".join(
        f"""<div style="display:flex;align-items:center;gap:12px;padding:12px 0;
            border-bottom:1px solid {BORDER};">
          <span style="min-width:170px;font-weight:600;font-size:.85rem;color:{T2};
            text-transform:uppercase;letter-spacing:.6px">{k}</span>
          <span style="font-size:.9rem;color:{T1}">{v}</span>
        </div>"""
        for k, v in rows
    )
    st.markdown(
        f'<div style="background:{BG2};border:1px solid {BORDER};border-radius:16px;'
        f'padding:8px 24px;margin-bottom:60px">{summary_html}</div>',
        unsafe_allow_html=True,
    )


# ─── Full results renderer — used on theme-toggle reruns (pipeline already done)
def render_results(df: pd.DataFrame, topics: list):
    # Articles
    st.markdown(_section_hdr("Collected Articles"), unsafe_allow_html=True)
    st.markdown(_article_cards(df), unsafe_allow_html=True)

    # Keywords
    kw_fig = _chart_keywords(df)
    if kw_fig:
        st.markdown(_section_hdr("Top Keywords"), unsafe_allow_html=True)
        st.plotly_chart(kw_fig, use_container_width=True)

    # Topics
    st.markdown(_section_hdr("LDA Topics Discovered"), unsafe_allow_html=True)
    t_cols = st.columns(len(topics))
    for col, block in zip(t_cols, [_topic_cards([t]) for t in topics]):
        with col:
            st.markdown(block, unsafe_allow_html=True)

    # Motive + Tone
    st.markdown(_section_hdr("Motive & Tone Analysis"), unsafe_allow_html=True)
    mc, tc = st.columns(2)
    with mc:
        st.plotly_chart(_chart_motive(df), use_container_width=True)
    with tc:
        st.plotly_chart(_chart_tone(df), use_container_width=True)

    # Sentiment + KPIs
    st.markdown(_section_hdr("Sentiment Analysis"), unsafe_allow_html=True)
    st.markdown(_insight_cards(df), unsafe_allow_html=True)
    dc, sc = st.columns([2, 3])
    with dc:
        st.plotly_chart(_chart_donut(df), use_container_width=True)
    with sc:
        st.plotly_chart(_chart_scatter(df), use_container_width=True)

    # Comments
    st.markdown(_section_hdr("AI-Generated Reader Comments"), unsafe_allow_html=True)
    st.markdown(_comment_cards(df), unsafe_allow_html=True)

    # Word cloud
    wc_buf = _wordcloud_image(df)
    if wc_buf:
        st.markdown(_section_hdr("Keyword Word Cloud"), unsafe_allow_html=True)
        st.image(wc_buf, use_container_width=True)

    # Summary
    _render_summary(df)


# ─── Pipeline executor with true per-stage progressive rendering ──────────────
def run_pipeline(keyword: str):
    STAGES = [
        ("Data Collection",           "🌐"),
        ("Comment Enrichment",        "💬"),
        ("Text Preprocessing",        "🧹"),
        ("TF-IDF Keyword Extraction", "📊"),
        ("LDA Topic Modelling",       "🧠"),
        ("Motive & Tone Detection",   "🎭"),
        ("Sentiment Analysis",        "💡"),
        ("AI Comment Generation",     "🤖"),
        ("Visualisation",             "🎨"),
    ]
    n = len(STAGES)
    states = ["pending"] * n

    # ── Pipeline header ───────────────────────────────────────────────────────
    st.markdown(f"""
<div class="ie-pipeline-hdr">
  <h2>Running Analysis</h2>
  <p>Keyword: <strong style="color:{ACC1}">{keyword}</strong></p>
</div>""", unsafe_allow_html=True)

    timeline_slot = st.empty()
    prog = st.progress(0)

    _, c, _ = st.columns([1, 4, 1])
    with c:
        stage_slots = [st.empty() for _ in range(n)]

    # ── Result slots pre-allocated BELOW the stage cards ─────────────────────
    # Each fills in as its stage completes — gives true progressive rendering.
    slot_articles    = st.empty()   # Stage 0
    slot_keywords    = st.empty()   # Stage 3
    slot_topics      = st.empty()   # Stage 4
    slot_motive_tone = st.empty()   # Stage 5
    slot_sentiment   = st.empty()   # Stage 7
    slot_comments    = st.empty()   # Stage 7
    slot_wordcloud   = st.empty()   # Stage 8
    slot_summary     = st.empty()   # Stage 8

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _start(i, detail):
        states[i] = "running"
        timeline_slot.markdown(_timeline(states), unsafe_allow_html=True)
        with c:
            stage_slots[i].markdown(
                _stage_card(STAGES[i][0], STAGES[i][1], "running", detail),
                unsafe_allow_html=True,
            )

    def _done(i, detail):
        states[i] = "complete"
        timeline_slot.markdown(_timeline(states), unsafe_allow_html=True)
        prog.progress((i + 1) / n)
        with c:
            stage_slots[i].markdown(
                _stage_card(STAGES[i][0], STAGES[i][1], "complete", detail),
                unsafe_allow_html=True,
            )

    # render all pending initially
    timeline_slot.markdown(_timeline(states), unsafe_allow_html=True)
    with c:
        for i in range(n):
            stage_slots[i].markdown(
                _stage_card(STAGES[i][0], STAGES[i][1], "pending", "Waiting…"),
                unsafe_allow_html=True,
            )

    # ── Stage 0: Data Collection ──────────────────────────────────────────────
    _start(0, "Scraping Wikipedia, Dev.to, HackerNews…")
    all_data, src_counts = [], {}
    for name, fn in [
        ("Wikipedia",  pl.scrape_wikipedia),
        ("Dev.to",     pl.scrape_devto),
        ("HackerNews", pl.scrape_hackernews),
    ]:
        result = fn(keyword)
        all_data += result
        src_counts[name] = len(result)
    df = pl.build_dataframe(all_data, keyword)
    _done(0, f"Collected {len(df)} articles — {', '.join(f'{k}: {v}' for k, v in src_counts.items())}")
    # ▶ Show articles immediately
    with slot_articles.container():
        st.markdown(_section_hdr("Collected Articles"), unsafe_allow_html=True)
        st.markdown(_article_cards(df), unsafe_allow_html=True)

    # ── Stage 1: Comment Enrichment ───────────────────────────────────────────
    _start(1, "Fetching real user comments from APIs…")
    df, fetched_count = pl.enrich_df_comments(df)
    _done(1, f"Enriched {fetched_count} article(s) with live comments")

    # ── Stage 2: Text Preprocessing ───────────────────────────────────────────
    _start(2, "Cleaning and normalising text…")
    df = df.copy()
    df["clean_text"] = df["raw_text"].apply(pl.clean_text)
    _done(2, f"Cleaned {len(df)} documents — stop-words removed")

    # ── Stage 3: TF-IDF ───────────────────────────────────────────────────────
    _start(3, "Fitting TF-IDF vectoriser, extracting keywords…")
    df, features, X = pl.run_tfidf(df)
    top_global = Counter(kw for kws in df["keywords"] for kw in kws).most_common(3)
    _done(3, "Top terms: " + ", ".join(w for w, _ in top_global))
    # ▶ Show keyword bar immediately
    kw_fig = _chart_keywords(df)
    if kw_fig:
        with slot_keywords.container():
            st.markdown(_section_hdr("Top Keywords"), unsafe_allow_html=True)
            st.plotly_chart(kw_fig, use_container_width=True)

    # ── Stage 4: LDA ──────────────────────────────────────────────────────────
    _start(4, "Fitting LDA topic model…")
    topics = pl.run_lda(X, features, len(df))
    _done(4, f"Discovered {len(topics)} latent topics")
    # ▶ Show topics immediately
    with slot_topics.container():
        st.markdown(_section_hdr("LDA Topics Discovered"), unsafe_allow_html=True)
        t_cols = st.columns(len(topics))
        for col, block in zip(t_cols, [_topic_cards([t]) for t in topics]):
            with col:
                st.markdown(block, unsafe_allow_html=True)

    # ── Stage 5: Motive & Tone ────────────────────────────────────────────────
    _start(5, "Classifying article motive and writing tone…")
    df = pl.apply_motive_tone(df)
    _done(5, f"Motives: {df['motive'].value_counts().to_dict()}")
    # ▶ Show motive + tone charts immediately
    with slot_motive_tone.container():
        st.markdown(_section_hdr("Motive & Tone Analysis"), unsafe_allow_html=True)
        mc, tc = st.columns(2)
        with mc:
            st.plotly_chart(_chart_motive(df), use_container_width=True)
        with tc:
            st.plotly_chart(_chart_tone(df), use_container_width=True)

    # ── Stage 6: Sentiment (scraped) ──────────────────────────────────────────
    _start(6, "Running TextBlob sentiment analysis on scraped comments…")
    df = df.copy()
    df["scraped_sentiment"] = df["comments"].apply(pl.get_sentiment)
    pos = (df["scraped_sentiment"] == "Positive").sum()
    neg = (df["scraped_sentiment"] == "Negative").sum()
    _done(6, f"Scraped comments — Positive: {pos}  Negative: {neg}")

    # ── Stage 7: AI Comment Generation ───────────────────────────────────────
    _start(7, "Loading Flan-T5 and generating reader comments…")
    generator = load_generator()
    df = df.copy()
    df["generated_comment"] = df["title"].apply(
        lambda t: pl.generate_comment(t, generator)
    )
    df = pl.apply_sentiment(df)
    _done(7, f"Generated {len(df)} comments — avg score {df['sentiment_score'].mean():+.2f}")
    # ▶ Show sentiment charts + insight KPIs + comments immediately
    with slot_sentiment.container():
        st.markdown(_section_hdr("Sentiment Analysis"), unsafe_allow_html=True)
        st.markdown(_insight_cards(df), unsafe_allow_html=True)
        dc, sc = st.columns([2, 3])
        with dc:
            st.plotly_chart(_chart_donut(df), use_container_width=True)
        with sc:
            st.plotly_chart(_chart_scatter(df), use_container_width=True)
    with slot_comments.container():
        st.markdown(_section_hdr("AI-Generated Reader Comments"), unsafe_allow_html=True)
        st.markdown(_comment_cards(df), unsafe_allow_html=True)

    # ── Stage 8: Visualisation ────────────────────────────────────────────────
    _start(8, "Generating word cloud…")
    wc_buf = _wordcloud_image(df)
    _done(8, "All visualisations ready")
    # ▶ Show wordcloud + summary immediately
    if wc_buf:
        with slot_wordcloud.container():
            st.markdown(_section_hdr("Keyword Word Cloud"), unsafe_allow_html=True)
            st.image(wc_buf, use_container_width=True)

    with slot_summary.container():
        _render_summary(df)

    # ── Persist ───────────────────────────────────────────────────────────────
    st.session_state.df            = df
    st.session_state.topics        = topics
    st.session_state.src_counts    = src_counts
    st.session_state.fetched_count = fetched_count
    st.session_state.app_state     = "done"


# ─── Keyword suggestion tooltip (JS injected into parent doc) ─────────────────
def inject_tooltip_js():
    components.html("""
<script>
(function () {
    const SUGGESTIONS = [
        "💡 Try: artificial intelligence",
        "💡 Try: blockchain trends",
        "💡 Try: climate change data",
        "💡 Try: machine learning ethics",
        "💡 Try: quantum computing",
        "💡 Try: data privacy laws",
        "💡 Try: neural networks",
    ];
    let idx = 0, idleTimer = null, fadeTimer = null;
    const pDoc = window.parent.document;

    function ensureTip() {
        let t = pDoc.getElementById("kw-tip");
        if (!t) {
            t = pDoc.createElement("div");
            t.id = "kw-tip";
            pDoc.body.appendChild(t);
        }
        return t;
    }

    function showTip() {
        const inp = pDoc.querySelector(".stTextInput input");
        if (!inp || inp.value.trim() !== "") return;
        const tip = ensureTip();
        tip.textContent = SUGGESTIONS[idx % SUGGESTIONS.length];
        idx++;
        const rect = inp.getBoundingClientRect();
        tip.style.top  = (rect.bottom + 10) + "px";
        tip.style.left = rect.left + "px";
        tip.style.display = "block";
        requestAnimationFrame(() => { tip.style.opacity = "1"; });
        fadeTimer = setTimeout(() => {
            tip.style.opacity = "0";
            setTimeout(() => { tip.style.display = "none"; }, 420);
        }, 3200);
    }

    function hideTip() {
        clearTimeout(fadeTimer);
        const t = pDoc.getElementById("kw-tip");
        if (t) { t.style.opacity = "0"; setTimeout(() => { t.style.display = "none"; }, 300); }
    }

    function schedule() { clearTimeout(idleTimer); hideTip(); idleTimer = setTimeout(showTip, 7000); }

    function attach() {
        const inp = pDoc.querySelector(".stTextInput input");
        if (!inp) { setTimeout(attach, 600); return; }
        inp.addEventListener("focus", schedule);
        inp.addEventListener("blur",  () => { clearTimeout(idleTimer); hideTip(); });
        inp.addEventListener("input", () => {
            hideTip(); clearTimeout(idleTimer);
            if (inp.value.trim() === "") schedule();
        });
        if (pDoc.activeElement === inp) schedule();
    }
    attach();
})();
</script>
""", height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PAGE RENDER
# ─────────────────────────────────────────────────────────────────────────────

# ── Navigation bar ────────────────────────────────────────────────────────────
show_reset = (
    st.session_state.app_state == "done"
    and st.session_state.df is not None
)
nav_pad_l, nav_l, nav_r, nav_pad_r = st.columns([0.04, 5, 1, 0.04], vertical_alignment="center")
with nav_l:
    st.markdown(
        f'<div style="padding:16px 0;">'
        f'<span class="ie-logo">🧠 VibeQuery</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with nav_r:
    if show_reset:
        st.button(
            "↩ Reset",
            key="nav_reset_btn",
            type="secondary",
            on_click=_reset_analysis,
        )

st.markdown(f'<hr style="margin:0;border-color:{BORDER}">', unsafe_allow_html=True)

# ── Hero section (always visible) ────────────────────────────────────────────
if st.session_state.app_state in ("idle", "done"):
    st.markdown("""
<div class="ie-hero">
  <h1 class="ie-title">Discover <span>Intelligence</span><br>in Any Topic</h1>
</div>""", unsafe_allow_html=True)

    _, inp_col, _ = st.columns([1, 3, 1])
    with inp_col:
        keyword_val = st.text_input(
            label="keyword",
            placeholder="e.g. artificial intelligence, quantum computing, climate data…",
            value=st.session_state.keyword,
            key="kw_input",
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        def _on_analyze():
            kw = st.session_state.kw_input.strip()
            if kw:
                st.session_state.keyword   = kw
                st.session_state.app_state = "running"
                st.session_state.df        = None
                st.session_state.topics    = None

        btn_disabled = not bool(st.session_state.kw_input.strip()) if st.session_state.app_state == "idle" else False
        st.button(
            "Analyze →",
            key="analyze_btn",
            type="primary",
            on_click=_on_analyze,
            disabled=btn_disabled,
            use_container_width=True,
        )

    # Tooltip JS
    inject_tooltip_js()

# ── Pipeline execution (live) ─────────────────────────────────────────────────
if st.session_state.app_state == "running":
    keyword = st.session_state.keyword
    run_pipeline(keyword)

# ── Results (from session state — shown on reruns) ────────────────────────────
elif st.session_state.app_state == "done" and st.session_state.df is not None:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.divider()
    render_results(st.session_state.df, st.session_state.topics)
