"""Streamlit web app for the multi-document RAG system.

Run:
    streamlit run app.py

Features
- Multi-file uploader (PDF, DOCX, TXT, CSV, JSON, MD, HTML) + data/ folder
- Chunk settings (size / overlap / top-k) with a live overlap preview;
  the index is rebuilt only when settings or files actually change
- Per-document search-scope selector
- Chat with source citations, expandable retrieved chunks, /commands
- RAGAS evaluation tab with metric cards, charts, history and CSV export
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

import commands
from commands import is_command, parse_command, render_help

st.set_page_config(
    page_title="Knowledge Desk — Multi-Document RAG",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

def resolved_path(env_value: str, default_relative: str) -> str:
    """Resolve relative paths against the app project root."""
    candidate = env_value or default_relative
    if os.path.isabs(candidate):
        return candidate
    return os.path.abspath(os.path.join(PROJECT_ROOT, candidate))


DATA_DIR = resolved_path(os.getenv("DATA_DIR"), "data")
VECTOR_DIR = resolved_path(os.getenv("VECTOR_STORE_DIR"), "vector_store")
BUILD_CONFIG_PATH = os.path.join(VECTOR_DIR, "build_config.json")
EVAL_DATA_PATH = resolved_path(os.getenv("EVAL_DATA_PATH"), os.path.join("evaluation", "evaluation_data.json"))
RESULTS_PATH = resolved_path(os.getenv("RAGAS_RESULTS_PATH"), os.path.join("results", "ragas_results.csv"))
HISTORY_PATH = resolved_path(os.getenv("RAGAS_HISTORY_PATH"), os.path.join("results", "ragas_history.csv"))

# ---------------------------------------------------------------------------
# Cached resources (models only — never touches st.session_state inside)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """One RAG pipeline per process (embeddings + LLM are expensive to load)."""
    from src.rag_pipeline import RAGPipeline

    return RAGPipeline()


@st.cache_data(show_spinner=False)
def read_build_config(mtime: float) -> dict:
    """Read the chunk settings the on-disk index was built with.
    Keyed on file mtime so edits invalidate the cache automatically."""
    try:
        with open(BUILD_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = {
    "messages": [],          # {"role", "content", "sources", "contexts"}
    "last_sources": [],
    "last_contexts": [],
    "uploads_key": "none",   # names of files currently ingested
    "force_rebuild": False,
    "ingest_error": None,
    "ragas_df": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_chat():
    st.session_state.messages = []
    st.session_state.last_sources = []
    st.session_state.last_contexts = []


# ---------------------------------------------------------------------------
# Custom CSS — subtle polish on top of the default theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      :root {
        --bg: #08111d;
        --bg-2: #0d1626;
        --panel: rgba(16, 24, 38, 0.82);
        --panel-strong: rgba(20, 30, 46, 0.94);
        --line: rgba(148, 163, 184, 0.18);
        --line-strong: rgba(148, 163, 184, 0.32);
        --text: #edf4ff;
        --muted: #a7b8d2;
        --primary: #8aa7ff;
        --primary-2: #66e8d6;
        --primary-3: #baa8ff;
        --shadow: rgba(4, 8, 18, 0.42);
      }

      .stApp {
        background:
          radial-gradient(circle at top left, rgba(138,167,255,0.16), transparent 30%),
          radial-gradient(circle at bottom right, rgba(102,232,214,0.12), transparent 25%),
          linear-gradient(135deg, var(--bg) 0%, var(--bg-2) 55%, #101b2e 100%);
        color: var(--text);
      }

      .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
      }

      .top-shell {
        position: relative;
        overflow: hidden;
        padding: 1.4rem 1.5rem 1.2rem;
        margin: 0 0 1rem 0;
        border-radius: 1.2rem;
        border: 1px solid rgba(138,167,255,0.22);
        background: linear-gradient(135deg, rgba(15,25,38,0.92), rgba(18,30,46,0.86));
        box-shadow: 0 18px 40px rgba(4,8,18,0.32), inset 0 1px 0 rgba(255,255,255,0.06);
      }

      .top-shell::before {
        content: "";
        position: absolute;
        inset: -30% auto auto -8%;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(138,167,255,0.18), transparent 68%);
      }

      .top-shell::after {
        content: "";
        position: absolute;
        right: -40px;
        bottom: -40px;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(102,232,214,0.14), transparent 72%);
      }

      .top-shell .badge {
        position: relative;
        display: inline-block;
        font-size: .74rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: var(--primary-2);
        background: rgba(102,232,214,0.08);
        border: 1px solid rgba(102,232,214,0.18);
        padding: .38rem .6rem;
        border-radius: 999px;
        margin-bottom: .75rem;
      }

      .top-shell h1 {
        position: relative;
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.05;
        letter-spacing: -.05em;
        font-weight: 800;
        background: linear-gradient(90deg, #ecf5ff 0%, #dfe9ff 25%, #b8faf2 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }

      .top-shell p {
        position: relative;
        margin: .55rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        max-width: 720px;
      }

      .stTabs [role="tablist"] {
        gap: 0.6rem;
        border-bottom: 1px solid var(--line);
        margin: 0 0 1rem 0;
      }

      .stTabs [role="tab"] {
        height: 2.8rem;
        border-radius: 0.9rem 0.9rem 0 0;
        padding: 0 1rem;
        font-weight: 600;
        border: 1px solid transparent;
        color: var(--muted);
        background: transparent;
      }

      .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(138,167,255,.10), rgba(102,232,214,.07));
        border-color: rgba(138,167,255,.38);
        color: var(--text);
        box-shadow: inset 0 -1px 0 rgba(138,167,255,.18);
      }

      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11,18,32,0.94), rgba(16,26,45,0.98));
        border-right: 1px solid var(--line);
      }

      [data-testid="stSidebarNav"] {
        background: transparent;
      }

      .sidebar-header {
        font-weight: 800;
        font-size: 1.5rem;
        letter-spacing: -.04em;
        margin-bottom: .35rem;
        background: linear-gradient(90deg, #9fb3ff, #6ee7d8 55%, #c4b5fd);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }

      .stMarkdown, .stTextInput, .stTextArea, .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput {
        color: var(--text);
      }

      [data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(138,167,255,.10), rgba(102,232,214,.05));
        border: 1px solid rgba(138,167,255,.22);
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        box-shadow: 0 16px 30px rgba(4,8,18,0.22);
      }

      [data-testid="stMetricLabel"] p {
        font-size: .74rem !important;
        color: var(--muted) !important;
        letter-spacing: .09em;
        text-transform: uppercase;
      }

      [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
      }

      .stChatMessage {
        border-radius: 1rem;
        border: 1px solid rgba(148,163,184,.16);
        background: rgba(15, 23, 42, 0.54);
        box-shadow: 0 12px 18px rgba(15,23,42,0.12);
        padding: 0.9rem 1rem;
      }

      .stChatMessage[data-testid="stChatMessage"] {
        margin-bottom: 0.8rem;
      }

      .stChatMessage [data-testid="stMarkdownContainer"] p {
        margin: 0;
        color: var(--text);
      }

      .stButton > button,
      .stDownloadButton > button {
        border-radius: 0.9rem;
        border: 1px solid rgba(138,167,255,.28);
        background: linear-gradient(135deg, rgba(138,167,255,.18), rgba(102,232,214,.10));
        color: var(--text);
        font-weight: 600;
        letter-spacing: .01em;
        transition: transform .15s ease, box-shadow .15s ease;
        box-shadow: 0 10px 20px rgba(15,23,42,0.18);
      }

      .stButton > button:hover,
      .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 28px rgba(30,41,59,.24);
        border-color: rgba(102,232,214,.44);
      }

      div[data-testid="stExpander"] {
        border-radius: 1rem;
        border: 1px solid var(--line);
        background: rgba(15,23,42,0.38);
        box-shadow: 0 8px 18px rgba(2,6,23,0.14);
      }

      .stDataFrame, .stDataFrame > div {
        border-radius: 0.8rem;
        background: rgba(15,23,42,0.35);
      }

      .stFileUploader > div {
        border-radius: 1rem;
        border: 1px dashed rgba(138,167,255,.34);
        background: rgba(15,23,42,0.35);
      }

      .stSlider > div > div {
        background: rgba(15,23,42,0.35);
        border-radius: 0.9rem;
      }

      .stMultiSelect, .stSelectbox, .stTextInput, .stTextArea {
        background: rgba(15,23,42,0.35);
      }

      [data-testid="stStatusWidget"] {
        border-radius: 0.8rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-shell">
      <div class="badge">AI KNOWLEDGE WORKSPACE</div>
      <h1>Knowledge Desk</h1>
      <p>Search across your documents and retrieve grounded answers with transparent source tracing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-header">🗂️ Knowledge Desk</div>', unsafe_allow_html=True)
    st.caption("Multi-document RAG · Hugging Face + FAISS")

    with st.expander("❓ Help & commands", expanded=not st.session_state.messages):
        st.text(render_help("Slash commands"))
        st.caption("Type a slash command in the chat box, or just ask a question.")

    st.divider()
    st.subheader("⚙️ Chunking & retrieval")

    chunk_size = st.slider(
        "Chunk size (characters)", 100, 2000, step=50,
        value=int(os.getenv("CHUNK_SIZE", "500")),
        help="Maximum characters per chunk before embedding.",
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)", 0, 500, step=10,
        value=int(os.getenv("CHUNK_OVERLAP", "50")),
        help="Characters repeated between neighbouring chunks.",
    )
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size // 5, 0)
        st.warning(f"Overlap must be smaller than chunk size — clamped to {chunk_overlap}.")

    top_k = st.slider(
        "Top-K chunks per query", 1, 12, value=int(os.getenv("TOP_K", "4"))
    )

    # Overlap preview: how the chosen settings tile a 1,000-char document.
    st.caption("Overlap preview (a 1,000-character document)")
    span, pos, tiles = 1000, 0, []
    step = max(chunk_size - chunk_overlap, 1)
    while pos < span:
        tiles.append((pos, min(pos + chunk_size, span)))
        pos += step
    seg_bar = "".join(
        f"<span style='display:inline-block;height:14px;width:{(b - a) / span * 100:.2f}%;"
        f"background:hsl({(i * 47) % 360},65%,62%);opacity:.9;"
        f"border-right:1px solid #fff'></span>"
        for i, (a, b) in enumerate(tiles)
    )
    st.markdown(
        f"<div style='display:flex;width:100%;border-radius:4px;"
        f"overflow:hidden'>{seg_bar}</div>"
        f"<div style='font-size:.75rem;color:gray'>{len(tiles)} chunks · "
        f"{chunk_size} chars · {chunk_overlap} overlap</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("📄 Documents")

    uploaded = st.file_uploader(
        "Upload documents (merged with the data/ folder)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "json", "md", "markdown", "html", "htm"],
        key="uploader",
    )
    uploads_key = "|".join(sorted(f.name for f in uploaded)) or "none"

    # --- Rebuild decision ------------------------------------------------
    # The index is rebuilt only when: it doesn't exist, the on-disk build
    # config differs from the sliders, the uploaded file set changed, or the
    # user explicitly asked for a rebuild. This is what kills the cache
    # errors: no cached function touches st.session_state any more.
    disk_cfg = read_build_config(os.path.getmtime(BUILD_CONFIG_PATH)) if os.path.exists(BUILD_CONFIG_PATH) else {}
    config_matches = (
        disk_cfg.get("chunk_size") == chunk_size
        and disk_cfg.get("chunk_overlap") == chunk_overlap
        and os.path.exists(os.path.join(VECTOR_DIR, "index.faiss"))
    )
    uploads_changed = uploads_key != st.session_state.uploads_key
    needs_build = (
        st.session_state.force_rebuild
        or not config_matches
        or uploads_changed
    )

    if needs_build:
        with st.spinner("Building the vector store…"):
            try:
                from ingest_service import run_ingestion

                stats = run_ingestion(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    uploaded_files=list(uploaded),
                    log=lambda msg: None,
                )
                os.makedirs(VECTOR_DIR, exist_ok=True)
                with open(BUILD_CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(
                        {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                         "updated_at": datetime.now().isoformat(timespec="seconds")},
                        f,
                    )
                st.session_state.uploads_key = uploads_key
                st.session_state.force_rebuild = False
                st.session_state.ingest_error = None
                read_build_config.clear()
                st.toast(f"Index ready — {stats['chunks']} chunks", icon="✅")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                # Do not kill the whole app: keep the tabs usable and let the
                # user retry, instead of stopping the script mid-render.
                st.session_state.force_rebuild = False
                st.session_state.ingest_error = str(exc)

    # --- Per-document search scope ---------------------------------------
    stats = commands.index_stats(VECTOR_DIR)
    selected_sources = None
    if stats:
        all_sources = list(stats["sources"].keys())
        selected_sources = st.multiselect(
            "Search scope (per-document selector)",
            options=all_sources,
            default=all_sources,
            help="Restrict retrieval to the chosen documents.",
        )
        if not selected_sources:
            selected_sources = None  # nothing selected -> search everything

        st.caption(
            f"{stats['total_chunks']} chunks · {len(stats['sources'])} documents"
            + (f" · dim {stats['dimension']}" if stats["dimension"] else "")
        )
        with st.expander("Per-document chunk counts"):
            st.dataframe(
                pd.DataFrame(
                    sorted(stats["sources"].items()),
                    columns=["Document", "Chunks"],
                ),
                hide_index=True,
                width="stretch",
            )

    if st.button("♻️ Re-ingest now", width="stretch"):
        st.session_state.force_rebuild = True
        st.rerun()

    st.divider()
    if st.button("🗑️ Reset conversation", width="stretch"):
        reset_chat()
        st.rerun()

# ---------------------------------------------------------------------------
# RAGAS evaluation runner
# ---------------------------------------------------------------------------


def normalize_ragas_df(df):
    """Support both modern and legacy RAGAS column names."""
    if df is None:
        return df
    normalized = df.copy()
    rename_map = {
        "user_input": "question",
        "retrieved_contexts": "contexts",
        "response": "answer",
        "reference": "ground_truth",
    }
    normalized = normalized.rename(columns={k: v for k, v in rename_map.items() if k in normalized.columns})
    return normalized


def run_ragas_eval() -> str:
    """Execute the RAGAS evaluation, stash the DataFrame, append history."""
    try:
        from evaluation.ragas_evaluation import require_hf_token, run_ragas
        from evaluation.evaluation_utils import load_eval_dataset, save_results_csv

        hf_token = require_hf_token()
    except EnvironmentError as exc:
        return f"⚠️ {exc}"
    except Exception as exc:
        return f"⚠️ Could not prepare evaluation: {exc}"

    with st.spinner("Running RAGAS evaluation (judge LLM + embeddings)…"):
        try:
            dataset = load_eval_dataset(EVAL_DATA_PATH)
            df = run_ragas(dataset, hf_token, pipeline=get_pipeline())
            save_results_csv(df.to_dict(orient="records"), RESULTS_PATH)

            metric_cols = [c for c in df.columns if c in
                           ("faithfulness", "answer_relevancy",
                            "context_precision", "context_recall")]
            if metric_cols:
                avg = df[metric_cols].mean(numeric_only=True)
                history_row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                               "questions": len(df), **{k: round(v, 4) for k, v in avg.items()}}
                os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
                try:
                    hist = pd.read_csv(HISTORY_PATH)
                    if hist.empty or "timestamp" not in hist.columns:
                        hist = pd.DataFrame(columns=["timestamp"] + metric_cols)
                    for col in metric_cols:
                        if col not in hist.columns:
                            hist[col] = pd.NA
                    hist = pd.concat([hist, pd.DataFrame([history_row])], ignore_index=True)
                except Exception:
                    hist = pd.DataFrame([history_row])
                hist = hist[[c for c in ["timestamp", "questions"] + metric_cols if c in hist.columns]].copy()
                hist.to_csv(HISTORY_PATH, index=False)

            st.session_state.ragas_df = normalize_ragas_df(df)
        except Exception as exc:
            return f"⚠️ RAGAS evaluation failed: {exc}"

    st.toast("RAGAS evaluation complete", icon="📊")
    return "Evaluation finished — see the **📊 Evaluation** tab for scores and charts."


# ---------------------------------------------------------------------------
# Command handling (shared command layer)
# ---------------------------------------------------------------------------


def run_command(name: str, arg: str) -> str:
    """Handle a slash command; returns the text to show the user."""
    if name in ("/help", "/commands"):
        return render_help("Available commands")

    if name == "/ingest":
        st.session_state.force_rebuild = True
        st.rerun()

    if name == "/list":
        stats = commands.index_stats(VECTOR_DIR)
        if not stats:
            return "No index found yet. Upload documents or press **Re-ingest now**."
        lines = [f"{stats['total_chunks']} chunks across "
                 f"{len(stats['sources'])} document(s):", ""]
        for src, count in stats["sources"].items():
            lines.append(f"  • {src} — {count} chunk(s)")
        return "\n".join(lines)

    if name == "/stats":
        stats = commands.index_stats(VECTOR_DIR)
        if not stats:
            return "No index found yet."
        return (
            f"**Index stats**\n\n"
            f"- Total chunks: {stats['total_chunks']}\n"
            f"- Documents: {len(stats['sources'])}\n"
            f"- Embedding dimension: {stats['dimension'] or '?'}\n"
            f"- Chunk size / overlap: {chunk_size} / {chunk_overlap}\n"
            f"- Top-K per query: {top_k}\n\n"
            f"Documents: {', '.join(stats['sources'])}"
        )

    if name == "/sources":
        if not st.session_state.last_sources:
            return "No previous answer to show sources for yet."
        lines = ["**Sources behind the last answer:**", ""]
        for i, src in enumerate(st.session_state.last_sources, 1):
            lines.append(f"  {i}. {src}")
        return "\n".join(lines)

    if name == "/eval":
        return run_ragas_eval()

    if name == "/reset":
        reset_chat()
        st.rerun()

    if name in ("/exit", "/quit"):
        return "This is the web app — just close the tab to leave. 👋"

    return commands.unknown_command_msg(name)


# ---------------------------------------------------------------------------
# Q&A flow
# ---------------------------------------------------------------------------


def render_contexts(contexts):
    """Expandable view of the retrieved chunks with similarity scores."""
    if not contexts:
        return
    with st.expander(f"🔎 Retrieved chunks ({len(contexts)})"):
        for i, ctx in enumerate(contexts, 1):
            score = ctx.get("score")
            src = os.path.basename(str(ctx.get("source")))
            page = ctx.get("page")
            label = f"{i}. {src}" + (f" · page {page}" if page else "")
            if score is not None:
                label += f" · score {score:.3f}"
            st.markdown(f"**{label}**")
            st.text(ctx.get("text", "")[:600])
            st.divider()


def answer_question(question: str):
    """Full Q&A round-trip: retrieve, generate, render, store in history."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                result = get_pipeline().answer(
                    question, top_k=top_k, source_filter=selected_sources
                )
            except Exception as exc:
                result = {"answer": f"⚠️ Generation failed: {exc}",
                          "sources": [], "contexts": []}
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))
        render_contexts(result["contexts"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"],
         "sources": result["sources"], "contexts": result["contexts"]}
    )
    st.session_state.last_sources = result["sources"]
    st.session_state.last_contexts = result["contexts"]
    st.rerun()


# ---------------------------------------------------------------------------
# Main area — tabs: Chat / Evaluation
# ---------------------------------------------------------------------------

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

with tab_chat:
    if st.session_state.get("ingest_error"):
        st.error(
            "⚠️ Last ingestion did not complete: "
            f"{st.session_state['ingest_error']} — upload documents or refresh "
            "chunk settings to retry."
        )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))
                render_contexts(msg.get("contexts"))

    # Sample questions (from the evaluation set) shown on an empty chat.
    sample_questions = []
    try:
        with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
            sample_questions = [item["question"] for item in json.load(f)][:3]
    except Exception:
        pass
    if not st.session_state.messages and sample_questions:
        st.info("👋 Ask anything about your documents — or start with a sample:")
        cols = st.columns(len(sample_questions))
        for col, q in zip(cols, sample_questions):
            if col.button(q, key=f"sample_{q}", width="stretch"):
                answer_question(q)

    question = st.chat_input("Ask about your documents… (type / for commands)")

    if question:
        if is_command(question):
            name, arg = parse_command(question)
            reply = run_command(name, arg)
            if name not in ("/ingest", "/reset"):  # those rerun the app
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()
        else:
            answer_question(question)

    # Export the conversation as Markdown.
    if st.session_state.messages:
        md_lines = ["# Knowledge Desk — conversation export",
                    f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
        for msg in st.session_state.messages:
            who = "🧑 **Question**" if msg["role"] == "user" else "🤖 **Answer**"
            md_lines += [f"### {who}", msg["content"], ""]
            if msg.get("sources"):
                md_lines += ["Sources: " + ", ".join(msg["sources"]), ""]
        st.download_button(
            "⬇️ Export chat as Markdown",
            data="\n".join(md_lines).encode("utf-8"),
            file_name="rag_conversation.md",
            mime="text/markdown",
            width="stretch",
        )

with tab_eval:
    st.subheader("📊 RAGAS evaluation")
    st.markdown(
        "Scores the pipeline on `evaluation/evaluation_data.json` with four metrics: "
        "**Faithfulness**, **Answer Relevancy**, **Context Precision**, **Context Recall**. "
        "The judge LLM runs on the Hugging Face Inference API (chat endpoint), so "
        "`HF_TOKEN` must be set in `.env`. Ground-truth answers are required for "
        "context recall."
    )

    if st.button("▶️ Run RAGAS evaluation", type="primary"):
        msg = run_command("/eval", "")
        if msg:
            st.markdown(msg)  # show errors / completion inside the eval tab

    df = normalize_ragas_df(st.session_state.ragas_df)
    if df is not None:
        metric_cols = [c for c in df.columns if c in
                       ("faithfulness", "answer_relevancy",
                        "context_precision", "context_recall")]
        avg = df[metric_cols].mean(numeric_only=True)

        if metric_cols:
            cols = st.columns(len(metric_cols))
            for col, (mname, value) in zip(cols, avg.items()):
                col.metric(mname.replace("_", " ").title(), f"{value:.2f}")
        else:
            st.info("No metric columns in the latest result — the judge LLM returned "
                    "no scores (quota / rate limit?). Re-run the evaluation.")

        # DataFrames can't render raw lists (contexts) — show a count instead.
        display_df = df.copy()
        if "contexts" in display_df.columns:
            display_df["contexts"] = display_df["contexts"].apply(
                lambda x: len(x) if isinstance(x, list) else x
            )
            display_df = display_df.rename(columns={"contexts": "n_contexts"})
        if "question" not in display_df.columns and "user_input" in display_df.columns:
            display_df = display_df.rename(columns={"user_input": "question"})
        st.dataframe(display_df, width="stretch")

        try:
            import plotly.express as px

            question_col = "question" if "question" in df.columns else "user_input"
            chart_df = df[[question_col] + metric_cols].melt(
                id_vars=question_col, var_name="Metric", value_name="Score"
            )
            fig = px.bar(
                chart_df, x="Metric", y="Score", color="Metric",
                facet_col=question_col, range_y=[0, 1],
                title="Per-question RAGAS scores",
            )
            fig.update_xaxes(matches=None, showticklabels=False)
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.bar_chart(avg)

        # Trend across evaluation runs, when history exists.
        if os.path.exists(HISTORY_PATH):
            try:
                hist = pd.read_csv(HISTORY_PATH)
                valid_cols = [c for c in metric_cols if c in hist.columns]
                if valid_cols and "timestamp" in hist.columns and len(hist) > 0:
                    hist = hist[["timestamp"] + valid_cols].dropna(subset=["timestamp"]).copy()
                    if len(hist) > 0:
                        st.markdown("**Score history across runs**")
                        history_chart = px.line(
                            hist, x="timestamp", y=valid_cols, markers=True,
                            range_y=[0, 1], labels={"value": "score", "timestamp": "run"},
                        )
                        st.plotly_chart(history_chart, width="stretch")
            except Exception as exc:
                st.caption(f"History chart unavailable: {exc}")

        cdl1, cdl2 = st.columns(2)
        with cdl1:
            st.download_button(
                "⬇️ Download results CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="ragas_results.csv",
                mime="text/csv",
            )
        with cdl2:
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, "rb") as f:
                    st.download_button(
                        "⬇️ Download run history",
                        data=f.read(),
                        file_name="ragas_history.csv",
                        mime="text/csv",
                    )
    else:
        st.caption("No results yet — run the evaluation above, or type `/eval` in chat.")
