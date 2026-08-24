from __future__ import annotations

import hashlib
import re
from pathlib import Path
import sys

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.app.document_service import (
    DocumentService,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Engineering Intelligence",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    visibility: visible !important;
    background: transparent !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"] button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

section[data-testid="stSidebar"] {
    padding-top: 1rem;
}

.block-container {
    max-width: 1240px;
    padding-top: 1.5rem !important;
    padding-bottom: 6rem !important;
}

.main-eyebrow {
    text-align: center;
    color: #5f6b7a;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.main-title {
    text-align: center;
    font-size: 2.7rem;
    line-height: 1.1;
    font-weight: 800;
    margin-bottom: 0.55rem;
}

.main-subtitle {
    text-align: center;
    font-size: 1.05rem;
    line-height: 1.55;
    color: #667085;
    max-width: 900px;
    margin: 0 auto 1.5rem auto;
}

.hero-badges {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.8rem;
}

.hero-badge {
    border: 1px solid #d9dee7;
    background: #f8fafc;
    border-radius: 999px;
    padding: 0.35rem 0.75rem;
    color: #475467;
    font-size: 0.82rem;
    font-weight: 600;
}

.capability-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin: 0 0 1.25rem 0;
}

.capability-card {
    border: 1px solid #e4e7ec;
    border-radius: 14px;
    padding: 1rem;
    background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
    min-height: 118px;
}

.capability-icon {
    font-size: 1.25rem;
    margin-bottom: 0.35rem;
}

.capability-title {
    font-weight: 750;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}

.capability-text {
    color: #667085;
    font-size: 0.8rem;
    line-height: 1.45;
}

.info-card {
    border: 1px solid #e4e7ec;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    background: #ffffff;
}

.section-label {
    color: #667085;
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 650;
}

[data-testid="stChatInput"] {
    z-index: 1000;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
}

[data-testid="stMarkdownContainer"] {
    line-height: 1.65;
}

.katex-display {
    margin: 1rem 0 !important;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.2rem 0;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .main-title {
        font-size: 2rem;
    }

    .capability-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 600px) {
    .capability-grid {
        grid-template-columns: 1fr;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# ENGINEERING MARKDOWN / LATEX RENDERING
# ============================================================

def render_engineering_markdown(text: str) -> str:
    """Normalize common LLM math delimiters for Streamlit Markdown.

    Streamlit renders textbook-style LaTeX reliably when inline math uses
    $...$ and display math uses $$...$$. Models sometimes return the
    equivalent LaTeX delimiters \\( ... \\) and \\[ ... \\]. This helper
    converts those forms while deliberately leaving fenced code blocks alone.
    """

    text = str(text or "")
    parts = re.split(r"(```[\s\S]*?```)", text)

    for i in range(0, len(parts), 2):
        block = parts[i]

        # Display mathematics: \[ ... \] -> $$ ... $$
        block = re.sub(
            r"\\\[\s*([\s\S]*?)\s*\\\]",
            lambda m: "$$\n" + m.group(1).strip() + "\n$$",
            block,
        )

        # Inline mathematics: \( ... \) -> $ ... $
        block = re.sub(
            r"\\\([\s\S]*?\\\)",
            lambda m: "$" + m.group(0)[2:-2].strip() + "$",
            block,
        )

        # Equation environments -> display math.
        block = re.sub(
            r"\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}",
            lambda m: "$$\n" + m.group(1).strip() + "\n$$",
            block,
        )

        parts[i] = block

    return "".join(parts)


def show_answer(text: str) -> None:
    """Render an assistant answer with Markdown + textbook-style LaTeX."""
    st.markdown(
        render_engineering_markdown(text),
        unsafe_allow_html=False,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "service" not in st.session_state:
    st.session_state.service = DocumentService(
        text_model="phi3:mini",
        advanced_text_model="qwen3:8b",
        reasoning_model="deepseek-r1:7b",
        gemma3_model="gemma3:4b",
        llama_model="llama3.1:8b",
        mistral_model="mistral:7b",
        gemma2_model="gemma2:9b",
        vision_model="qwen2.5vl:7b",
        advanced_vision_model="qwen3-vl:8b",
        embedding_model="nomic-embed-text",
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "active_image_paths" not in st.session_state:
    st.session_state.active_image_paths = []

if "active_image_names" not in st.session_state:
    st.session_state.active_image_names = []

if "documents_analyzed" not in st.session_state:
    st.session_state.documents_analyzed = False


service: DocumentService = (
    st.session_state.service
)


# ============================================================
# PERSISTENT UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

def save_uploaded_file(
    uploaded_file,
) -> Path:

    original_name = Path(
        uploaded_file.name
    ).name

    suffix = (
        Path(
            original_name
        ).suffix.lower()
    )

    content = uploaded_file.getvalue()

    digest = hashlib.sha256(
        content
    ).hexdigest()[:16]

    safe_name = (
        f"{Path(original_name).stem}"
        f"_{digest}"
        f"{suffix}"
    )

    destination = (
        UPLOAD_DIR
        / safe_name
    )

    destination.write_bytes(
        content
    )

    return destination


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="main-eyebrow">LOCAL-FIRST · MULTIMODAL · ENGINEERING AI</div>
<div class="main-title">Engineering Intelligence</div>
<div class="main-subtitle">
Your AI assistant for engineering, science and mathematics — analyze manuals,
drawings, schematics, figures, formulas and technical data, then ask questions
in natural language.
</div>
<div class="hero-badges">
    <span class="hero-badge">📄 Documents & RAG</span>
    <span class="hero-badge">👁️ Vision & Schematics</span>
    <span class="hero-badge">∑ Engineering Calculations</span>
    <span class="hero-badge">🌐 Current Information</span>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 📐 Engineering Intelligence"
    )

    st.markdown(
        "### Upload"
    )

    st.caption(
        "Upload engineering PDFs or images. Analyze text, "
        "tables, drawings, schematics, labels and figures."
    )

    uploaded_files = st.file_uploader(
        "Upload engineering documents or images",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        accept_multiple_files=True,
        key="engineering_upload",
        help=(
            "PDFs are indexed for text and visual "
            "document retrieval. Images are passed "
            "directly to the vision-language model."
        ),
    )

    st.divider()

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} file(s) selected**"
        )

        for uploaded_file in uploaded_files:

            suffix = (
                Path(
                    uploaded_file.name
                ).suffix.lower()
            )

            if suffix == ".pdf":
                icon = "📄"
            else:
                icon = "🖼️"

            st.caption(
                f"{icon} {uploaded_file.name}"
            )

    else:

        st.caption(
            "No files selected."
        )

    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    analyze = st.button(
        "🔍 Analyze Files",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    # --------------------------------------------------------
    # Clear chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        service.clear_conversation()

        st.rerun()

    # --------------------------------------------------------
    # Clear documents
    # --------------------------------------------------------

    if st.button(
        "🧹 Clear Indexed Documents",
        use_container_width=True,
    ):

        service.clear_documents()

        st.session_state.analysis_results = []

        st.session_state.documents_analyzed = False

        st.rerun()

    # --------------------------------------------------------
    # Clear active images
    # --------------------------------------------------------

    if st.button(
        "🖼️ Clear Active Images",
        use_container_width=True,
    ):

        st.session_state.active_image_paths = []

        st.session_state.active_image_names = []

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    st.markdown(
        "### AI stack"
    )

    info = service.model_info

    st.caption("💬 NLP / LLM pool")
    for model_name in info.get("text_models", []):
        st.caption(f"• {model_name}")

    st.caption(
        f"🧠 Reasoning: {info['reasoning_model']}"
    )

    st.caption(
        f"👁️ Vision: "
        f"{info['vision_model']}"
    )

    st.caption(
        f"🔬 Advanced vision: "
        f"{info['advanced_vision_model']}"
    )

    st.caption(
        f"🔎 Embeddings: "
        f"{info['embedding_model']}"
    )

    st.divider()

    # --------------------------------------------------------
    # Session status
    # --------------------------------------------------------

    st.markdown(
        "### Workspace"
    )

    st.caption(
        f"Documents: "
        f"{service.document_count}"
    )

    st.caption(
        f"Chunks: "
        f"{service.chunk_count}"
    )

    st.caption(
        f"Active images: "
        f"{len(st.session_state.active_image_paths)}"
    )


# ============================================================
# PROCESS UPLOADS
# ============================================================

if analyze:

    st.session_state.uploaded_files = list(
        uploaded_files
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    results = []

    image_paths = []

    image_names = []

    total = len(
        uploaded_files
    )

    try:

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            status.info(
                f"Processing {uploaded_file.name}..."
            )

            progress.progress(
                int(
                    index
                    / total
                    * 100
                )
            )

            suffix = (
                Path(
                    uploaded_file.name
                ).suffix.lower()
            )

            # ------------------------------------------------
            # Persist file.
            #
            # IMPORTANT:
            # Do NOT delete this file after processing.
            # The vision model may need it later.
            # ------------------------------------------------

            saved_path = save_uploaded_file(
                uploaded_file
            )

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            if suffix == ".pdf":

                result = (
                    service.index_document(
                        saved_path,
                        use_multimodal=True,
                    )
                )

                results.append(
                    result
                )

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            else:

                image_paths.append(
                    str(
                        saved_path.resolve()
                    )
                )

                image_names.append(
                    uploaded_file.name
                )

        # ----------------------------------------------------
        # Store active images.
        # ----------------------------------------------------

        st.session_state.active_image_paths = (
            image_paths
        )

        st.session_state.active_image_names = (
            image_names
        )

        # ----------------------------------------------------
        # Finish
        # ----------------------------------------------------

        progress.progress(
            100
        )

        status.success(
            "Files analyzed successfully."
        )

        st.session_state.analysis_results = (
            results
        )

        st.session_state.documents_analyzed = (
            bool(results)
        )

    except Exception as exc:

        status.error(
            "File analysis failed."
        )

        st.exception(
            exc
        )


# ============================================================
# ACTIVE IMAGES
# ============================================================

active_image_paths = (
    st.session_state.active_image_paths
)

if active_image_paths:

    with st.expander(
        "🖼️ Active engineering images",
        expanded=True,
    ):

        st.info(
            "These images are available to the AI assistant for visual analysis."
        )

        for index, image_path in enumerate(
            active_image_paths
        ):

            image_name = (
                st.session_state.active_image_names[index]
                if index
                < len(
                    st.session_state.active_image_names
                )
                else Path(
                    image_path
                ).name
            )

            st.caption(
                f"Image {index + 1}: {image_name}"
            )

            try:

                st.image(
                    image_path,
                    caption=image_name,
                    width="stretch",
                )

            except Exception:
                st.warning(
                    f"Could not preview {image_name}."
                )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if (
    st.session_state.analysis_results
):

    with st.expander(
        "📄 Indexed document information",
        expanded=True,
    ):

        for result in (
            st.session_state.analysis_results
        ):

            st.markdown(
                f"**{result.get('file_name', 'Document')}**"
            )

            st.write(
                f"Pages: "
                f"{result.get('page_count', '?')}"
            )

            st.write(
                f"Chunks: "
                f"{result.get('chunk_count', '?')}"
            )

            st.write(
                f"Indexed: "
                f"{result.get('indexed_count', '?')}"
            )

            if result.get(
                "visual_context_chunks"
            ) is not None:

                st.write(
                    "Visual-context chunks: "
                    f"{result.get('visual_context_chunks')}"
                )


# ============================================================
# WELCOME / CAPABILITIES
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
<div class="capability-grid">
    <div class="capability-card">
        <div class="capability-icon">📄</div>
        <div class="capability-title">Engineering Documents</div>
        <div class="capability-text">Search manuals, specifications, tables, procedures and technical notes with document-aware retrieval.</div>
    </div>
    <div class="capability-card">
        <div class="capability-icon">👁️</div>
        <div class="capability-title">Drawings & Schematics</div>
        <div class="capability-text">Analyze components, labels, connections, figures, diagrams and other engineering visuals.</div>
    </div>
    <div class="capability-card">
        <div class="capability-icon">∑</div>
        <div class="capability-title">Math & Engineering</div>
        <div class="capability-text">Work through formulas, equations, units, calculations and technical explanations with formatted notation.</div>
    </div>
    <div class="capability-card">
        <div class="capability-icon">🌐</div>
        <div class="capability-title">Current Information</div>
        <div class="capability-text">Use web research when a question depends on current, recent or changing information.</div>
    </div>
</div>

<div class="info-card">
<div class="section-label">Start a conversation</div>

**Upload a file when your question depends on a document or image — or simply ask an engineering, science or mathematics question.**

Try:

- Explain the schematic and trace the conventional current path.
- Calculate the current through a $10\\,\\Omega$ resistor connected to $24\\,\\mathrm{V}$.
- Derive the relationship between power, voltage and current.
- Explain the function of each component in Figure 4.
- Solve the motor problem and show every formula, substitution and unit.
- Write and balance the chemical equation for the reaction shown.
- What is the latest version of Python?
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        show_answer(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
        ):

            model = message.get(
                "model"
            )

            mode = message.get(
                "mode"
            )

            if model:

                if mode:

                    st.caption(
                        f"Model: {model} · Mode: {mode}"
                    )

                else:

                    st.caption(
                        f"Model: {model}"
                    )

            sources = message.get(
                "sources",
                [],
            )

            if sources:

                with st.expander(
                    "📚 Retrieved document context"
                ):

                    seen = set()

                    for source in sources:

                        file_name = source.get(
                            "file_name",
                            "Document",
                        )

                        page = source.get(
                            "page_number"
                        )

                        key = (
                            file_name,
                            page,
                        )

                        if key in seen:
                            continue

                        seen.add(
                            key
                        )

                        if page is not None:

                            st.caption(
                                f"📄 {file_name} "
                                f"— Page {page}"
                            )

                        else:

                            st.caption(
                                f"📄 {file_name}"
                            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask an engineering, science, math or document question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            question
        )

    # --------------------------------------------------------
    # IMPORTANT ROUTING RULE
    # --------------------------------------------------------
    # Do NOT block general questions when no document/image
    # has been uploaded.
    #
    # DocumentService.ask() is the single source of truth for
    # deciding whether the question needs:
    #   - deterministic calculation
    #   - normal text generation
    #   - advanced text generation
    #   - document retrieval / RAG
    #   - visual analysis
    #
    # This keeps the Streamlit UI independent from routing
    # logic and allows questions such as:
    #   "What is Ohm's law?"
    #   "Explain Kirchhoff's laws."
    #   "What is 9 minus 2?"
    # without requiring an uploaded file.
    # --------------------------------------------------------

    active_image_paths = list(
        st.session_state.active_image_paths
    )

    with st.chat_message(
        "assistant"
    ):

        placeholder = st.empty()

        placeholder.markdown(
            "Thinking..."
        )

        try:

            response = service.ask(
                question=question,
                top_k=5,
                image_paths=active_image_paths,
            )

            answer = str(
                response.answer or ""
            ).strip()

            metadata = (
                response.metadata
                or {}
            )

            # ------------------------------------------------
            # Model / mode
            # ------------------------------------------------

            model = (
                metadata.get(
                    "selected_model"
                )
                or metadata.get(
                    "model"
                )
            )

            mode = metadata.get(
                "mode"
            )

            # ------------------------------------------------
            # Display answer
            # ------------------------------------------------

            placeholder.markdown(
                render_engineering_markdown(answer),
                unsafe_allow_html=False,
            )

            # ------------------------------------------------
            # Display selected model
            # ------------------------------------------------

            if model:

                if mode:

                    st.caption(
                        f"Model: {model} · Mode: {mode}"
                    )

                else:

                    st.caption(
                        f"Model: {model}"
                    )

            # ------------------------------------------------
            # Retrieved document sources
            # ------------------------------------------------

            sources = (
                response.sources
                or []
            )

            if sources:

                with st.expander(
                    "📚 Retrieved document context"
                ):

                    seen = set()

                    for source in sources:

                        if isinstance(
                            source,
                            dict,
                        ):

                            file_name = source.get(
                                "file_name",
                                "Document",
                            )

                            page = source.get(
                                "page_number"
                            )

                        else:

                            source_metadata = getattr(
                                source,
                                "metadata",
                                {},
                            ) or {}

                            file_name = (
                                source_metadata.get(
                                    "file_name"
                                )
                                or source_metadata.get(
                                    "document_name"
                                )
                                or "Document"
                            )

                            page = (
                                getattr(
                                    source,
                                    "page_number",
                                    None,
                                )
                                or source_metadata.get(
                                    "page_number"
                                )
                            )

                        key = (
                            file_name,
                            page,
                        )

                        if key in seen:
                            continue

                        seen.add(
                            key
                        )

                        if page is not None:

                            st.caption(
                                f"📄 {file_name} "
                                f"— Page {page}"
                            )

                        else:

                            st.caption(
                                f"📄 {file_name}"
                            )

            # ------------------------------------------------
            # Visual information
            # ------------------------------------------------

            visual_count = metadata.get(
                "visual_image_count",
                0,
            )

            if visual_count:

                st.caption(
                    "🖼️ Visual images analyzed: "
                    f"{visual_count}"
                )

            # ------------------------------------------------
            # Save assistant message
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "model": model,
                    "mode": mode,
                    "sources": sources,
                }
            )

        except Exception as exc:

            error = (
                "I couldn't complete that request.\n\n"
                f"Error: `{exc}`"
            )

            placeholder.error(
                error
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error,
                }
            )