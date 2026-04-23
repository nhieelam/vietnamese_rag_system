import html
import re
from typing import List, Union

import streamlit as st

from app.services.session_service import SessionService
from app.models.citation import Citation


def render_chat_header():
    st.title("Vietnamese RAG Assistant")
    st.markdown("Ask questions about your uploaded documents and get comprehensive answers")

    documents = SessionService.get_documents()
    if documents:
        st.success(f"Knowledge base active with {len(documents)} document(s)")
    else:
        st.warning("No documents uploaded. Please upload documents from the sidebar to begin.")

    st.divider()


def render_chat_messages():
    chat_container = st.container()

    with chat_container:
        messages = SessionService.get_messages_for_ui()

        if not messages:
            _render_welcome_message()
        else:
            _render_message_history(messages)

        if st.session_state.get("message_sent", False):
            st.session_state.message_sent = False
            st.rerun()


def _render_welcome_message():
    st.info(
        """
        I'm here to help you find information from your departmental documents.

        **How to start:**
        1. Upload your documents using the sidebar (←)
        2. Ask me any questions about the content
        3. I'll provide comprehensive answers based on the documents
        """
    )


def _render_message_history(messages):
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")

        if role == "user":
            _render_user_message(message)
            index += 1
            continue

        if role == "assistant":
            mode = message.get("mode")
            if not mode:
                mode, _ = _extract_mode_content(message)

            if mode in ("RAG", "Co-RAG") and index + 1 < len(messages):
                nxt = messages[index + 1]
                if nxt.get("role") == "assistant":
                    next_mode = nxt.get("mode") or _extract_mode_content(nxt)[0]
                    if next_mode and {mode, next_mode} == {"RAG", "Co-RAG"}:
                        if mode == "RAG":
                            _render_comparison_pair(message, nxt, index)
                        else:
                            _render_comparison_pair(nxt, message, index)
                        index += 2
                        continue

            _render_assistant_message(message, index)
            index += 1
            continue

        _render_assistant_message(message, index)
        index += 1


def _render_user_message(message):
    st.markdown(
        f"""<div class="message-container">
        <div style="text-align: right; margin-bottom: 4px;">
            <small style="color: #666;">You</small>
        </div>
        <div class="user-message">{html.escape(message.get("content", ""))}</div>
        <div style="text-align: right; margin-top: 2px;">
            <small style="color: #999;">{message.get("timestamp", "")}</small>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_assistant_message(message, msg_index: int):
    mode = message.get("mode", "Assistant")
    content = message.get("content", "")
    citations = _as_citation_list(message.get("citations", []))

    rendered_content = _render_inline_refs(content, len(citations))

    st.markdown(
        f"""<div class="message-container">
        <div style="text-align: left; margin-bottom: 4px;">
            <small style="color: #666;">{mode}</small>
        </div>
        <div class="assistant-message">{rendered_content}</div>
        <div style="text-align: left; margin-top: 2px;">
            <small style="color: #999;">{message.get("timestamp", "")}</small>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if citations:
        _render_citations(citations, f"{mode}-{msg_index}")


def _extract_mode_content(message):
    content = message.get("content", "")
    if not content:
        return None, ""
    if content.startswith("[RAG]\n"):
        return "RAG", content[len("[RAG]\n"):]
    if content.startswith("[Co-RAG]\n"):
        return "Co-RAG", content[len("[Co-RAG]\n"):]
    return None, content


def _as_citation_list(raw) -> List[Citation]:
    result: List[Citation] = []
    for item in raw or []:
        if isinstance(item, Citation):
            result.append(item)
        elif isinstance(item, dict):
            try:
                result.append(Citation.from_dict(item))
            except Exception:
                continue
    return result


def _render_inline_refs(content: str, n_citations: int) -> str:
    """Escape content and turn [n] into colored badges."""
    escaped = html.escape(content or "")

    def repl(match: re.Match) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= n_citations:
            return f'<span class="cite-badge">[{idx}]</span>'
        return match.group(0)

    return re.sub(r"\[(\d+)\]", repl, escaped)


def _render_citations(citations: List[Citation], key_prefix: str):
    st.markdown("**Sources**")

    for i, c in enumerate(citations, 1):
        title_parts = [f"[{c.ref_index or i}]", c.source_name or "Unknown"]
        if c.page_number is not None:
            title_parts.append(f"trang {c.page_number}")
        if c.relevance_score:
            title_parts.append(f"relevance {c.relevance_score:.0%}")
        title = " · ".join(title_parts)

        with st.expander(title, expanded=False):
            meta_bits = []
            if c.chunk_id is not None:
                meta_bits.append(f"chunk #{c.chunk_id}")
            if c.char_start is not None and c.char_end is not None:
                meta_bits.append(f"offset {c.char_start}–{c.char_end}")
            if meta_bits:
                st.caption(" · ".join(meta_bits))

            full = c.full_text or c.excerpt or ""
            st.markdown(
                f'<div class="cite-chunk">{html.escape(full)}</div>',
                unsafe_allow_html=True,
            )

            if c.document_id is not None:
                pdf_entry = SessionService.get_pdf(c.document_id)
                if pdf_entry and pdf_entry.get("bytes"):
                    label = "Tải PDF"
                    if c.page_number is not None:
                        label += f" (gợi ý mở trang {c.page_number})"
                    st.download_button(
                        label=label,
                        data=pdf_entry["bytes"],
                        file_name=pdf_entry.get("name", "document.pdf"),
                        mime="application/pdf",
                        key=f"pdf-{key_prefix}-{i}",
                    )


def _render_comparison_pair(rag_message, co_rag_message, msg_index: int):
    rag_citations = _as_citation_list(rag_message.get("citations", []))
    co_rag_citations = _as_citation_list(co_rag_message.get("citations", []))

    rag_content = _render_inline_refs(rag_message.get("content", ""), len(rag_citations))
    co_rag_content = _render_inline_refs(co_rag_message.get("content", ""), len(co_rag_citations))

    st.markdown("<div class='compare-title'>RAG vs Co-RAG</div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            f"""<div class="compare-card compare-rag">
            <div class="compare-label">RAG</div>
            <div class="assistant-message compare-message">{rag_content}</div>
            <div class="compare-time">{rag_message.get("timestamp", "")}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if rag_citations:
            _render_citations(rag_citations, f"rag-{msg_index}")

    with col_right:
        st.markdown(
            f"""<div class="compare-card compare-co-rag">
            <div class="compare-label">Co-RAG</div>
            <div class="assistant-message compare-message">{co_rag_content}</div>
            <div class="compare-time">{co_rag_message.get("timestamp", "")}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if co_rag_citations:
            _render_citations(co_rag_citations, f"corag-{msg_index}")
