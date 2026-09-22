"""Streamlit chat UI with multi-turn memory, citations, feedback, and an admin tab."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isearch.agent import Assistant   # noqa: E402
from isearch.config import Settings   # noqa: E402

st.set_page_config(page_title="Institutional AI Search", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading models and building the index...")
def get_assistant() -> Assistant:
    return Assistant(Settings.from_env())


assistant = get_assistant()

with st.sidebar:
    st.header("Institutional AI Search")
    role = st.selectbox("Viewing as", ["public", "student", "staff"], index=1)
    category = st.selectbox("Category", ["(all)"] + assistant.index.categories if assistant.index else ["(all)"])
    category = None if category == "(all)" else category
    use_llm_choice = st.radio("Answer style", ["Auto", "LLM", "Extractive only"], index=0)
    use_llm = {"Auto": None, "LLM": True, "Extractive only": False}[use_llm_choice]
    st.caption(f"Ollama: {'available' if assistant.llm.available() else 'not running'} ({assistant.s.ollama_model})")
    if st.button("Reload documents"):
        assistant.reload()
        st.cache_resource.clear()
        st.rerun()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

tab_chat, tab_docs, tab_admin = st.tabs(["Chat", "Documents", "Admin"])

with tab_chat:
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"Sources ({m['mode']})"):
                    for e in m["sources"]:
                        page = f", p.{e['page']}" if e["page"] else ""
                        flag = " ⚠️ superseded" if e["superseded"] else ""
                        st.markdown(f"**[{e['n']}] {e['source']}{page}** — {e['section'] or 'n/a'}{flag}")
                        st.caption(e["text"])
            if m.get("query_id") is not None:
                c1, c2 = st.columns([1, 12])
                if c1.button("👍", key=f"up{m['query_id']}"):
                    assistant.store.add_feedback(m["query_id"], 1)
                    st.toast("Thanks for the feedback.")
                if c2.button("👎", key=f"down{m['query_id']}"):
                    assistant.store.add_feedback(m["query_id"], -1)
                    st.toast("Thanks — this will show up in the admin tab.")

    if question := st.chat_input("Ask about admissions, academics, hostel, research, policy..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"q": m["content"], "a": st.session_state.messages[i + 1]["content"]}
                   for i, m in enumerate(st.session_state.messages[:-1])
                   if m["role"] == "user" and i + 1 < len(st.session_state.messages) - 1]
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                r = assistant.ask(question, history=history, role=role, category=category, use_llm=use_llm)
            st.markdown(r.answer)
            if r.evidence:
                with st.expander(f"Sources ({r.mode})"):
                    for e in r.evidence:
                        page = f", p.{e['page']}" if e["page"] else ""
                        flag = " ⚠️ superseded" if e["superseded"] else ""
                        st.markdown(f"**[{e['n']}] {e['source']}{page}** — {e['section'] or 'n/a'}{flag}")
                        st.caption(e["text"])
        st.session_state.messages.append({"role": "assistant", "content": r.answer, "sources": r.evidence,
                                          "mode": r.mode, "query_id": r.query_id})

with tab_docs:
    st.subheader("Add a document")
    up_col, cat_col = st.columns([3, 1])
    upload = up_col.file_uploader("PDF, TXT, MD or HTML", type=["pdf", "txt", "md", "html", "htm"])
    upload_category = cat_col.text_input("Category", value="general")
    if upload and st.button("Add to index"):
        try:
            info = assistant.save_document(upload.name, upload.getvalue(), upload_category.strip() or "general")
            st.success(f"Indexed {info['n_chunks']} chunks from {info['path']}")
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("Indexed documents")
    for d in assistant.documents():
        state = "⚠️ skipped: " + d["note"] if d["status"] == "skipped" else ("superseded" if d["superseded"] else "active")
        st.write(f"**{d['path']}** — {d['category']} / {d['audience']} — {d['n_chunks']} chunks — {state}")

with tab_admin:
    st.subheader("Usage")
    st.json(assistant.store.stats())
    st.subheader("Top unanswered questions (documentation gaps)")
    st.table(assistant.store.unanswered(30))
    st.subheader("Recently downvoted answers")
    st.table(assistant.store.downvoted(20))
