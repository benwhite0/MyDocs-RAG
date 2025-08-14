import os

import requests
import streamlit as st

st.set_page_config(page_title="MyDocs-RAG", layout="wide")
st.title("MyDocs-RAG")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Ensure a flag exists to know when to submit
if "do_query" not in st.session_state:
    st.session_state["do_query"] = False


def _submit():
    st.session_state["do_query"] = True


# Sidebar: Upload and controls
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=False, key="uploaded_file"
    )
    if uploaded is not None:
        # Avoid re-uploading same file on each rerun
        already = st.session_state.get("_uploaded_name") == uploaded.name and st.session_state.get(
            "_uploaded_size"
        ) == getattr(uploaded, "size", None)
        if not already:
            with st.spinner("Uploading and indexing..."):
                try:
                    files = {
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                    }
                    r = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                    if r.status_code == 200:
                        st.success("File uploaded and index rebuilt.")
                        st.session_state["_uploaded_name"] = uploaded.name
                        st.session_state["_uploaded_size"] = getattr(uploaded, "size", None)
                    else:
                        st.error(f"Upload failed: {r.text}")
                except Exception as e:
                    st.error(f"Upload error: {e}")
    st.markdown("---")
    k = st.slider("Top-K", 1, 20, 5)

# Main layout: center a narrow column for chat
left, center, right = st.columns([1, 2, 1])
with center:
    q = st.text_input("Ask a question about your documents", key="question", on_change=_submit)
    ask = st.button("Ask")
    if ask:
        st.session_state["do_query"] = True

    # Execute when either Enter was pressed or the button clicked
    if st.session_state.get("do_query") and st.session_state.get("question"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={"question": st.session_state["question"], "k": k},
                    timeout=120,
                )
                data = r.json()
                st.subheader("Answer")
                # Scrollable, wrapped answer box
                st.markdown(
                    f"""
                    <div style='
                        max-width: 820px;
                        border: 1px solid #e6e6e6;
                        border-radius: 8px;
                        padding: 12px;
                        background: #fafafa;
                        overflow-y: auto;
                        max-height: 360px;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    '>
                    {data.get("answer", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Prompt & Sources"):
                    st.code(data.get("prompt", ""))
                    for s in data.get("sources", []):
                        st.write(f"{s.get('source','')}  (score: {round(s.get('score', 0.0), 3)})")
            except Exception as e:
                st.error(f"Query failed: {e}")
            finally:
                # Reset the submit flag after handling
                st.session_state["do_query"] = False
