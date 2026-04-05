import streamlit as st
from ingestion import extract_chunks, build_vectorstore
from retrieval import retrieve_with_sources
from generator import generate_answer

st.set_page_config(page_title="DocuQuery v2", page_icon="📄", layout="wide")
st.title("📄 DocuQuery v2 — Multi-Document RAG")
st.caption("Upload multiple PDFs and ask questions across all of them.")

# Sidebar for uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Reading and indexing your documents..."):
                chunks = extract_chunks(uploaded_files)
                collection, embedding_fn = build_vectorstore(chunks)
                st.session_state["collection"] = collection
                st.session_state["embedding_fn"] = embedding_fn
                st.session_state["num_chunks"] = len(chunks)
                st.session_state["num_docs"] = len(uploaded_files)
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")

    if "num_chunks" in st.session_state:
        st.info(f"📚 {st.session_state['num_docs']} doc(s) | {st.session_state['num_chunks']} chunks loaded")

# Main chat area
if "collection" not in st.session_state:
    st.info("Upload PDFs from the sidebar and click 'Process Documents' to get started.")
else:
    query = st.text_input("Ask a question about your documents", placeholder="e.g. What is a foreign key?")

    if query:
        with st.spinner("Searching and generating answer..."):
            retrieved = retrieve_with_sources(
                query,
                st.session_state["collection"],
                st.session_state["embedding_fn"]
            )
            answer = generate_answer(query, retrieved)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        for i, r in enumerate(retrieved[:3]):
            with st.expander(f"Source {i+1} — {r['source']}, Page {r['page']} (relevance: {r['score']})"):
                st.write(r["text"])