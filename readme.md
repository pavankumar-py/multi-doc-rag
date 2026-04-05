# 📄 DocuQuery v2 — Multi-Document RAG with Source Attribution

A production-ready Retrieval-Augmented Generation (RAG) application that lets you upload multiple PDF documents and ask questions across all of them — with every answer citing the exact source document and page number it came from.

**[🚀 Live Demo](https://multi-doc-rag.streamlit.app/)**

---

## ✨ Features

- **Multi-document support** — Upload multiple PDFs simultaneously
- **Source attribution** — Every answer cites the exact document and page number
- **Relevance scoring** — See how relevant each source chunk is to your question
- **Fast semantic search** — FAISS vector search for accurate retrieval
- **Local embeddings** — No external embedding API needed (runs on-device)
- **Clean UI** — Simple Streamlit interface, no setup required for end users

---

## 🏗️ Architecture

```
PDFs → Text Extraction (PyMuPDF) → Chunking (LangChain) → Embeddings (sentence-transformers)
                                                                        ↓
User Question → Embed Query → FAISS Similarity Search → Top-K Chunks + Metadata
                                                                        ↓
                                              LLM (Groq / Llama 3.1) → Answer with Citations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| Text Chunking | LangChain Text Splitters |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| LLM | Groq API (Llama 3.1 8B Instant) |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```
multi-doc-rag/
├── app.py          # Streamlit UI
├── ingestion.py    # PDF parsing, chunking, vectorstore
├── retrieval.py    # Semantic search with source metadata
├── generator.py    # LLM answer generation with citations
├── requirements.txt
└── .gitignore
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/pavankumarkatchi/multi-doc-rag.git
cd multi-doc-rag
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

---

## 💡 How It Works

1. **Upload** one or more PDF files via the sidebar
2. Click **Process Documents** — the app extracts text page by page, chunks it, and builds a FAISS vector index
3. **Ask any question** in the text box
4. The app embeds your question, finds the most relevant chunks across all documents, and sends them to the LLM
5. You get a **cited answer** showing exactly which document and page each fact came from

---

## 🔮 Upcoming Features

- [ ] Evaluation dashboard with RAGAS metrics (faithfulness, relevancy, precision)
- [ ] Chat history / multi-turn conversations
- [ ] Support for .txt and .docx files
- [ ] Conflict detection across documents

---

## 👨‍💻 Author

**Pavan Kumar Katchi**
- GitHub: [@pavankumarkatchi](https://github.com/pavankumark-py)
- LinkedIn: [linkedin.com/in/pavankumarkatchi](https://linkedin.com/in/pavankumarkatchi)

---

## 📄 License

MIT License — free to use and modify.
