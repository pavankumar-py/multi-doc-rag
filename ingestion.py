import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


def extract_chunks(uploaded_files):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    for file in uploaded_files:
        doc = fitz.open(stream=file.read(), filetype="pdf")

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()

            if not text:
                continue

            chunks = splitter.split_text(text)

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": file.name,
                        "page": page_num + 1
                    }
                })

        doc.close()

    return all_chunks


def build_vectorstore(chunks):
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, model, texts, metadatas
