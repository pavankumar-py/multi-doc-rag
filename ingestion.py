import fitz #PyMuPDF
import chromadb
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
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    client = chromadb.Client()

    try:
        client.delete_collection("multi_doc_rag")
    except:
        pass

    collection = client.create_collection(
        name="multi_doc_rag",
        embedding_function=embedding_fn
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return collection, embedding_fn 
