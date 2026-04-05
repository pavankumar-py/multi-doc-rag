import numpy as np


def retrieve_with_sources(query, index, model, texts, metadatas, k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        chunks.append({
            "text": texts[idx],
            "source": metadatas[idx]["source"],
            "page": metadatas[idx]["page"],
            "score": round(1 / (1 + dist), 3)
        })

    return chunks
