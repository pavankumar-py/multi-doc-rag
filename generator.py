from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

def generate_answer(query, chunks):
    context = ""
    for i, c in enumerate(chunks):
        context += f"[Source {i+1}: {c['source']}, Page {c['page']}]\n{c['text']}\n\n"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
Cite sources like (Source 1) or (Source 2) where relevant.
Be concise. Do not repeat the same fact multiple times.
If the answer is not in the context, say "Not found in the documents."

Context:
{context}

Question: {query}

Answer:"""

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    response = llm.invoke(prompt)
    return response.content