import json
import os
from pathlib import Path

import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from groq import Groq

# =====================================================
# Paths
# =====================================================
BASE = Path(__file__).parent
ART_DIR = BASE / "artifacts"

CHUNKS_PATH = ART_DIR / "chunks.json"
INDEX_PATH = ART_DIR / "faiss.index"
META_PATH = ART_DIR / "meta.json"

# =====================================================
# Load artifacts
# =====================================================
def load_artifacts():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_PATH))
    embedder = SentenceTransformer(meta["model"])
    return chunks, index, embedder

CHUNKS, INDEX, EMBEDDER = load_artifacts()

# =====================================================
# Groq client (LLM)
# =====================================================
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =====================================================
# RAG function (RETRIEVE → SUMMARIZE)
# =====================================================
def rag_answer(query, top_k=5):
    if not query.strip():
        return "", ""

    # ---- Embed query ----
    q_emb = EMBEDDER.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    # ---- Retrieve ----
    scores, ids = INDEX.search(q_emb, top_k)

    retrieved_chunks = []
    for idx in ids[0]:
        if idx < 0:
            continue
        c = CHUNKS[int(idx)]
        retrieved_chunks.append(
            f"[SOURCE]\n{c['source']}\n\n{c['text']}"
        )

    if not retrieved_chunks:
        return "Not found in the provided documents.", ""

    context = "\n\n---\n\n".join(retrieved_chunks)

    # ---- PROMPT (CORRECT + ROBUST) ----
    prompt = f"""
You are a regulatory and technical expert.

Use ONLY the context below.

If the context contains any information relevant to the question,
summarize it clearly and precisely.

If the context does NOT mention the topic at all,
say exactly:
"Not found in the provided documents."

Context:
{context}

Question:
{query}
"""

    # ---- LLM CALL (NON-DEPRECATED MODEL) ----
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = completion.choices[0].message.content.strip()

    return answer, context

# =====================================================
# Gradio UI
# =====================================================
with gr.Blocks(title="RAG Demo ") as demo:
    gr.Markdown(
        "# RAG Demo\n"
        "Ask a question about the ingested documents."
    )

    query = gr.Textbox(
        label="Question",
        placeholder="Example: What does the document say about ...?"
    )

    topk = gr.Slider(
        1, 10, value=5, step=1, label="Top K"
    )

    btn = gr.Button("Retrieve")

    answer = gr.Textbox(
        label="Answer",
        lines=6
    )

    context = gr.Textbox(
        label="Retrieved Context",
        lines=20
    )

    btn.click(
        fn=rag_answer,
        inputs=[query, topk],
        outputs=[answer, context]
    )

# =====================================================
# Launch
# =====================================================
if __name__ == "__main__":
    print("Starting Gradio app...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,  # avoids port conflicts
        debug=True
    )
