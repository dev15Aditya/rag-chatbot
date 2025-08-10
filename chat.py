import os
import json
from pathlib import Path
from typing import List

import numpy as np
import faiss
import gradio as gr

from mistralai import Mistral
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# -------------------------
# Config — change if needed
# -------------------------

try:
    load_dotenv()
    if not os.environ.get("MISTRAL_API_KEY"):
        raise ValueError("Missing MISTRAL_API_KEY in .env")
except Exception as e:
    print(f"Error loading .env: {e}")
    exit(1)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
DOCS_DIR = "./docs"
INDEX_FILE = "faiss.index"
METADATA_FILE = "faiss_metadata.json"
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-large-latest"
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200
TOP_K = 3     
# -------------------------

if not MISTRAL_API_KEY:
    raise RuntimeError("Please set environment variable MISTRAL_API_KEY")

client = Mistral(MISTRAL_API_KEY)

def load_and_split_documents(docs_dir: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Load PDFs from docs_dir and split into text chunks"""
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    # Convert the LangChain Document objects to plain text strings and keep order
    return [chunk.page_content for chunk in chunks]

def get_mistral_embedding(text: str) -> List[float]:
    """Get embedding for a single text using Mistral embeddings API"""
    # Using the client.embeddings.create(...) response shape per Mistral docs
    resp = client.embeddings.create(model=EMBEDDING_MODEL, inputs=text)
    return resp.data[0].embedding

def build_faiss_index(chunks: List[str], index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
    """Create FAISS index from chunks and save index + metadata"""
    if len(chunks) == 0:
        raise ValueError("No chunks provided to build index.")
    print(f"[build] creating embeddings for {len(chunks)} chunks (this may take a while)...")
    # Create embeddings (batch sequentially to control memory & rate)
    embeddings = []
    for i, c in enumerate(chunks):
        e = get_mistral_embedding(c)
        embeddings.append(e)
        if (i + 1) % 50 == 0:
            print(f"[build] embeddings created for {i+1}/{len(chunks)} chunks")

    embeddings_np = np.array(embeddings).astype("float32")
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)

    faiss.write_index(index, index_path)
    # Save metadata (the chunk texts); indexing by integer position
    metadata = {"chunks": chunks}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print(f"[build] saved faiss index to {index_path} and metadata to {metadata_path}")

def load_faiss_index(index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
    """Load FAISS index and metadata (chunks list)"""
    if not Path(index_path).exists() or not Path(metadata_path).exists():
        raise FileNotFoundError("Index or metadata not found; run reindex first.")
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    chunks = metadata["chunks"]
    return index, chunks

def retrieve_top_k_chunks(question: str, index, chunks: List[str], k: int = TOP_K):
    """Embed the question, search FAISS and return the top k chunk texts"""
    q_emb = np.array([get_mistral_embedding(question)]).astype("float32")
    D, I = index.search(q_emb, k)  # D: distances, I: indices
    indices = I.tolist()[0]
    retrieved = [chunks[i] for i in indices if i < len(chunks)]
    return retrieved

def format_prompt_from_chunks(chunks: List[str], question: str) -> str:
    """Create prompt with retrieved context and user question (simple template from doc)"""
    joined = "\n\n---\n\n".join(chunks)
    prompt = f"""
Context information is below.
---------------------
{joined}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""
    return prompt

def answer_question_with_rag(question: str):
    """High-level function used by the Gradio UI to answer queries"""
    # Load index & chunks (lazy)
    try:
        index, chunks = load_faiss_index()
    except FileNotFoundError:
        return "Index not found. Please click 'Rebuild index' to create vector store from documents."

    retrieved = retrieve_top_k_chunks(question, index, chunks, k=TOP_K)
    prompt = format_prompt_from_chunks(retrieved, question)

    # call Mistral chat completion (using the prompt)
    messages = [{"role": "user", "content": prompt}]
    chat_resp = client.chat.complete(model=CHAT_MODEL, messages=messages, temperature=0.0)
    # Grab the model output text
    text = chat_resp.choices[0].message.content
    return text

# -------------------------
# Gradio UI
# -------------------------
def rebuild_index_and_notify():
    """Action for Rebuild button in UI — (re)builds the faiss index from DOCS_DIR"""
    chunks = load_and_split_documents(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    build_faiss_index(chunks)
    return "Rebuilt index from documents. Number of chunks: " + str(len(chunks))

def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# Mistral RAG Chatbot (FAISS + mistral-embed)")
        with gr.Row():
            question_input = gr.Textbox(lines=2, placeholder="Ask about your documents...", label="Question")
            rebuild_btn = gr.Button("Rebuild index (from ./docs)")
        answer_output = gr.Textbox(label="Answer", interactive=False)
        rebuild_status = gr.Textbox(label="Index status", value="Ready", interactive=False)

        def on_query(q):
            if not q or q.strip() == "":
                return "Please type a question."
            return answer_question_with_rag(q)

        submit_btn = gr.Button("Ask")
        submit_btn.click(on_query, question_input, answer_output)
        rebuild_btn.click(rebuild_index_and_notify, outputs=rebuild_status)

    demo.launch(share=False)

if __name__ == "__main__":
    # If index doesn't exist, build on startup (optional)
    if not Path(INDEX_FILE).exists() or not Path(METADATA_FILE).exists():
        print("[startup] No index found, building one from docs...")
        chunks = load_and_split_documents(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
        build_faiss_index(chunks)
    launch_gradio()
