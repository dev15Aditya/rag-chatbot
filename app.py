from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import gradio as gr
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
# from dotenv import load_dotenv

# try:
#     load_dotenv()
#     if not os.getenv("HF_TOKEN"):
#         raise ValueError("Missing HF_TOKEN in .env")
# except Exception as e:
#     print(f"Error loading .env: {e}")
#     exit(1)

def load_docs(folder_path="docs"):
    loaders = [PyPDFLoader, CSVLoader]
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
    return documents

def init_rag():
    documents = load_docs()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}  # or "cuda" if you have GPU
    )
    vector_db = FAISS.from_documents(chunks, embeddings)

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())
    return qa_chain

def run_gui(qa):
    print("QA-->", qa)
    def respond(question):
        return qa.invoke({"query": question})["result"]
    gr.Interface(fn=respond, inputs="text", outputs="text").launch()

if __name__ == "__main__":
    qa_system = init_rag()
    run_gui(qa_system)
