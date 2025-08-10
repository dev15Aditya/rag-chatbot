# Mistral RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that uses Mistral AI's embedding and chat models with FAISS vector search to answer questions based on your PDF documents.

## Features

- **Document Processing**: Automatically loads and processes PDF documents from a specified directory
- **Vector Search**: Uses FAISS (Facebook AI Similarity Search) for efficient similarity search
- **Mistral Integration**: Leverages Mistral's embedding model for document vectorization and chat model for responses
- **Web Interface**: Clean Gradio-based UI for easy interaction
- **Persistent Index**: Saves and loads vector indices for quick startup

### Dependencies

```bash
pip install numpy faiss-cpu gradio mistralai langchain-community python-dotenv pathlib
```

### API Key

You'll need a Mistral AI API key. Get one from [Mistral AI Console](https://console.mistral.ai/).

## Setup

### 1. Environment Configuration

Create a `.env` file in your project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 2. Document Directory

Create a `docs` directory and place your PDF files there:

```
project/
├── docs/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── .env
└── chat.py
```

### 3. Install Dependencies

```bash
pip install numpy faiss-cpu gradio mistralai langchain-community python-dotenv
```

## Usage

### Running the Application
```bash
cd rag-chatbot
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
python main.py
```

The application will:
1. Check for existing vector index
2. If no index exists, automatically process all PDFs in `./docs` and create one
3. Launch a Gradio web interface (usually at `http://localhost:7860`)

### Using the Interface

1. **Ask Questions**: Type your question in the text box and click "Ask"
2. **Rebuild Index**: Click "Rebuild index" after adding new documents to the `docs` folder

## Configuration

You can modify these settings at the top of the script:

```python
DOCS_DIR = "./docs"              # Directory containing PDF files
EMBEDDING_MODEL = "mistral-embed" # Mistral embedding model
CHAT_MODEL = "mistral-large-latest" # Mistral chat model
CHUNK_SIZE = 2048                # Size of text chunks
CHUNK_OVERLAP = 200              # Overlap between chunks
TOP_K = 3                        # Number of relevant chunks to retrieve
```

## How It Works

### 1. Document Processing
- Loads PDF files from the `docs` directory
- Splits documents into overlapping chunks (default: 2048 characters with 200 character overlap)
- This chunking helps maintain context while keeping pieces manageable for embedding

### 2. Vector Embedding
- Each chunk is converted to a vector embedding using Mistral's `mistral-embed` model
- Embeddings capture semantic meaning of the text in high-dimensional space
- Vectors are stored in a FAISS index for efficient similarity search

### 3. Query Processing
- User questions are embedded using the same model
- FAISS searches for the most similar document chunks (default: top 3)
- Retrieved chunks provide context for answering the question

### 4. Response Generation
- Retrieved chunks and the user question are formatted into a prompt
- Mistral's chat model (`mistral-large-latest`) generates a response based on the context
- The model is instructed to answer only based on the provided context

## Troubleshooting

### Common Issues

1. **"Missing MISTRAL_API_KEY"**: Ensure your `.env` file contains a valid Mistral API key

2. **"Index not found"**: Click "Rebuild index" or restart the application to process documents

3. **No documents found**: Check that PDF files are in the `./docs` directory

4. **Memory issues**: Reduce `CHUNK_SIZE` or process fewer documents at once

### Performance Tips

- For large document collections, consider processing in batches
- Adjust `TOP_K` based on your needs (more chunks = more context but slower processing)
- Use `faiss-gpu` instead of `faiss-cpu` for better performance with large indices

## API Models Used

- **Embedding**: `mistral-embed` - Converts text to vector representations
- **Chat**: `mistral-large-latest` - Generates responses based on context

## License

This project uses the Mistral AI API. Please review Mistral's terms of service for usage guidelines.# rag-chatbot
