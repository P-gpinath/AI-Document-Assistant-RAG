# AI-Powered Document Assistant using RAG

## 1. Project Overview

The AI-Powered Document Assistant is a Python-based application that allows users to upload PDF documents and ask questions about their content.

It uses Retrieval-Augmented Generation (RAG) to find relevant information from uploaded documents and generate answers using a locally running Large Language Model.

## 2. Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB
- Ollama
- Llama 3.2 (1B)

## 3. Features

- Upload PDF documents.
- Extract text from PDF files.
- Clean and split text into smaller chunks.
- Generate text embeddings.
- Store embeddings in ChromaDB.
- Perform semantic similarity search.
- Retrieve relevant document content.
- Generate answers using Llama 3.2.
- Ask multiple questions about the uploaded PDF.
- View and clear chat history.

## 4. How the Project Works

1. The user uploads a PDF document.
2. PyPDF extracts text from the document.
3. The extracted text is cleaned.
4. The text is divided into smaller chunks.
5. Sentence Transformers converts chunks into numerical embeddings.
6. ChromaDB stores the chunks and embeddings.
7. The user's question is converted into an embedding.
8. ChromaDB retrieves the most relevant chunks.
9. The retrieved content is passed to the Llama 3.2 model.
10. The model generates an answer based on the retrieved content.

## 5. Project Structure

```text
project_1/
│
├── data/
├── src/
│   ├── pdf_extractor.py
│   ├── text_processor.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
│
├── chroma_db/
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

## 6. Installation and Execution

### Prerequisites

- Python installed
- Ollama installed
- Llama 3.2 model downloaded

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the project folder

```bash
cd project_1
```

### Step 3: Create a virtual environment

```bash
py -m venv venv
```

### Step 4: Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
py -m pip install -r requirements.txt
```

### Step 6: Download the Llama model

```bash
ollama pull llama3.2:1b
```

### Step 7: Run the application

```bash
py -m streamlit run app.py
```

## 7. Project Limitations

- Supports one uploaded PDF at a time.
- Scanned PDFs without extractable text may not work.
- Chat history is maintained during the current application session.
- Answer quality depends on the retrieved document content and language model.

## 8. Author

**Pranay GopiNath**
