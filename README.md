# Domain-Specific RAG Chatbot for PDF Question Answering

## 1. Project Overview

This project is a domain-specific Retrieval-Augmented Generation (RAG) chatbot that answers questions using information from uploaded PDF documents.

The system extracts text from PDF files, divides the text into smaller chunks, converts the chunks into embeddings, and stores them in a FAISS vector database. When a user asks a question, the system retrieves the most relevant document chunks and provides them as context to a Large Language Model (LLM).

The chatbot is designed to answer only from the uploaded documents and avoid generating unsupported information.

---

## 2. Objectives

- Upload one or more PDF documents.
- Extract text from the uploaded PDFs.
- Preserve document name and page number.
- Split extracted text into smaller chunks.
- Generate embeddings using Sentence Transformers.
- Store embeddings in a FAISS vector database.
- Retrieve relevant document chunks for a user question.
- Generate answers using an LLM.
- Display the source document and page number.
- Refuse to answer when the required information is not available in the uploaded documents.
- Provide a simple Streamlit-based user interface.

---

## 3. Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- LangChain
- Groq LLM
- python-dotenv

### Embedding Model

`all-MiniLM-L6-v2`

### Vector Database

FAISS

### LLM

`openai/gpt-oss-20b` through Groq

---

## 4. Project Structure

```text
domain_rag_chatbot/
│
├── documents/
│   └── PDF documents
│
├── tests/
│   └── test_questions.txt
│
├── vector_store/
│   ├── faiss.index
│   └── metadata.json
│
├── venv/
│
├── app.py
├── document_loader.py
├── prompt.py
├── rag_pipeline.py
├── vector_store.py
├── .env
├── .gitignore
└── README.md