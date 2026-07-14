# Assignment 3 - LLM RAG Project

## Student Information
- **Name:** Tasneem Syed
- **Course:** Large Language Models, Prompting, and Agentic AI
- **Assignment:** Assignment 3 

## Project Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) system using Groq and LangChain. The application retrieves relevant information from uploaded PDF documents and generates accurate answers based on the retrieved content.

## Features

- Loads multiple PDF documents
- Splits documents into text chunks
- Generates embeddings using HuggingFace Sentence Transformers
- Stores embeddings in a FAISS vector database
- Retrieves relevant document chunks
- Uses the Groq LLM to generate context-aware answers
- Compares RAG-based responses with standard LLM responses

## Technologies Used

- Python
- Groq API
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- PyPDF

## Project Structure

```
.
├── data/
│   ├── computers 1.pdf
│   ├── computers 2.pdf
│   ├── computers 3.pdf
│   ├── computers 4.pdf
│   └── computers 5.pdf
├── main.py
├── requirements.txt
├── output.txt
└── README.md
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python main.py
```

## Output

The application loads the PDF documents, creates embeddings, retrieves relevant information, and generates answers using the Groq language model.
