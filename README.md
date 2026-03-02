# RAG System for Apple (2024) & Tesla (2023) 10-K Filings

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline
that answers financial and legal questions using:

-   Apple Inc. -- 2024 Form 10-K\
-   Tesla, Inc. -- 2023 Form 10-K

The system retrieves relevant document chunks using semantic search +
reranking and generates grounded answers using an open-access LLM.

It strictly follows assignment requirements: - Open-source embeddings\
- Vector database\
- Re-ranking\
- Open-access LLM (no GPT-4 / Claude)\
- Metadata preservation\
- Structured JSON output\
- Strict refusal handling

------------------------------------------------------------------------

## Live Runnable Notebook

Colab (fully end-to-end):

https://colab.research.google.com/drive/1R6T6imPBqjui-zWCEXMc2Y3zrxhjaDGn#scrollTo=k0b116_7SGYQ

The notebook: - Clones the GitHub repo\
- Installs dependencies\
- Parses both PDFs\
- Builds FAISS index\
- Loads Mistral-7B (4-bit)\
- Runs inference on all 13 evaluation questions\
- Outputs required JSON format

------------------------------------------------------------------------

## GitHub Repository

Public Repo:

https://github.com/KrishnaKant2508/rag-apple-tesla.git

Repository structure:

rag-apple-tesla/ │ ├── RAG_system_for_Apple_Tesla.ipynb ├──
requirements.txt ├── design_report.md └── README.md

------------------------------------------------------------------------

# System Architecture

## 1️⃣ Document Ingestion & Chunking

-   PDFs parsed using PyMuPDF (fitz)
-   Cleaning steps:
    -   Remove checkbox characters (☐ ☒)
    -   Remove repeated headers/footers
    -   Normalize whitespace

### Chunking Strategy

-   Chunk size: 400 characters\
-   Overlap: 80 characters

Why 400?

-   Large enough to capture complete financial facts
-   Small enough to keep embeddings semantically focused
-   Prevents dilution of meaning in large filings

Each chunk preserves metadata:

{ "text": "...", "document": "Apple 10-K", "section": "Item 8", "page":
282 }

------------------------------------------------------------------------

## 2️⃣ Embedding & Retrieval Pipeline

### Embedding Model

sentence-transformers/all-mpnet-base-v2 (CPU)

### Vector Store

FAISS IndexFlatIP (cosine similarity after L2 normalization)

Retrieval Steps: 1. Query embedding 2. Top-15 chunk retrieval 3.
Apple/Tesla document filter 4. Cross-encoder reranking
(BAAI/bge-reranker-large) 5. Top-5 chunks passed to LLM

------------------------------------------------------------------------

## 3️⃣ LLM Choice

Model: mistralai/Mistral-7B-Instruct-v0.2

Loaded using: - 4-bit quantization (nf4) - BitsAndBytes - fp16 compute

Reason: - Fits in Colab T4 GPU - Strong instruction following - Reliable
structured output

------------------------------------------------------------------------

## 4️⃣ Strict Answering Rules

The system enforces:

Case A -- Answer found in context\
→ Return direct factual answer with citation

Case B -- Topic exists but detail missing\
→ "Not specified in the document."

Case C -- Out-of-scope question\
→ "This question cannot be answered based on the provided documents."

Refusal returns empty sources list.

------------------------------------------------------------------------

## Required Interface

def answer_question(query: str) -\> dict: ''' Returns: { "answer":
"...", "sources": \["Apple 10-K", "Item 8", "p. 282"\] } '''

------------------------------------------------------------------------

## Example Output

{ "question_id": 1, "answer": "\$391,036 million", "sources": \["Apple
10-K", "Item 8", "p. 282"\] }

------------------------------------------------------------------------

## Compliance Checklist

-   Parse both 10-K filings ✅
-   Preserve document/section/page metadata ✅
-   Open-source embeddings ✅
-   FAISS vector database ✅
-   Re-ranking step ✅
-   Open-access LLM only ✅
-   Strict refusal handling ✅
-   Required JSON output format ✅
-   Public runnable Colab notebook ✅
-   Design report included ✅

------------------------------------------------------------------------

## Run Locally

git clone https://github.com/KrishnaKant2508/rag-apple-tesla.git cd
rag-apple-tesla pip install -r requirements.txt

Then open:

RAG_system_for_Apple_Tesla.ipynb
