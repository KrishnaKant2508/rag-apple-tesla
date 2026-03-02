# RAG Pipeline Design Report
**Assignment:** RAG + LLM for Financial Document Q&A (Apple 10-K 2024 & Tesla 10-K 2023)

---

## 1. Chunking Strategy

Documents were parsed page-by-page using PyMuPDF (`fitz`), with each page's text cleaned before chunking. Text cleaning removed checkbox characters, repeated headers/footers, and normalized whitespace to reduce noise in embeddings.

Chunks were created with a **fixed-size character window of 400 characters and an 80-character overlap**. This size was chosen deliberately:

- 400 characters (~60-80 words) is large enough to contain a complete financial fact (e.g., a revenue figure with its label and context) but small enough that the embedding remains semantically focused.
- The 80-character overlap ensures that facts split across chunk boundaries (e.g., a table cell and its row header) are still retrievable.
- Each chunk preserves metadata: `document` (Apple or Tesla 10-K), `section` (extracted via regex matching Item headers), and `page` number - enabling accurate source citation.

A larger chunk size (e.g., 1000+ characters) would dilute embeddings with irrelevant context; a smaller size risks breaking financial sentences mid-fact.

---

## 2. Embedding & Retrieval

**Embedding model:** `sentence-transformers/all-mpnet-base-v2` (run on CPU)

This model was chosen for its strong performance on semantic similarity benchmarks and its moderate size (~420MB), making it practical for a Colab T4 environment without consuming GPU VRAM needed for the LLM.

Embeddings are stored in a **FAISS IndexFlatIP** (inner product / cosine similarity after L2 normalization). FAISS was preferred over Chroma for its speed and zero-dependency vector search on CPU.

**Retrieval pipeline:**
1. Query is embedded using the same model.
2. Top-15 nearest chunks are retrieved via FAISS.
3. A keyword filter detects whether the query mentions "Apple" or "Tesla" and restricts results to the relevant document, reducing cross-document noise.
4. A **CrossEncoder reranker** (`BAAI/bge-reranker-large`, on CPU) re-scores the top-15 chunks against the query and selects the top-5 for the LLM prompt.

The reranker was added because bi-encoder retrieval (FAISS) optimizes for approximate similarity, while a cross-encoder reads query and chunk jointly significantly improving precision for specific financial figures.

---

## 3. LLM Choice

**Model:** `mistralai/Mistral-7B-Instruct-v0.2`, loaded in **4-bit quantization** via BitsAndBytes (`nf4`, double quant, fp16 compute).

Mistral-7B was chosen because:
- It fits in ~4GB GPU VRAM after 4-bit quantization, leaving headroom on a Colab T4.
- Its instruction-tuned variant follows structured prompts reliably, critical for enforcing citation and refusal rules.
- It outperforms comparably-sized models (e.g., Phi-3-mini) on financial reasoning tasks.

GPT-4, Claude, and other closed APIs were explicitly excluded per assignment requirements.

---

## 4. Out-of-Scope Question Handling

The system prompt enforces three strict response rules:

1. **Answer is in context** >> Give a direct, factual answer with the exact figure.
2. **Topic is in scope but detail is missing** >> Respond: `"Not specified in the document."`
3. **Question is out of scope** (forecasts, current roles, physical descriptions, opinions) >> Respond: `"This question cannot be answered based on the provided documents."`

Questions 11 (Tesla stock forecast), 12 (Apple CFO in 2025), and 13 (Tesla HQ color) are handled by rule 3 - the LLM is instructed never to use outside knowledge, and these topics have no corresponding content in either 10-K filing.

When a refusal is triggered, `sources` is returned as an empty list `[]`.

---

## 5. Source Citation Format

Sources are cited as a flat list per the assignment specification:

```json
["Apple 10-K", "Item 8", "p. 28"]
```

The top retrieved chunk's metadata (document, section, page) is used for citation when a non-refusal answer is generated.
