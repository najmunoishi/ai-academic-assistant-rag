# AI Academic Assistant — Retrieval-Augmented Generation (RAG)

## Overview
A small educational knowledge-retrieval project demonstrating document ingestion,
semantic embeddings, FAISS vector search, source IDs, and basic retrieval tests.

## Architecture
JSONL documents -> SentenceTransformer embeddings -> FAISS similarity search
-> retrieved evidence with source IDs.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example questions
- How do I protect student records?
- Who can help me plan my courses?
- Where can I check exam dates?

## Limitations
This prototype retrieves evidence; it does not generate answers with an LLM.
The sample documents are synthetic examples, not official institutional policies.
It does not yet implement fine-tuning, a knowledge graph, or PII redaction.

## Evaluation
Run the retrieval tests with:
```bash
python -m unittest discover -s tests -v
```
