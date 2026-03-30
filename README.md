# CrossDocumentScalableAgent

A PDF semantic search pipeline that ingests documents, builds hybrid search indexes, and answers complex questions across multiple PDFs with citations.

Built with LangGraph (parallel fan-out/fan-in), OpenSearch (BM25 + kNN hybrid search), and any OpenAI-compatible LLM/embedding API.

## How it works

**Ingestion:** PDFs are extracted, chunked with overlap, summarized by LLM, and embedded into two OpenSearch indexes (doc-level for coarse filtering, chunk-level for fine retrieval).

**Query:** Complex questions are decomposed into sub-queries, run through two-stage hybrid retrieval, then each relevant PDF gets a parallel LLM extraction worker. Results are merged, checked for gaps (with an optional retrieval loop), and synthesized into a final cross-document answer with `[file, page]` citations.

See `pipeline_ingest.txt` and `pipeline_query.txt` for detailed architecture diagrams.

## Setup

```bash
# Install dependencies
pip install langgraph openai opensearch-py pymupdf python-dotenv

# Configure environment
cp .env.example .env
# Edit .env with your LLM, embedding, and OpenSearch endpoints
```

## Usage

```bash
# Index a folder of PDFs
python pdf_search_pipeline.py index --pdf-dir ./pdfs

# Query across indexed PDFs
python pdf_search_pipeline.py query --question "Compare authentication approaches across documents"
```

## Testing

```bash
# Unit tests only (no external services)
pytest test_pdf_pipeline.py -m "not integration and not pipeline and not benchmark"

# Integration tests (requires live LLM, embedding, OpenSearch)
pytest test_pdf_pipeline.py -m integration

# End-to-end pipeline tests
pytest test_pdf_pipeline.py -m pipeline

# Benchmark evaluation (22 questions across 10 PDFs)
pytest test_pdf_pipeline.py -m benchmark

# Everything
pytest test_pdf_pipeline.py
```

## Benchmark Results

22 questions (10 single-document, 12 cross-document) evaluated against 10 PDFs:

| Metric | Score |
|--------|-------|
| Source Match Accuracy | 22/22 (100%) |
| Keyword Coverage | 114/150 (76%) |
| Cross-Doc Multi-Source Rate | 11/12 (91.7%) |

Open `benchmark_results.html` for the full interactive dashboard.

## Tech Stack

- **Orchestration:** LangGraph (Command/Send fan-out, Annotated fan-in)
- **Search:** OpenSearch 3.x (hybrid BM25 + kNN with HNSW/faiss)
- **LLM/Embedding:** Any OpenAI-compatible API
- **PDF Parsing:** PyMuPDF
