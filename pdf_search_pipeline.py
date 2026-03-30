"""
PDF Semantic Search & Cross-PDF Insight Pipeline
=================================================
Tech: LangGraph (Command/Send fan-out), OpenSearch (hybrid vector+BM25), OpenAI-compatible API

Three phases:
  1. INDEXING  — ingest PDFs, chunk, embed, store in OpenSearch (two indexes)
  2. QUERY    — decompose question, hybrid retrieval, parallel per-PDF extraction
  3. SYNTHESIS — cross-PDF merge, gap-fill loop, final answer with citations

Usage:
  # Index a folder of PDFs
  python pdf_search_pipeline.py index --pdf-dir ./pdfs

  # Query
  python pdf_search_pipeline.py query --question "Compare Q1 vs Q3 expenses"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import operator
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pdf_pipeline")


@dataclass
class Config:
    # LLM API (OpenAI-compatible)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "sk-placeholder")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")

    # Embedding API (separate OpenAI-compatible endpoint)
    embed_base_url: str = os.getenv("EMBED_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    embed_api_key: str = os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY", "sk-placeholder"))
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1024"))

    # OpenSearch
    opensearch_host: str = os.getenv("OPENSEARCH_HOST", "localhost")
    opensearch_port: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    opensearch_user: str = os.getenv("OPENSEARCH_USER", "admin")
    opensearch_password: str = os.getenv("OPENSEARCH_PASSWORD", "admin")
    opensearch_use_ssl: bool = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
    doc_index_name: str = "pdf_doc_index"
    chunk_index_name: str = "pdf_chunk_index"

    # Chunking
    chunk_size: int = 500  # tokens (approx chars / 4)
    chunk_overlap: int = 50
    chars_per_token: int = 4  # rough approximation

    # Retrieval
    doc_top_k: int = 20
    chunk_top_k: int = 50
    rerank_top_k: int = 15
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    context_expand_pages: int = 2

    # Synthesis
    max_gap_fill_loops: int = 2
    max_context_tokens: int = 80_000

    # Parallelism
    ingest_batch_size: int = 50
    embed_batch_size: int = 64


def _load_dotenv():
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass

_load_dotenv()
CFG = Config()

# ---------------------------------------------------------------------------
# Clients (lazy singletons)
# ---------------------------------------------------------------------------


def _get_llm_client():
    from openai import OpenAI

    return OpenAI(base_url=CFG.llm_base_url, api_key=CFG.llm_api_key)


def _get_embed_client():
    from openai import OpenAI

    return OpenAI(base_url=CFG.embed_base_url, api_key=CFG.embed_api_key)


def _get_opensearch_client():
    from opensearchpy import OpenSearch

    return OpenSearch(
        hosts=[{"host": CFG.opensearch_host, "port": CFG.opensearch_port}],
        http_auth=(CFG.opensearch_user, CFG.opensearch_password),
        use_ssl=CFG.opensearch_use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
    )


# ---------------------------------------------------------------------------
# Helpers: LLM + Embedding
# ---------------------------------------------------------------------------


def llm_call(prompt: str, system: str = "You are a helpful assistant.", temperature: float = 0.0) -> str:
    client = _get_llm_client()
    resp = client.chat.completions.create(
        model=CFG.llm_model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts via OpenAI-compatible embedding endpoint."""
    client = _get_embed_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), CFG.embed_batch_size):
        batch = texts[i : i + CFG.embed_batch_size]
        resp = client.embeddings.create(model=CFG.embed_model, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
    return all_embeddings


# ---------------------------------------------------------------------------
# Helpers: PDF parsing + chunking
# ---------------------------------------------------------------------------


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text per page. Returns list of {page: int, text: str}."""
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError:
        import fitz

    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages: list[dict], source_file: str) -> list[dict]:
    """Split page texts into overlapping chunks with metadata."""
    full_text_parts: list[tuple[int, str]] = []  # (page_num, text)
    for p in pages:
        full_text_parts.append((p["page"], p["text"]))

    chunk_char_size = CFG.chunk_size * CFG.chars_per_token
    overlap_chars = CFG.chunk_overlap * CFG.chars_per_token

    # Build a flat text with page boundary tracking
    flat_text = ""
    page_boundaries: list[tuple[int, int, int]] = []  # (start_char, end_char, page_num)
    for page_num, text in full_text_parts:
        start = len(flat_text)
        flat_text += text + "\n"
        page_boundaries.append((start, len(flat_text), page_num))

    chunks = []
    pos = 0
    chunk_id = 0
    while pos < len(flat_text):
        end = min(pos + chunk_char_size, len(flat_text))
        chunk_text = flat_text[pos:end].strip()
        if not chunk_text:
            break

        # Determine which pages this chunk spans
        chunk_pages_set = set()
        for bstart, bend, pnum in page_boundaries:
            if pos < bend and end > bstart:
                chunk_pages_set.add(pnum)

        chunks.append(
            {
                "chunk_id": f"{hashlib.md5(source_file.encode()).hexdigest()[:8]}_{chunk_id}",
                "source_file": source_file,
                "pages": sorted(chunk_pages_set),
                "text": chunk_text,
            }
        )
        chunk_id += 1
        new_pos = end - overlap_chars
        if new_pos <= pos:
            break  # prevent infinite loop when remainder <= overlap
        pos = new_pos
        if pos >= len(flat_text):
            break

    return chunks


def summarize_document(pages: list[dict]) -> str:
    """Generate a 2-3 sentence summary of a document from its first few pages."""
    sample_text = "\n".join(p["text"][:2000] for p in pages[:5])[:8000]
    return llm_call(
        f"Summarize this document in 2-3 sentences. Focus on the main topic, type of document, and key entities:\n\n{sample_text}",
        system="You are a document summarizer. Be concise and factual.",
    )


# ---------------------------------------------------------------------------
# Helpers: OpenSearch index management
# ---------------------------------------------------------------------------


def ensure_indexes():
    """Create the doc-level and chunk-level OpenSearch indexes if they don't exist."""
    client = _get_opensearch_client()

    doc_index_body = {
        "settings": {
            "index": {"knn": True, "number_of_shards": 2, "number_of_replicas": 0},
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
        },
        "mappings": {
            "properties": {
                "source_file": {"type": "keyword"},
                "filename": {"type": "text", "analyzer": "standard"},
                "summary": {"type": "text", "analyzer": "standard"},
                "total_pages": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": CFG.embed_dim,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss"},
                },
            }
        },
    }

    chunk_index_body = {
        "settings": {
            "index": {"knn": True, "number_of_shards": 4, "number_of_replicas": 0},
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "pages": {"type": "integer"},
                "text": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": CFG.embed_dim,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss"},
                },
            }
        },
    }

    for idx_name, idx_body in [
        (CFG.doc_index_name, doc_index_body),
        (CFG.chunk_index_name, chunk_index_body),
    ]:
        if not client.indices.exists(index=idx_name):
            client.indices.create(index=idx_name, body=idx_body)
            log.info(f"Created index: {idx_name}")
        else:
            log.info(f"Index already exists: {idx_name}")


def bulk_index_docs(docs: list[dict], index_name: str):
    """Bulk index documents into OpenSearch."""
    from opensearchpy.helpers import bulk

    client = _get_opensearch_client()
    actions = []
    for doc in docs:
        doc_id = doc.get("chunk_id") or hashlib.md5(doc["source_file"].encode()).hexdigest()
        actions.append({"_index": index_name, "_id": doc_id, "_source": doc})

    if actions:
        success, errors = bulk(client, actions, raise_on_error=False)
        log.info(f"Indexed {success} docs into {index_name}, errors: {len(errors)}")


# ---------------------------------------------------------------------------
# Helpers: OpenSearch hybrid search
# ---------------------------------------------------------------------------


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    index_name: str,
    top_k: int,
    text_field: str = "text",
    filter_source_files: list[str] | None = None,
) -> list[dict]:
    """
    Hybrid search: BM25 + kNN vector, combined via weighted score normalization.
    OpenSearch doesn't natively combine in one query, so we run both and merge.
    """
    client = _get_opensearch_client()

    must_filter = []
    if filter_source_files:
        must_filter.append({"terms": {"source_file": filter_source_files}})

    # --- BM25 leg ---
    bm25_query: dict[str, Any] = {
        "size": top_k * 2,
        "query": {
            "bool": {
                "must": [{"match": {text_field: {"query": query_text, "operator": "or"}}}],
                "filter": must_filter,
            }
        },
        "_source": {"excludes": ["embedding"]},
    }
    bm25_resp = client.search(index=index_name, body=bm25_query)
    bm25_hits = {h["_id"]: h for h in bm25_resp["hits"]["hits"]}
    bm25_max = max((h["_score"] for h in bm25_resp["hits"]["hits"]), default=1.0) or 1.0

    # --- kNN leg ---
    knn_query: dict[str, Any] = {
        "size": top_k * 2,
        "query": {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": query_embedding, "k": top_k * 2}}}],
                "filter": must_filter,
            }
        },
        "_source": {"excludes": ["embedding"]},
    }
    knn_resp = client.search(index=index_name, body=knn_query)
    knn_hits = {h["_id"]: h for h in knn_resp["hits"]["hits"]}
    knn_max = max((h["_score"] for h in knn_resp["hits"]["hits"]), default=1.0) or 1.0

    # --- Merge via weighted normalized scores ---
    all_ids = set(bm25_hits.keys()) | set(knn_hits.keys())
    scored: list[tuple[float, dict]] = []
    for doc_id in all_ids:
        bm25_score = (bm25_hits[doc_id]["_score"] / bm25_max) if doc_id in bm25_hits else 0.0
        knn_score = (knn_hits[doc_id]["_score"] / knn_max) if doc_id in knn_hits else 0.0
        combined = CFG.bm25_weight * bm25_score + CFG.vector_weight * knn_score
        source = (bm25_hits.get(doc_id) or knn_hits[doc_id])["_source"]
        source["_score"] = combined
        source["_id"] = doc_id
        scored.append((combined, source))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: INDEXING GRAPH
# ═══════════════════════════════════════════════════════════════════════════


class IngestState(TypedDict):
    pdf_dir: str
    pdf_files: list[str]
    all_chunks: Annotated[list[dict], operator.add]
    doc_records: Annotated[list[dict], operator.add]
    status: str


def discover_pdfs(state: IngestState) -> Command:
    """Find all PDF files in the directory."""
    pdf_dir = state["pdf_dir"]
    pdf_files = sorted(str(p) for p in Path(pdf_dir).rglob("*.pdf"))
    log.info(f"Discovered {len(pdf_files)} PDFs in {pdf_dir}")
    # Fan-out: send each batch to a parallel processing node
    batch_size = CFG.ingest_batch_size
    sends = []
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i : i + batch_size]
        sends.append(Send("process_pdf_batch", {"pdf_batch": batch}))
    return Command(update={"pdf_files": pdf_files}, goto=sends)


class PdfBatchInput(TypedDict):
    pdf_batch: list[str]


def process_pdf_batch(state: PdfBatchInput) -> Command:
    """Process a batch of PDFs: extract text, chunk, summarize, embed."""
    batch = state["pdf_batch"]
    all_chunks: list[dict] = []
    doc_records: list[dict] = []

    for pdf_path in batch:
        try:
            pages = extract_pdf_text(pdf_path)
            if not pages:
                log.warning(f"No text extracted from {pdf_path}")
                continue

            # Chunk
            chunks = chunk_pages(pages, pdf_path)
            all_chunks.extend(chunks)

            # Summarize
            summary = summarize_document(pages)
            doc_records.append(
                {
                    "source_file": pdf_path,
                    "filename": Path(pdf_path).name,
                    "summary": summary,
                    "total_pages": len(pages),
                }
            )
            log.info(f"Processed {pdf_path}: {len(pages)} pages, {len(chunks)} chunks")
        except Exception as e:
            log.error(f"Failed to process {pdf_path}: {e}")

    return Command(update={"all_chunks": all_chunks, "doc_records": doc_records}, goto="embed_and_store")


def embed_and_store(state: IngestState) -> Command:
    """Embed all chunks and doc summaries, then bulk-index into OpenSearch."""
    ensure_indexes()

    # --- Embed doc summaries ---
    doc_records = state["doc_records"]
    if doc_records:
        doc_texts = [d["summary"] for d in doc_records]
        doc_embeddings = embed_texts(doc_texts)
        for rec, emb in zip(doc_records, doc_embeddings):
            rec["embedding"] = emb
        bulk_index_docs(doc_records, CFG.doc_index_name)
        log.info(f"Indexed {len(doc_records)} doc summaries")

    # --- Embed chunks ---
    all_chunks = state["all_chunks"]
    if all_chunks:
        chunk_texts = [c["text"] for c in all_chunks]
        chunk_embeddings = embed_texts(chunk_texts)
        for chunk, emb in zip(all_chunks, chunk_embeddings):
            chunk["embedding"] = emb
        # Bulk index in batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            bulk_index_docs(all_chunks[i : i + batch_size], CFG.chunk_index_name)
        log.info(f"Indexed {len(all_chunks)} chunks")

    return Command(update={"status": f"Indexed {len(doc_records)} docs, {len(all_chunks)} chunks"})


def build_ingest_graph() -> StateGraph:
    g = StateGraph(IngestState)
    g.add_node("discover_pdfs", discover_pdfs)
    g.add_node("process_pdf_batch", process_pdf_batch)
    g.add_node("embed_and_store", embed_and_store)

    g.add_edge(START, "discover_pdfs")
    # discover_pdfs uses Send → process_pdf_batch (dynamic fan-out)
    # process_pdf_batch → embed_and_store (via Command goto)
    g.add_edge("embed_and_store", END)
    return g


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 + 3: QUERY GRAPH
# ═══════════════════════════════════════════════════════════════════════════


class QueryState(TypedDict):
    question: str
    sub_queries: list[str]
    retrieved_chunks: list[dict]
    top_doc_files: list[str]
    pdf_groups: dict[str, list[dict]]  # source_file → chunks
    extractions: Annotated[list[dict], operator.add]  # fan-in reducer
    synthesis: str
    citations: list[dict]
    gap_fill_count: int
    needs_gap_fill: bool


# -- Step 2A: Query Decomposition --


def decompose_query(state: QueryState) -> dict:
    """Break complex questions into sub-queries for independent retrieval."""
    question = state["question"]
    raw = llm_call(
        f"""Given this user question, decompose it into 1-4 independent sub-queries that can
each be searched separately. If the question is simple, return just the original question.

Return ONLY a JSON array of strings, nothing else.

Question: {question}""",
        system="You decompose questions into search sub-queries. Return valid JSON only.",
    )
    try:
        sub_queries = json.loads(raw)
        if not isinstance(sub_queries, list):
            sub_queries = [question]
    except json.JSONDecodeError:
        sub_queries = [question]

    log.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
    return {"sub_queries": sub_queries}


# -- Step 2B: Two-Stage Hybrid Retrieval --


def retrieve(state: QueryState) -> dict:
    """Two-stage retrieval: doc-level filter → chunk-level search → re-rank."""
    sub_queries = state["sub_queries"]

    # Embed all sub-queries at once
    query_embeddings = embed_texts(sub_queries)

    # --- Stage 1: Doc-level coarse filter ---
    doc_scores: dict[str, float] = {}
    for sq, emb in zip(sub_queries, query_embeddings):
        docs = hybrid_search(sq, emb, CFG.doc_index_name, CFG.doc_top_k, text_field="summary")
        for doc in docs:
            src = doc["source_file"]
            doc_scores[src] = doc_scores.get(src, 0.0) + doc["_score"]

    # Top docs across all sub-queries
    top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[: CFG.doc_top_k]
    top_doc_files = [src for src, _ in top_docs]
    log.info(f"Doc-level filter: {len(top_doc_files)} candidate PDFs")

    # --- Stage 2: Chunk-level fine-grained search (scoped to top docs) ---
    chunk_scores: dict[str, tuple[float, dict]] = {}
    for sq, emb in zip(sub_queries, query_embeddings):
        chunks = hybrid_search(
            sq, emb, CFG.chunk_index_name, CFG.chunk_top_k, filter_source_files=top_doc_files
        )
        for chunk in chunks:
            cid = chunk["chunk_id"]
            existing_score = chunk_scores[cid][0] if cid in chunk_scores else 0.0
            # Boost chunks from higher-ranked docs
            doc_rank_boost = 1.0
            src = chunk["source_file"]
            if src in dict(top_docs):
                rank = top_doc_files.index(src)
                doc_rank_boost = 1.0 + 0.1 * (len(top_doc_files) - rank) / len(top_doc_files)
            new_score = existing_score + chunk["_score"] * doc_rank_boost
            chunk_scores[cid] = (new_score, chunk)

    # Re-rank and take top K
    ranked = sorted(chunk_scores.values(), key=lambda x: x[0], reverse=True)[: CFG.rerank_top_k]
    retrieved_chunks = [chunk for _, chunk in ranked]

    # Group by source file for fan-out extraction
    pdf_groups: dict[str, list[dict]] = {}
    for chunk in retrieved_chunks:
        src = chunk["source_file"]
        pdf_groups.setdefault(src, []).append(chunk)

    log.info(f"Retrieved {len(retrieved_chunks)} chunks from {len(pdf_groups)} PDFs")
    return {
        "retrieved_chunks": retrieved_chunks,
        "top_doc_files": top_doc_files,
        "pdf_groups": pdf_groups,
    }


# -- Step 2C+2D: Fan-out per-PDF extraction --


class ExtractionInput(TypedDict):
    source_file: str
    chunks: list[dict]
    question: str
    sub_queries: list[str]


def dispatch_extractions(state: QueryState) -> Command:
    """Fan-out: send each PDF group to a parallel extraction worker."""
    pdf_groups = state["pdf_groups"]
    sends = []
    for source_file, chunks in pdf_groups.items():
        sends.append(
            Send(
                "extract_from_pdf",
                {
                    "source_file": source_file,
                    "chunks": chunks,
                    "question": state["question"],
                    "sub_queries": state["sub_queries"],
                },
            )
        )
    log.info(f"Dispatching extraction to {len(sends)} parallel workers")
    return Command(goto=sends)


def extract_from_pdf(state: ExtractionInput) -> Command:
    """
    Extract structured facts from a single PDF's chunks.
    Expands context by reading surrounding pages if possible.
    """
    source_file = state["source_file"]
    chunks = state["chunks"]
    question = state["question"]

    # Build context from chunks (with page references)
    context_parts = []
    all_pages = set()
    for chunk in chunks:
        pages = chunk.get("pages", [])
        all_pages.update(pages)
        page_ref = f"[pages {min(pages)}-{max(pages)}]" if pages else ""
        context_parts.append(f"{page_ref}\n{chunk['text']}")

    # Expand context: read surrounding pages from original PDF
    if all_pages:
        try:
            expanded_pages = set()
            for p in all_pages:
                for offset in range(-CFG.context_expand_pages, CFG.context_expand_pages + 1):
                    expanded_pages.add(p + offset)
            expanded_pages = {p for p in expanded_pages if p >= 1}

            pdf_pages = extract_pdf_text(source_file)
            pdf_page_map = {p["page"]: p["text"] for p in pdf_pages}
            extra_pages = expanded_pages - all_pages
            for p in sorted(extra_pages):
                if p in pdf_page_map:
                    context_parts.append(f"[expanded context - page {p}]\n{pdf_page_map[p]}")
        except Exception as e:
            log.warning(f"Could not expand context for {source_file}: {e}")

    combined_context = "\n\n---\n\n".join(context_parts)

    # Truncate if too large (budget per PDF = total budget / number of PDFs)
    max_chars = (CFG.max_context_tokens * CFG.chars_per_token) // max(len(chunks), 1)
    if len(combined_context) > max_chars:
        combined_context = combined_context[:max_chars] + "\n...[truncated]"

    # LLM extraction
    extraction_prompt = f"""You are analyzing a PDF document: {Path(source_file).name}

QUESTION: {question}

DOCUMENT CONTENT:
{combined_context}

Extract ALL relevant information from this document that helps answer the question.
Return a JSON object with these fields:
{{
  "source_file": "{source_file}",
  "filename": "{Path(source_file).name}",
  "relevant": true/false,
  "answer_fragments": ["direct quotes or paraphrased answers"],
  "facts": ["key facts found"],
  "numbers": {{"metric_name": "value"}},
  "page_references": [list of page numbers where info was found],
  "confidence": "high" | "medium" | "low"
}}

Return ONLY valid JSON."""

    raw = llm_call(extraction_prompt, system="You extract structured information from documents. Return valid JSON only.")
    try:
        extraction = json.loads(raw)
    except json.JSONDecodeError:
        extraction = {
            "source_file": source_file,
            "filename": Path(source_file).name,
            "relevant": False,
            "answer_fragments": [],
            "facts": [raw[:500]],
            "numbers": {},
            "page_references": sorted(all_pages),
            "confidence": "low",
        }

    return Command(update={"extractions": [extraction]})


# -- Step 3A+3B: Merge, gap-fill check, synthesize --


def merge_and_check_gaps(state: QueryState) -> dict:
    """Merge per-PDF extractions, detect contradictions and gaps."""
    extractions = state["extractions"]
    question = state["question"]
    gap_fill_count = state.get("gap_fill_count", 0)

    relevant = [e for e in extractions if e.get("relevant", False)]
    if not relevant:
        relevant = extractions  # Fall back to all if none marked relevant

    extraction_summary = json.dumps(relevant, indent=2, default=str)

    check_prompt = f"""Given these extractions from multiple PDFs for the question: "{question}"

EXTRACTIONS:
{extraction_summary}

Analyze:
1. Are there contradictions between documents?
2. Are there information gaps that a targeted search might fill?
3. Is there enough information to answer the question fully?

Return JSON:
{{
  "has_contradictions": true/false,
  "contradictions": ["description of each"],
  "has_gaps": true/false,
  "gaps": ["what info is missing"],
  "sufficient_to_answer": true/false,
  "gap_queries": ["targeted search queries to fill gaps, max 2"]
}}

Return ONLY valid JSON."""

    raw = llm_call(check_prompt, system="You analyze document extractions for completeness. Return valid JSON only.")
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        analysis = {"has_gaps": False, "sufficient_to_answer": True, "gap_queries": []}

    needs_gap_fill = (
        analysis.get("has_gaps", False)
        and not analysis.get("sufficient_to_answer", True)
        and len(analysis.get("gap_queries", [])) > 0
        and gap_fill_count < CFG.max_gap_fill_loops
    )

    if needs_gap_fill:
        # Inject gap-fill queries as new sub-queries for another retrieval round
        log.info(f"Gap-fill round {gap_fill_count + 1}: {analysis['gap_queries']}")
        return {
            "sub_queries": analysis["gap_queries"],
            "needs_gap_fill": True,
            "gap_fill_count": gap_fill_count + 1,
        }

    return {"needs_gap_fill": False}


def route_after_gap_check(state: QueryState) -> str:
    """Route: loop back to retrieval if gaps found, else synthesize."""
    if state.get("needs_gap_fill", False):
        return "retrieve"
    return "synthesize"


# -- Step 3C: Final synthesis --


def synthesize(state: QueryState) -> dict:
    """Generate final cross-PDF answer with citations."""
    extractions = state["extractions"]
    question = state["question"]

    relevant = [e for e in extractions if e.get("relevant", False)]
    if not relevant:
        relevant = extractions

    extraction_text = json.dumps(relevant, indent=2, default=str)

    synth_prompt = f"""You are synthesizing insights from multiple PDF documents to answer a question.

QUESTION: {question}

EXTRACTIONS FROM INDIVIDUAL PDFs:
{extraction_text}

Instructions:
1. Provide a direct, comprehensive answer to the question.
2. Cross-reference information across PDFs — highlight agreements, contradictions, and trends.
3. For EVERY claim, cite the source: [filename, page X].
4. If comparing numeric data, present as a table.
5. Flag any conflicting information with confidence levels.
6. Note any limitations or gaps in the available data.

Provide a well-structured answer:"""

    synthesis = llm_call(synth_prompt, system="You synthesize cross-document insights with precise citations.")

    # Build citations list
    citations = []
    for ext in relevant:
        citations.append(
            {
                "file": ext.get("filename", "unknown"),
                "pages": ext.get("page_references", []),
                "confidence": ext.get("confidence", "unknown"),
            }
        )

    return {"synthesis": synthesis, "citations": citations}


def build_query_graph() -> StateGraph:
    g = StateGraph(QueryState)

    g.add_node("decompose_query", decompose_query)
    g.add_node("retrieve", retrieve)
    g.add_node("dispatch_extractions", dispatch_extractions)
    g.add_node("extract_from_pdf", extract_from_pdf)
    g.add_node("merge_and_check_gaps", merge_and_check_gaps)
    g.add_node("synthesize", synthesize)

    g.add_edge(START, "decompose_query")
    g.add_edge("decompose_query", "retrieve")
    g.add_edge("retrieve", "dispatch_extractions")
    # dispatch_extractions uses Send → extract_from_pdf (dynamic fan-out)
    # extract_from_pdf results fan-in via Annotated reducer on extractions
    g.add_edge("extract_from_pdf", "merge_and_check_gaps")
    g.add_conditional_edges("merge_and_check_gaps", route_after_gap_check, ["retrieve", "synthesize"])
    g.add_edge("synthesize", END)

    return g


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════


def run_ingest(pdf_dir: str):
    """Run the indexing pipeline on a directory of PDFs."""
    graph = build_ingest_graph().compile()
    result = graph.invoke(
        {
            "pdf_dir": pdf_dir,
            "pdf_files": [],
            "all_chunks": [],
            "doc_records": [],
            "status": "",
        }
    )
    log.info(f"Ingest complete: {result['status']}")
    return result


def run_query(question: str) -> dict:
    """Run the query + synthesis pipeline."""
    graph = build_query_graph().compile()
    result = graph.invoke(
        {
            "question": question,
            "sub_queries": [],
            "retrieved_chunks": [],
            "top_doc_files": [],
            "pdf_groups": {},
            "extractions": [],
            "synthesis": "",
            "citations": [],
            "gap_fill_count": 0,
            "needs_gap_fill": False,
        }
    )
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["synthesis"])
    print("\n" + "-" * 80)
    print("SOURCES")
    print("-" * 80)
    for cite in result["citations"]:
        pages = ", ".join(str(p) for p in cite["pages"]) if cite["pages"] else "N/A"
        print(f"  • {cite['file']} (pages: {pages}) [confidence: {cite['confidence']}]")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="PDF Semantic Search Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="Index a directory of PDFs")
    idx.add_argument("--pdf-dir", required=True, help="Path to directory containing PDFs")

    qry = sub.add_parser("query", help="Query the indexed PDFs")
    qry.add_argument("--question", required=True, help="Question to answer")

    args = parser.parse_args()

    if args.command == "index":
        run_ingest(args.pdf_dir)
    elif args.command == "query":
        run_query(args.question)


if __name__ == "__main__":
    main()
