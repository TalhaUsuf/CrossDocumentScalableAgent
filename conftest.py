"""
Shared fixtures and pytest configuration for the PDF search pipeline test suite.
"""
import json
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests requiring live external services (embedding, LLM, OpenSearch)")
    config.addinivalue_line("markers", "pipeline: end-to-end pipeline tests requiring all services")
    config.addinivalue_line("markers", "benchmark: benchmark tests driven by benchmark.json")


# ---------------------------------------------------------------------------
# Score tracking for benchmark reporting
# ---------------------------------------------------------------------------

class ScoreTracker:
    """Accumulates per-question benchmark scores and prints a summary."""

    def __init__(self):
        self.results: list[dict] = []

    def record(self, result: dict):
        self.results.append(result)

    def summary(self) -> str:
        if not self.results:
            return "No benchmark results recorded."

        lines = []
        lines.append("")
        lines.append("=" * 90)
        lines.append("BENCHMARK RESULTS SUMMARY")
        lines.append("=" * 90)
        lines.append(
            f"{'ID':<12} {'Source Match':<14} {'KW Hits':<10} {'KW Total':<10} {'KW%':<8} {'Multi-Src':<10}"
        )
        lines.append("-" * 90)

        total_source_match = 0
        total_kw_hits = 0
        total_kw_total = 0
        total_multi_source = 0
        count = len(self.results)

        for r in self.results:
            sm = r.get("source_match", False)
            kw_hits = r.get("keyword_hits", 0)
            kw_total = r.get("total_keywords", 0)
            multi = r.get("multi_source", False)
            kw_pct = (kw_hits / kw_total * 100) if kw_total > 0 else 0.0

            total_source_match += int(sm)
            total_kw_hits += kw_hits
            total_kw_total += kw_total
            total_multi_source += int(multi)

            lines.append(
                f"{r.get('id', '?'):<12} {'YES' if sm else 'NO':<14} {kw_hits:<10} {kw_total:<10} {kw_pct:<8.1f} {'YES' if multi else 'NO':<10}"
            )

        lines.append("-" * 90)
        overall_kw_pct = (total_kw_hits / total_kw_total * 100) if total_kw_total > 0 else 0.0
        lines.append(f"Source Match Accuracy : {total_source_match}/{count} ({total_source_match/count*100:.1f}%)")
        lines.append(f"Keyword Coverage      : {total_kw_hits}/{total_kw_total} ({overall_kw_pct:.1f}%)")
        lines.append(f"Multi-Source Coverage  : {total_multi_source}/{count}")
        lines.append("=" * 90)
        return "\n".join(lines)


_tracker = ScoreTracker()


@pytest.fixture(scope="session")
def score_tracker():
    return _tracker


def pytest_sessionfinish(session, exitstatus):
    """Print the benchmark summary at the end of the test run."""
    if _tracker.results:
        print(_tracker.summary())


# ---------------------------------------------------------------------------
# OpenSearch test index cleanup fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_index_names():
    """Return test-specific index names to avoid polluting production indexes."""
    return {
        "doc_index": "test_pdf_doc_index",
        "chunk_index": "test_pdf_chunk_index",
    }


@pytest.fixture(scope="session")
def cleanup_test_indexes(test_index_names):
    """Delete test indexes after the session completes."""
    yield test_index_names
    try:
        from opensearchpy import OpenSearch
        client = OpenSearch(
            hosts=[{"host": "localhost", "port": 9200}],
            http_auth=("admin", "admin"),
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )
        for idx in test_index_names.values():
            if client.indices.exists(index=idx):
                client.indices.delete(index=idx)
    except Exception:
        pass
