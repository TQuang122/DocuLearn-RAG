"""Vector-store replacement tests."""

from __future__ import annotations


def test_old_version_filter_keeps_current_document_id() -> None:
    """Given a replacement, cleanup must target only older versions of that filename."""
    from src.store import _old_document_versions_filter

    cleanup_filter = _old_document_versions_filter("document.pdf", "current-id")

    assert cleanup_filter.must is not None
    assert cleanup_filter.must_not is not None
    assert cleanup_filter.must[0].key == "metadata.filename"
    assert cleanup_filter.must_not[0].key == "metadata.document_id"
