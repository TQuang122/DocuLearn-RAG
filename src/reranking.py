from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from src.schemas import RetrievedChunk

TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\w+", re.UNICODE)
STOP_WORDS = {
    "ai",
    "bao",
    "bị",
    "các",
    "cho",
    "có",
    "của",
    "để",
    "được",
    "dùng",
    "gì",
    "giúp",
    "hình",
    "khi",
    "là",
    "mô",
    "một",
    "nào",
    "này",
    "như",
    "những",
    "theo",
    "thì",
    "trong",
    "từ",
    "và",
    "về",
    "với",
}


def tokenize_for_reranking(text: str) -> list[str]:
    tokens = (match.group(0) for match in TOKEN_PATTERN.finditer(text.casefold()))
    return [
        token
        for token in tokens
        if len(token) > 1 and token not in STOP_WORDS and not token.isdigit()
    ]


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def bm25_scores(query: str, chunks: list[RetrievedChunk]) -> list[float]:
    if not chunks:
        return []
    token_counts = [Counter(tokenize_for_reranking(chunk.text)) for chunk in chunks]
    document_frequency = Counter(token for counts in token_counts for token in counts)
    query_terms = list(dict.fromkeys(tokenize_for_reranking(query)))
    average_length = sum(sum(counts.values()) for counts in token_counts) / len(token_counts)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for counts in token_counts:
        document_length = sum(counts.values())
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1
                + (len(chunks) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * document_length / max(average_length, 1)
            )
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def rerank_candidates(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    k: int,
    dense_weight: float = 0.0,
    max_chunks_per_page: int | None = 1,
) -> list[RetrievedChunk]:
    if k < 1:
        raise ValueError("k must be positive.")
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be between 0 and 1.")
    if max_chunks_per_page is not None and max_chunks_per_page < 1:
        raise ValueError("max_chunks_per_page must be positive when provided.")
    if not chunks:
        return []

    dense_scores = _normalize([chunk.score for chunk in chunks])
    lexical_scores = _normalize(bm25_scores(query, chunks))
    combined_scores = [
        dense_weight * dense + (1 - dense_weight) * lexical
        for dense, lexical in zip(dense_scores, lexical_scores, strict=True)
    ]
    ranked = sorted(
        zip(combined_scores, chunks, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )

    selected: list[RetrievedChunk] = []
    page_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    for score, chunk in ranked:
        page_key = (chunk.metadata.filename, chunk.metadata.page)
        if max_chunks_per_page is not None and page_counts[page_key] >= max_chunks_per_page:
            continue
        page_counts[page_key] += 1
        selected.append(chunk.model_copy(update={"score": score}))
        if len(selected) == k:
            break
    return selected
