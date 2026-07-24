"""Ragas evaluation adapter for grounded DocuLearn-RAG answers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.embeddings import get_embeddings
from src.rag import answer as default_answer_fn
from src.schemas import RagAnswer


class EvaluationCase(TypedDict):
    question: str
    ground_truth: str


def _evaluation_dataset(
    test_cases: list[EvaluationCase],
    answer_fn: Callable[[str], RagAnswer],
) -> Dataset:
    if not test_cases:
        raise ValueError("test_cases cannot be empty.")

    data: dict[str, list[object]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for index, case in enumerate(test_cases):
        question = case.get("question", "").strip()
        ground_truth = case.get("ground_truth", "").strip()
        if not question or not ground_truth:
            raise ValueError(
                "Each test case must contain non-empty 'question' and "
                f"'ground_truth' values (index {index})."
            )
        rag_response = answer_fn(question)
        data["question"].append(question)
        data["answer"].append(rag_response.answer)
        data["contexts"].append([chunk.text for chunk in rag_response.chunks])
        data["ground_truth"].append(ground_truth)
    return Dataset.from_dict(data)


def run_evaluation(
    test_cases: list[EvaluationCase],
    *,
    answer_fn: Callable[[str], RagAnswer] = default_answer_fn,
    timeout_s: int = 180,
    max_retries: int = 3,
    max_workers: int = 4,
):
    """Evaluate grounded answers with Ragas 0.3's stable LangChain adapter."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Ragas evaluation.")

    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    dataset = _evaluation_dataset(test_cases, answer_fn)
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=timeout_s,
            max_retries=max_retries,
            max_workers=max_workers,
        ),
    )
