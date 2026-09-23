import json
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    IDBasedContextPrecision,
    IDBasedContextRecall,
)

from src.pipeline.query import answer_question
from src.retrieval.evaluate import TEST_CASES as HAND_LABELED_CASES


class CompatEmbeddings(HuggingFaceEmbeddings):
    """
    ragas 0.4.3's ResponseRelevancy metric still calls the legacy LangChain
    embeddings interface (embed_query/embed_documents), but
    ragas.embeddings.HuggingFaceEmbeddings only implements the newer
    embed_text/embed_texts. Bridge the two.
    """

    def embed_query(self, text):
        return self.embed_text(text)

    def embed_documents(self, texts):
        return self.embed_texts(texts)


GENERATED_TESTSET_PATH = Path("data/eval/generated_testset.json")
RESULTS_DIR = Path("data/eval")


def load_test_cases():
    cases = list(HAND_LABELED_CASES)

    if GENERATED_TESTSET_PATH.exists():
        with GENERATED_TESTSET_PATH.open("r", encoding="utf-8") as f:
            generated = json.load(f)
        cases = cases + [
            {"id": f"gen-{i}", "question": c["question"], "relevant_chunks": c["relevant_chunks"]}
            for i, c in enumerate(generated)
        ]

    return cases


def build_dataset(test_cases):
    samples = []

    for case in test_cases:
        result = answer_question(case["question"], top_k=10, top_n=5)

        samples.append({
            "user_input": case["question"],
            "response": result["answer"] or "",
            "retrieved_contexts": [c["text"] for c in result["reranked"]],
            "retrieved_context_ids": [c["chunk_id"] for c in result["reranked"]],
            "reference_context_ids": case["relevant_chunks"],
        })

    return EvaluationDataset.from_list(samples)


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    judge_llm = llm_factory("openai/gpt-4o-mini", provider="openai", client=client)
    judge_embeddings = CompatEmbeddings(model="BAAI/bge-m3", normalize_embeddings=True)

    test_cases = load_test_cases()
    print(f"Evaluating {len(test_cases)} test cases...")

    dataset = build_dataset(test_cases)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
            IDBasedContextPrecision(),
            IDBasedContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ragas_results_{timestamp}.csv"
    result.to_pandas().to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(result)


if __name__ == "__main__":
    main()
