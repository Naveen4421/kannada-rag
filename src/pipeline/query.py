import argparse

from src.retrieval.search import search
from src.retrieval.reranker import rerank
from src.generation.prompt import build_prompt
from src.generation.llm import generate_answer


def answer_question(question, top_k=10, top_n=5, where=None):
    retrieved = search(question, top_k=top_k, where=where)
    reranked = rerank(question, retrieved, top_n=top_n)
    prompt_text = build_prompt(question, reranked)

    answer = None
    error = None

    try:
        answer = generate_answer(prompt_text)
    except NotImplementedError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"LLM call failed: {exc}"

    return {
        "question": question,
        "retrieved": retrieved,
        "reranked": reranked,
        "prompt": prompt_text,
        "answer": answer,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description="Ask a Kannada question against the indexed book collection.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    result = answer_question(args.question, top_k=args.top_k, top_n=args.top_n)

    print("=" * 70)
    print(f"Retrieved (top {args.top_k}):")
    print("=" * 70)
    for rank, chunk in enumerate(result["retrieved"], start=1):
        print(f"{rank}. {chunk['chunk_id']}  score={chunk['score']:.4f}")

    print()
    print("=" * 70)
    print(f"Reranked (top {args.top_n}):")
    print("=" * 70)
    for rank, chunk in enumerate(result["reranked"], start=1):
        print(f"{rank}. {chunk['chunk_id']}  score={chunk['score']:.4f}")

    print()
    print("=" * 70)
    print("Prompt sent to LLM:")
    print("=" * 70)
    print(result["prompt"])

    print()
    print("=" * 70)
    print("Answer:")
    print("=" * 70)
    if result["error"]:
        print(f"[{result['error']}]")
    else:
        print(result["answer"])


if __name__ == "__main__":
    main()
