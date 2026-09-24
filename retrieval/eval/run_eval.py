"""Evaluation runner (needs the live stack: Qdrant + models [+ LLM for `rag`]).

    python -m retrieval.eval.run_eval retrieval [--pool 20] [--top-n 5] [--categories rare_word exact_word]
    python -m retrieval.eval.run_eval rag [--limit 10]

retrieval  A. Retrieval: dense-only / sparse-only / fused (RRF) rankings, scored with
              Recall@K, Precision@K, MRR, nDCG@K per category, from ONE hybrid
              query per case (channel rankings come from the per-channel ranks
              kept on each candidate).
           B. Reranking: the top `pool` fused candidates are reranked and the
              top-n scored the same way, so before/after are directly comparable.
           Also reports per-stage latency (mean/p50/p95) as the performance baseline.
rag        C/D. Runs answer_question() per case and reports grounding metrics
              (supported/unsupported claim rate, citation validity) and abstention
              behaviour. Faithfulness / answer relevance / context precision remain
              in retrieval.eval.ragas_eval (LLM-judged); this runner does not replace it.

Results are written to data/eval/<kind>_<timestamp>.json.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from retrieval.eval import metrics
from retrieval.eval.gold import load_gold

OUT_DIR = Path("data/eval")
KS = (1, 3, 5, 10)


def _rank_by(results, rank_key):
    ranked = [r for r in results if r.get(rank_key) is not None]
    return [r["chunk_id"] for r in sorted(ranked, key=lambda r: r[rank_key])]


def evaluate_retrieval(cases, search_fn, rerank_fn=None, pool=20, top_n=5, ks=KS):
    """search_fn(question, top_k, where) -> RetrievalOutput; rerank_fn(question, candidates, top_n) -> list.
    Only cases with relevant chunks are scored (abstain/ambiguous cases have none)."""
    stages = {"dense": {}, "sparse": {}, "fused": {}, "reranked": {}}
    latencies = {}
    filter_violations = []

    for case in cases:
        relevant = case.get("relevant_chunks") or []
        if not relevant:
            continue

        where = {"book_id": case["book_id"]} if case.get("book_id") else None
        out = search_fn(case["question"], max(pool, max(ks), 50), where)

        for stage, ms in out.timings_ms.items():
            latencies.setdefault(f"retrieval.{stage}", []).append(ms)

        if where:
            wrong = [r["chunk_id"] for r in out.results if r.get("book_id") != case["book_id"]]
            if wrong:
                filter_violations.append({"case": case["id"], "chunks": wrong[:5]})

        rankings = {
            "dense": _rank_by(out.results, "dense_rank"),
            "sparse": _rank_by(out.results, "sparse_rank"),
            "fused": [r["chunk_id"] for r in out.results],
        }

        if rerank_fn is not None:
            reranked = rerank_fn(case["question"], out.results[:pool], top_n)
            rankings["reranked"] = [r["chunk_id"] for r in reranked]

        for stage, ranked in rankings.items():
            stages[stage].setdefault(case["category"], []).append(metrics.score_ranking(ranked, relevant, ks))

    return {
        "stages": {stage: metrics.aggregate(rows) for stage, rows in stages.items() if rows},
        "latency_ms": metrics.latency_summary(latencies),
        "filter_violations": filter_violations,
    }


def evaluate_rag(cases, answer_fn):
    """answer_fn(question, where) -> answer_question() result dict."""
    groundings, abstentions, rows = [], [], []

    for case in cases:
        where = {"book_id": case["book_id"]} if case.get("book_id") else None
        result = answer_fn(case["question"], where)
        grounding = result.get("grounding")

        if grounding:
            groundings.append(grounding)
        abstained = bool(grounding and grounding.get("abstained")) or result.get("error") is not None
        abstentions.append((case["expected_behavior"], abstained))

        cited = {c for g in [grounding or {}] for c in g.get("citations", {}).get("cited_ids", [])}
        by_id = {e["evidence_id"]: e["chunk_id"] for e in result.get("evidence", [])}
        rows.append({
            "id": case["id"], "category": case["category"], "expected": case["expected_behavior"],
            "abstained": abstained, "error": result.get("error"),
            "grounded": grounding and grounding["grounded"],
            "cited_chunks": sorted(by_id[c] for c in cited if c in by_id),
            "cites_relevant_chunk": bool(set(case.get("relevant_chunks", [])) & {by_id[c] for c in cited if c in by_id}),
        })

    answerable = [r for r in rows if r["expected"] == "answer" and not r["abstained"]]
    return {
        "grounding": metrics.grounding_summary(groundings),
        "abstention": metrics.abstention_summary(abstentions),
        # Of answered, answerable questions: how often does the answer cite a chunk labelled relevant?
        "citation_correctness": (sum(r["cites_relevant_chunk"] for r in answerable) / len(answerable)) if answerable else None,
        "cases": rows,
    }


def _print_stage_table(name, agg):
    print(f"\n[{name}]")
    cols = ["n", "mrr"] + [f"recall@{k}" for k in KS] + ["ndcg@5"]
    print("category".ljust(24) + "".join(c.rjust(11) for c in cols))
    for category, row in agg.items():
        cells = ["-" if row.get(c) is None else (str(row[c]) if c == "n" else f"{row[c]:.3f}") for c in cols]
        print(category.ljust(24) + "".join(cell.rjust(11) for cell in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["retrieval", "rag"])
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--pool", type=int, default=20, help="fused candidates passed to the reranker")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    cases = load_gold(categories=args.categories)
    if args.limit:
        cases = cases[:args.limit]

    if args.mode == "retrieval":
        from retrieval.retrieve.reranker import rerank
        from retrieval.retrieve.search import hybrid_search

        report = evaluate_retrieval(
            cases,
            search_fn=lambda q, k, where: hybrid_search(q, top_k=k, where=where),
            rerank_fn=None if args.no_rerank else (lambda q, cands, n: rerank(q, cands, top_n=n)),
            pool=args.pool, top_n=args.top_n,
        )
        for stage, agg in report["stages"].items():
            _print_stage_table(stage, agg)
        print("\nLatency (ms):", json.dumps({k: {m: round(v, 1) for m, v in s.items()} for k, s in report["latency_ms"].items()}))
        print("Filter violations:", report["filter_violations"] or "none")
    else:
        from retrieval.pipeline.query import answer_question

        report = evaluate_rag(cases, lambda q, where: answer_question(q, top_k=args.pool, top_n=args.top_n, where=where))
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
