import argparse

from app.rag.extract import extract_pages
from app.rag.pipeline import build_records, search
from scripts.eval_set import EVAL_SET, OFF_TOPIC

TOP_K = 5


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def validate(path: str) -> None:
    pages = extract_pages(path)
    normalized_pages = [(page_no, normalize(text)) for page_no, text in pages]
    for i, item in enumerate(EVAL_SET, start=1):
        target = normalize(item["answer_substring"])
        found_pages = [
            page_num for page_num, page_text in normalized_pages if target in page_text
        ]
        expected = item["expected_pages"]
        missing = [p for p in expected if p not in found_pages]
        extra = [p for p in found_pages if p not in expected]
        status = "OK" if not missing and not extra else "MISMATCH"
        print(f"Q{i}: {item['question']}")
        print(f"   answer   : {item['answer_substring']!r}")
        print(f"   expected : {expected}")
        print(f"   found on : {found_pages}")
        print(f"   missing  : {missing} | extra: {extra}  [{status}]")
        print("-" * 60)


def first_rank_with_answer(results: list[dict], answer: str) -> int | None:
    answer = normalize(answer)
    for rank, result in enumerate(results, start=1):
        if answer in normalize(result["text"]):
            return rank
    return None


def evaluate(path: str, size: int, overlap: int) -> None:
    """Run every question, print one line each, then a summary."""
    print(f"Settings: size={size} , Overlap={overlap}")
    records = build_records(path, size, overlap)
    print(f"indexed {len(records)} records\n")

    ranks: list[int | None] = []
    best_scores: list[float] = []
    page_hits = 0

    for item in EVAL_SET:
        results = search(item["question"], records, TOP_K)
        rank = first_rank_with_answer(results, item["answer_substring"])
        page_hit = any(r["page"] in item["expected_pages"] for r in results)
        best = results[0]["score"]

        status = "PASS" if rank is not None else "FAIL"
        rank_str = str(rank) if rank is not None else "-"
        hit_str = "yes" if page_hit else "No"
        q_preview = item["question"][:55]

        print(
            f"{status:>4} | rank:{rank_str:>2} | page_hit:{hit_str:<3}"
            f"| best: {best:.2f} | {q_preview}"
        )
        ranks.append(rank)
        best_scores.append(best)
        if page_hit:
            page_hits += 1

    total = len(EVAL_SET)
    found_TOP_K = sum(1 for r in ranks if r is not None)
    found_rank_1 = sum(1 for r in ranks if r == 1)
    mrr = sum(1 / r for r in ranks if r is not None) / total

    print("\n---Summary---")
    print(f"Found in top k: {TOP_K}: {found_TOP_K}/{total}")
    print(f"At rank 1     : {found_rank_1}/{total}")
    print(f"MRR           : {mrr:.3f}")
    print(f"page hits     : {page_hits}/{total}")

    print("\n--- Off-Topic Check ---")
    off_topic_scores: list[float] = []
    for question in OFF_TOPIC:
        results = search(question, records, TOP_K)
        best = results[0]["score"]
        off_topic_scores.append(best)
        print(f"best: {best:.2f} | {question}")

    print(f"\nHighest off-topic best : {max(off_topic_scores):.2f}")
    print(f"Lowest in-document best  : {min(best_scores):.2f}")


def main() -> None:
    """Parse command-line options and run either validation or evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on the transformer pdf"
    )
    parser.add_argument("pdf", help="path to the pdf")
    parser.add_argument(
        "--size", type=int, default=500, help="chunk size in characters"
    )
    parser.add_argument(
        "--overlap", type=int, default=100, help="chunk overlap in characters"
    )
    parser.add_argument(
        "--check", action="store_true", help="only validate the question set"
    )
    args = parser.parse_args()

    if args.check:
        validate(args.pdf)
    else:
        evaluate(args.pdf, args.size, args.overlap)


if __name__ == "__main__":
    main()
