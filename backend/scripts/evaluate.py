import sys

from app.rag.extract import extract_pages
from scripts.eval_set import EVAL_SET


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


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m scripts.evaluate path/to/file.pdf")
    validate(sys.argv[1])


if __name__ == "__main__":
    main()
