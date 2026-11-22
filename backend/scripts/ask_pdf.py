import sys

from app.rag.pipeline import build_records, search


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m scripts.ask_pdf path/to/file.pdf")
    path = sys.argv[1]
    print("Indexing...")
    records = build_records(path)

    while True:
        inputs = input("\nEnter a relevant question:")
        if not inputs.strip():
            break
        results = search(inputs, records, 5)
        for rank, result in enumerate(results, start=1):
            print()
            print(
                f"rank:{rank} , page: {result['page']} , score: {result['score']:.2f} , text: {result['text'].replace('\n', ' ')}"
            )
            print("-" * 60)


if __name__ == "__main__":
    main()
