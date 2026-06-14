"""CLI tester: python -m agent.run "your question" """

import argparse
import logging
import sys


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ask the medical RAG agent a question")
    parser.add_argument("question", help="A medical question to answer from the indexed references")
    args = parser.parse_args()

    from agent.graph import run_agent  # import after arg parsing: model load is slow

    result = run_agent(args.question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\nCITATIONS")
    print("-" * 70)
    print("\n".join(result["citations"]) if result["citations"] else "(none)")

    print("\nREASONING PATH")
    print("-" * 70)
    for i, step in enumerate(result["reasoning_steps"], 1):
        print(f"{i}. {step}")

    print("\nRETRIES:    ", result["retries"])
    print("GROUNDED:   ", result["grounded"])
    print("LATENCY_ms: ", result["latency_ms"])
    if "trace_url" in result:
        print("TRACE_URL:  ", result["trace_url"])


if __name__ == "__main__":
    sys.exit(main())
