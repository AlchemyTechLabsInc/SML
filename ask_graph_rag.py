# ask_graph_rag.py
import argparse
from graph_rag import ask

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*", help="Your question")
    p.add_argument("--entity", action="append", help="Scope to an entity name (e.g., Vendor)")
    args = p.parse_args()

    q = " ".join(args.question) or "Which pages mention contingency or change orders?"
    ans, refs = ask(q, entities=args.entity)
    print("\nQ:", q)
    print("\nA:", ans)
    print("\nCitations:", refs)
