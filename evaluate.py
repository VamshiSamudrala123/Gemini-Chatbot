"""Opt-in live retrieval evaluation. Sends case questions to Google's embedding API."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from dotenv import load_dotenv

from chatbot import load_documents, split_documents
from providers import create_embeddings, create_vectorstore
from settings import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    settings = replace(load_settings(), retrieval_k=args.k, min_similarity=args.threshold)
    chunks = split_documents(load_documents())
    embeddings = create_embeddings(settings)
    index = create_vectorstore(chunks, settings, embeddings)
    cases = json.loads((root / "evaluation_cases.json").read_text())
    hits, supported = 0, 0
    for case in cases:
        results = index.search(embeddings.embed_query(case["question"]), settings.retrieval_k)
        accepted = [(doc, score) for doc, score in results if score >= settings.min_similarity]
        expected = case.get("expected_section")
        if expected:
            supported += 1
            hit = any(expected.lower() in doc.section.lower() for doc, _ in accepted)
            hits += hit
        else:
            hit = None
        print(json.dumps({
            "question": case["question"], "section_hit": hit,
            "retrieved": [{"section": doc.section, "cosine": round(score, 4)}
                          for doc, score in accepted],
        }, ensure_ascii=False))
    print(f"Supported-question section recall@{settings.retrieval_k}: {hits}/{supported}")
    print("Inspect unrelated/unavailable cases manually; similarity alone does not establish factual support.")


if __name__ == "__main__":
    main()
