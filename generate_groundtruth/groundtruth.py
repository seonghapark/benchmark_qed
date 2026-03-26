#!/usr/bin/env python3
"""Generate ground-truth answers for selected seed question files.

This script reuses the local benchmark-qed AutoE components from autoe.py
to produce extractive, evidence-based answers for every
/storage/output_fast_runs/seed_*/data_local_questions/selected_questions_text.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from autoe import (
    LocalAutoE,
    build_title_to_text,
    load_questions,
    resolve_index_dir,
    split_sentences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate groundtruth answers for seed question sets using benchmark-qed AutoE"
    )
    parser.add_argument(
        "--questions-glob",
        type=str,
        default="/storage/output_fast_runs/seed_*/data_local_questions/selected_questions_text.json",
        help="Glob to locate selected question files",
    )
    parser.add_argument("--dataset-dir", type=str, default="/storage/data/new_samples")
    parser.add_argument("--index-dir", type=str, default="/storage/hybrid_100_newdata")
    parser.add_argument("--fallback-index-dir", type=str, default="/storage/hybrid_test_100_newdata")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--answer-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--dense-top-k", type=int, default=40)
    parser.add_argument("--sparse-top-k", type=int, default=40)
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--sparse-weight", type=float, default=0.5)
    parser.add_argument(
        "--limit-per-seed",
        type=int,
        default=0,
        help="Maximum number of questions to answer per seed (0 means all)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="groundtruth_answers.json",
        help="Output filename written into each seed data_local_questions folder",
    )
    parser.add_argument(
        "--qa-only-name",
        type=str,
        default="groundtruth_qa_only.json",
        help="QA-only output filename written into each seed data_local_questions folder",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seed folders that already have output files",
    )
    return parser.parse_args()


def build_qa_pairs(
    autoe: LocalAutoE,
    questions: list[str],
    title_to_text: dict[str, str],
) -> list[dict[str, Any]]:
    qa_pairs: list[dict[str, Any]] = []

    for pair_id, question in enumerate(questions, start=1):
        retrieved = autoe.retrieve(question)
        retrieved = autoe.rerank_chunks(question, retrieved)

        grouped_evidence: dict[str, list[str]] = defaultdict(list)
        source_titles: list[str] = []
        for item in retrieved:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            title = metadata.get("title") if isinstance(metadata, dict) else None
            if not isinstance(title, str) or not title:
                title = "unknown"

            if title not in source_titles:
                source_titles.append(title)

            if title in title_to_text:
                grouped_evidence[title].extend(split_sentences(title_to_text[title])[:8])

            text = item.get("text", "") if isinstance(item, dict) else ""
            if isinstance(text, str) and text.strip():
                grouped_evidence[title].extend(split_sentences(text)[:8])

        compact_grouped_evidence: dict[str, list[str]] = {}
        for title, sentences in grouped_evidence.items():
            deduped: list[str] = []
            seen: set[str] = set()
            for sentence in sentences:
                key = sentence.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(sentence)
            if deduped:
                compact_grouped_evidence[title] = deduped[:80]

        answer = autoe.answer(question, compact_grouped_evidence)
        qa_pairs.append(
            {
                "id": pair_id,
                "question": question,
                "answer": answer,
                "source_titles": source_titles[:5],
                "retrieved_chunks": [
                    {
                        "chunk_idx": item["chunk_idx"],
                        "text": item["text"],
                        "metadata": item.get("metadata", {}),
                        "scores": item.get("scores", {}),
                    }
                    for item in retrieved
                ],
            }
        )

    return qa_pairs


def write_outputs(
    question_dir: Path,
    output_name: str,
    qa_only_name: str,
    payload: dict[str, Any],
) -> None:
    output_path = question_dir / output_name
    qa_only_path = question_dir / qa_only_name

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    qa_only = [
        {"question": item.get("question"), "answer": item.get("answer")}
        for item in payload.get("qa_pairs", [])
        if isinstance(item, dict)
    ]
    qa_only_path.write_text(json.dumps(qa_only, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    matched_files = sorted(Path(p) for p in glob.glob(args.questions_glob))
    if not matched_files:
        raise SystemExit(f"No files matched: {args.questions_glob}")

    index_dir = resolve_index_dir(Path(args.index_dir), Path(args.fallback_index_dir))
    title_to_text = build_title_to_text(Path(args.dataset_dir))

    autoe = LocalAutoE(
        index_dir=index_dir,
        embedding_model_name=args.embedding_model,
        answer_model_name=args.answer_model,
        reranker_model_name=args.reranker_model,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        final_top_k=args.final_top_k,
        nprobe=args.nprobe,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        use_reranker=not args.no_reranker,
    )

    completed = 0
    skipped = 0

    for questions_file in matched_files:
        question_dir = questions_file.parent
        output_path = question_dir / args.output_name
        qa_only_path = question_dir / args.qa_only_name

        if args.skip_existing and output_path.exists() and qa_only_path.exists():
            skipped += 1
            print(f"Skipping existing outputs in {question_dir}")
            continue

        questions = load_questions(question_dir, limit=args.limit_per_seed)
        if not questions:
            print(f"No questions found in {question_dir}, skipping")
            skipped += 1
            continue

        qa_pairs = build_qa_pairs(autoe=autoe, questions=questions, title_to_text=title_to_text)
        payload = {
            "config": {
                "questions_file": str(questions_file),
                "dataset_dir": args.dataset_dir,
                "index_dir": str(index_dir),
                "embedding_model": args.embedding_model,
                "answer_model": args.answer_model,
                "reranker_model": args.reranker_model,
                "num_pairs": len(qa_pairs),
            },
            "qa_pairs": qa_pairs,
        }
        write_outputs(
            question_dir=question_dir,
            output_name=args.output_name,
            qa_only_name=args.qa_only_name,
            payload=payload,
        )

        completed += 1
        print(f"Generated {len(qa_pairs)} QA pairs in {question_dir}")

    print(f"Done. completed={completed}, skipped={skipped}, matched={len(matched_files)}")


if __name__ == "__main__":
    main()
