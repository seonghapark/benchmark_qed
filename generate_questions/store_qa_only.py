#!/usr/bin/env python3
"""Extract only qustion and answer fields from a JSON list of QA objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def resolve_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        qa_pairs = data.get("qa_pairs")
        if isinstance(qa_pairs, list):
            return [item for item in qa_pairs if isinstance(item, dict)]

    raise ValueError("Input JSON must be a list of objects or an object containing 'qa_pairs'.")


def extract_qa_only(data: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in resolve_records(data):
        # Accept both spellings from source, but always emit requested key: "qustion".
        question = item.get("qustion", item.get("question"))
        answer = item.get("answer")

        if question is None and answer is None:
            continue

        out.append({"qustion": question, "answer": answer})

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract only qustion and answer fields from a JSON file."
    )
    parser.add_argument(
        "--input",
        default="/storage/benchmark_qed/output_araia1/test2.json",
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        default="/storage/benchmark_qed/output_araia1/qa2.json",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    extracted = extract_qa_only(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(extracted)} records to {output_path}")


if __name__ == "__main__":
    main()
