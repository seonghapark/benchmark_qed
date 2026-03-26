import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    return norm.split()


def exact_match_score(prediction: str, references: list[str]) -> float:
    pred = normalize_text(prediction)
    if not references:
        return 0.0
    for ref in references:
        if pred == normalize_text(ref):
            return 1.0
    return 0.0


def accuracy_score(prediction: str, references: list[str]) -> float:
    # For single-answer QA, accuracy aligns with exact match.
    return exact_match_score(prediction, references)


def token_f1_score(prediction: str, references: list[str]) -> float:
    pred_tokens = tokenize(prediction)
    if not references:
        return 0.0

    best = 0.0
    pred_counter = Counter(pred_tokens)
    pred_len = len(pred_tokens)

    for ref in references:
        ref_tokens = tokenize(ref)
        ref_counter = Counter(ref_tokens)
        overlap = sum((pred_counter & ref_counter).values())
        if overlap == 0:
            f1 = 0.0
        else:
            precision = overlap / max(1, pred_len)
            recall = overlap / max(1, len(ref_tokens))
            f1 = 2.0 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1

    return float(best)


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n_f1(prediction: str, references: list[str], n: int) -> float:
    pred_tokens = tokenize(prediction)
    if not references:
        return 0.0

    pred_counts = _ngram_counts(pred_tokens, n)
    best = 0.0

    for ref in references:
        ref_tokens = tokenize(ref)
        ref_counts = _ngram_counts(ref_tokens, n)

        overlap = sum((pred_counts & ref_counts).values())
        pred_total = sum(pred_counts.values())
        ref_total = sum(ref_counts.values())

        if overlap == 0 or pred_total == 0 or ref_total == 0:
            f1 = 0.0
        else:
            precision = overlap / pred_total
            recall = overlap / ref_total
            f1 = 2.0 * precision * recall / (precision + recall)

        if f1 > best:
            best = f1

    return float(best)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1_score(prediction: str, references: list[str]) -> float:
    pred_tokens = tokenize(prediction)
    if not references:
        return 0.0

    best = 0.0
    for ref in references:
        ref_tokens = tokenize(ref)
        lcs = _lcs_len(pred_tokens, ref_tokens)
        if lcs == 0:
            f1 = 0.0
        else:
            precision = lcs / max(1, len(pred_tokens))
            recall = lcs / max(1, len(ref_tokens))
            f1 = 2.0 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return float(best)


def bleu_score(prediction: str, references: list[str], max_n: int = 4) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens_list = [tokenize(ref) for ref in references]

    if not pred_tokens or not ref_tokens_list:
        return 0.0

    p_ns: list[float] = []
    for n in range(1, max_n + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        pred_total = sum(pred_counts.values())
        if pred_total == 0:
            p_ns.append(0.0)
            continue

        max_ref_counts: Counter[tuple[str, ...]] = Counter()
        for ref_tokens in ref_tokens_list:
            ref_counts = _ngram_counts(ref_tokens, n)
            for gram, count in ref_counts.items():
                if count > max_ref_counts[gram]:
                    max_ref_counts[gram] = count

        clipped = 0
        for gram, count in pred_counts.items():
            clipped += min(count, max_ref_counts[gram])

        # Add-one smoothing to avoid hard zero from higher-order n-grams.
        p_ns.append((clipped + 1.0) / (pred_total + 1.0))

    if any(p <= 0.0 for p in p_ns):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in p_ns) / max_n)

    pred_len = len(pred_tokens)
    ref_lens = [len(tokens) for tokens in ref_tokens_list]
    ref_len = min(ref_lens, key=lambda rl: (abs(rl - pred_len), rl))

    if pred_len > ref_len:
        bp = 1.0
    elif pred_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - (ref_len / pred_len))

    return float(bp * geo_mean)


def load_reference_answers(path: Path) -> dict[str, list[str]]:
    """Load references as {question: [acceptable_answer1, ...]}.

    Supported formats:
      - JSON mapping: {"question": "answer"} or {"question": ["a1", "a2"]}
      - JSON list: [{"question": ..., "answer": ...}, ...]
      - JSON object with qa_pairs: {"qa_pairs": [{"question": ..., "answer": ...}, ...]}
      - JSONL with one object/string per line
    """

    def _to_refs(value: Any) -> list[str]:
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, list):
            out = [v.strip() for v in value if isinstance(v, str) and v.strip()]
            return out
        return []

    def _extract_from_item(item: Any) -> tuple[str | None, list[str]]:
        if not isinstance(item, dict):
            return (None, [])
        q = item.get("question") or item.get("query") or item.get("text")
        if not isinstance(q, str) or not q.strip():
            return (None, [])

        refs: list[str] = []
        candidate_keys = [
            "reference_answers",
            "references",
            "gold_answers",
            "gold",
            "answers",
            "answer",
            "response",
        ]
        for key in candidate_keys:
            refs = _to_refs(item.get(key))
            if refs:
                break

        return (q.strip(), refs)

    refs_by_question: dict[str, list[str]] = {}
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            q, refs = _extract_from_item(item)
            if q and refs:
                refs_by_question[q] = refs
        return refs_by_question

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        if "qa_pairs" in data and isinstance(data["qa_pairs"], list):
            for item in data["qa_pairs"]:
                q, refs = _extract_from_item(item)
                if q and refs:
                    refs_by_question[q] = refs
            return refs_by_question

        # Mapping question -> answers
        non_meta_mapping = True
        for _, value in data.items():
            if not (isinstance(value, str) or isinstance(value, list)):
                non_meta_mapping = False
                break
        if non_meta_mapping:
            for q, value in data.items():
                if not isinstance(q, str) or not q.strip():
                    continue
                refs = _to_refs(value)
                if refs:
                    refs_by_question[q.strip()] = refs
            return refs_by_question

    if isinstance(data, list):
        for item in data:
            q, refs = _extract_from_item(item)
            if q and refs:
                refs_by_question[q] = refs

    return refs_by_question


def _run_judge_model_scoring(
    rows: list[dict[str, Any]],
    model_name: str,
    max_new_tokens: int,
    device: str,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if not rows:
        return []

    if device == "cuda" and torch.cuda.is_available():
        pipeline_device = 0
    elif device == "cpu":
        pipeline_device = -1
    else:
        pipeline_device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    judge = pipeline("text-generation", model=model, tokenizer=tokenizer, device=pipeline_device)

    scored: list[dict[str, Any]] = []
    for row in rows:
        question = row["question"]
        prediction = row["prediction"]
        references = row["references"]
        ref_text = "\n".join(f"- {ref}" for ref in references)

        prompt = (
            "You are evaluating an answer against reference answers.\n"
            "Score from 1 to 5 where 5 is fully correct and 1 is incorrect.\n"
            "Return in this exact format:\n"
            "SCORE: <1-5>\n"
            "REASON: <short reason>\n\n"
            f"QUESTION: {question}\n\n"
            f"REFERENCE ANSWERS:\n{ref_text}\n\n"
            f"PREDICTED ANSWER:\n{prediction}\n"
        )

        out = judge(
            prompt,
            max_new_tokens=max(16, int(max_new_tokens)),
            do_sample=False,
            return_full_text=False,
            pad_token_id=judge.tokenizer.eos_token_id,
        )
        text = str(out[0].get("generated_text", "")).strip()

        match = re.search(r"\b([1-5])\b", text)
        score = float(match.group(1)) if match else 0.0
        reason = text[:500]

        scored.append(
            {
                "question": question,
                "judge_score": score,
                "judge_reason": reason,
            }
        )

    return scored


def evaluate_generated_answers(
    predictions: list[dict[str, str]],
    references_by_question: dict[str, list[str]],
    judge_model_name: str | None = None,
    judge_max_new_tokens: int = 96,
    judge_device: str = "auto",
) -> dict[str, Any]:
    """Evaluate generated answers against references.

    Args:
      predictions: list of {"question": str, "answer": str}
      references_by_question: question -> list of acceptable references
      judge_model_name: optional model name for judge scoring pass

    Returns:
      Dict with per_question metrics and aggregate metrics.
    """

    per_question: list[dict[str, Any]] = []
    metric_buckets: dict[str, list[float]] = {
        "exact_match": [],
        "accuracy": [],
        "f1": [],
        "rouge1_f1": [],
        "rouge2_f1": [],
        "rougeL_f1": [],
        "bleu": [],
    }

    judge_rows: list[dict[str, Any]] = []

    for item in predictions:
        question = str(item.get("question", "")).strip()
        prediction = str(item.get("answer", "")).strip()
        references = references_by_question.get(question, [])

        if not question:
            continue

        if not references:
            per_question.append(
                {
                    "question": question,
                    "prediction": prediction,
                    "references": [],
                    "missing_reference": True,
                }
            )
            continue

        em = exact_match_score(prediction, references)
        acc = accuracy_score(prediction, references)
        f1 = token_f1_score(prediction, references)
        r1 = _rouge_n_f1(prediction, references, n=1)
        r2 = _rouge_n_f1(prediction, references, n=2)
        rl = rouge_l_f1_score(prediction, references)
        bleu = bleu_score(prediction, references, max_n=4)

        metric_buckets["exact_match"].append(em)
        metric_buckets["accuracy"].append(acc)
        metric_buckets["f1"].append(f1)
        metric_buckets["rouge1_f1"].append(r1)
        metric_buckets["rouge2_f1"].append(r2)
        metric_buckets["rougeL_f1"].append(rl)
        metric_buckets["bleu"].append(bleu)

        row = {
            "question": question,
            "prediction": prediction,
            "references": references,
            "exact_match": em,
            "accuracy": acc,
            "f1": f1,
            "rouge1_f1": r1,
            "rouge2_f1": r2,
            "rougeL_f1": rl,
            "bleu": bleu,
            "missing_reference": False,
        }
        per_question.append(row)
        judge_rows.append(row)

    aggregate = {
        name: (sum(values) / len(values) if values else 0.0)
        for name, values in metric_buckets.items()
    }

    judge_summary: dict[str, Any] = {
        "enabled": bool(judge_model_name),
        "model": judge_model_name,
    }

    if judge_model_name:
        try:
            judge_items = _run_judge_model_scoring(
                rows=judge_rows,
                model_name=judge_model_name,
                max_new_tokens=judge_max_new_tokens,
                device=judge_device,
            )
            judge_by_question = {item["question"]: item for item in judge_items}
            judge_scores: list[float] = []
            for row in per_question:
                q = row.get("question", "")
                if q in judge_by_question:
                    row["judge_score"] = judge_by_question[q]["judge_score"]
                    row["judge_reason"] = judge_by_question[q]["judge_reason"]
                    judge_scores.append(float(judge_by_question[q]["judge_score"]))

            judge_summary.update(
                {
                    "status": "ok",
                    "mean_judge_score": (sum(judge_scores) / len(judge_scores) if judge_scores else 0.0),
                    "count": len(judge_scores),
                }
            )
        except Exception as exc:
            judge_summary.update(
                {
                    "status": "failed",
                    "error": str(exc),
                }
            )
    else:
        judge_summary["status"] = "disabled"

    return {
        "aggregate": aggregate,
        "judge": judge_summary,
        "coverage": {
            "predictions_total": len(predictions),
            "evaluated_with_references": len(judge_rows),
            "missing_reference": len(predictions) - len(judge_rows),
        },
        "per_question": per_question,
    }
