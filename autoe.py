import argparse
from collections import defaultdict
import importlib.machinery
import json
import pickle
import re
import sys
import types
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

os = __import__("os")
os.environ.setdefault("TRANSFORMERS_NO_APEX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TRAINER", "1")

if "apex" not in sys.modules:
    apex_stub = types.ModuleType("apex")
    apex_stub.amp = object()
    apex_stub.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
    sys.modules["apex"] = apex_stub

from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    return out


def is_good_sentence(sentence: str) -> bool:
    if not sentence:
        return False
    s = sentence.strip()

    if len(s) < 40:
        return False
    if not s[0].isupper():
        return False
    if s.count(" ") < 6:
        return False
    if s.startswith(("of ", "and ", "but ", "because ")):
        return False

    return True


def select_diverse_sentences(
    question_vec: np.ndarray,
    sent_vecs: np.ndarray,
    sentences: list[str],
    top_k: int = 5,
) -> list[str]:
    scores = np.dot(sent_vecs, question_vec)
    order = np.argsort(scores)[::-1]

    selected: list[str] = []
    used_prefix: set[str] = set()

    for idx in order:
        sent = sentences[int(idx)].strip()
        if not is_good_sentence(sent):
            continue

        prefix = sent[:60].lower()
        if prefix in used_prefix:
            continue

        used_prefix.add(prefix)
        selected.append(sent)

        if len(selected) >= top_k:
            break

    return selected


def normalize_scores(raw_scores: dict[int, float], higher_is_better: bool) -> dict[int, float]:
    if not raw_scores:
        return {}
    values = np.asarray(list(raw_scores.values()), dtype=np.float32)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if abs(max_val - min_val) < 1e-12:
        return {idx: 1.0 for idx in raw_scores}
    if higher_is_better:
        return {idx: (score - min_val) / (max_val - min_val) for idx, score in raw_scores.items()}
    return {idx: (max_val - score) / (max_val - min_val) for idx, score in raw_scores.items()}


def load_questions(question_dir: Path, limit: int) -> list[str]:
    text_path = question_dir / "selected_questions_text.json"
    obj_path = question_dir / "selected_questions.json"
    candidate_path = question_dir / "candidate_questions.json"

    questions: list[str] = []
    if text_path.exists():
        data = json.loads(text_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    questions.append(item.strip())

    if not questions and obj_path.exists():
        data = json.loads(obj_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("question")
                if isinstance(text, str) and text.strip():
                    questions.append(text.strip())

    if candidate_path.exists():
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    questions.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("question")
                if isinstance(text, str) and text.strip():
                    questions.append(text.strip())

    if not questions:
        raise FileNotFoundError(
            f"No questions found in {question_dir}. Expected selected_questions_text.json, selected_questions.json, or candidate_questions.json"
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for question in questions:
        key = question.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(question)

    if limit > 0:
        deduped = deduped[:limit]
    return deduped


def resolve_index_dir(preferred: Path, fallback: Path) -> Path:
    preferred_dense = preferred / "dense_ivfpq.faiss"
    preferred_sparse = preferred / "sparse_bm25.pkl"
    if preferred_dense.exists() and preferred_sparse.exists():
        return preferred

    fallback_dense = fallback / "dense_ivfpq.faiss"
    fallback_sparse = fallback / "sparse_bm25.pkl"
    if fallback_dense.exists() and fallback_sparse.exists():
        return fallback

    raise FileNotFoundError(
        "Could not find dense_ivfpq.faiss and sparse_bm25.pkl in either "
        f"{preferred} or {fallback}"
    )


def build_title_to_text(dataset_dir: Path) -> dict[str, str]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    title_to_text: dict[str, str] = {}
    for json_path in sorted(dataset_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            continue

        fields = []
        for key in ("abstract", "text", "Introduction", "Methodology", "discussion", "conclusion"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                fields.append(value.strip())
        if fields:
            title_to_text[title.strip()] = "\n".join(fields)

    return title_to_text


class LocalAutoE:
    def __init__(
        self,
        index_dir: Path,
        embedding_model_name: str,
        answer_model_name: str,
        reranker_model_name: str,
        dense_top_k: int,
        sparse_top_k: int,
        final_top_k: int,
        nprobe: int,
        dense_weight: float,
        sparse_weight: float,
        use_reranker: bool,
    ) -> None:
        self.index = faiss.read_index(str(index_dir / "dense_ivfpq.faiss"))
        self.index.nprobe = nprobe

        payload = pickle.loads((index_dir / "sparse_bm25.pkl").read_bytes())
        self.token_corpus = payload["token_corpus"]
        self.chunk_texts = payload["chunk_texts"]
        self.chunk_metadata = payload["chunk_metadata"]
        self.bm25 = BM25Okapi(self.token_corpus)

        self.chunk_count = min(int(self.index.ntotal), len(self.token_corpus), len(self.chunk_texts), len(self.chunk_metadata))
        if self.chunk_count == 0:
            raise RuntimeError("Empty index/payload")

        # Retriever embedding model (requested): Qwen/Qwen3-Embedding-0.6B
        self.retriever_model = SentenceTransformer(embedding_model_name, device="cpu")
        # Local answer model (requested): sentence-transformers/all-MiniLM-L6-v2
        self.answer_ranker = SentenceTransformer(answer_model_name, device="cpu")
        self.reranker = CrossEncoder(reranker_model_name, device="cpu") if use_reranker else None

        self.dense_top_k = max(1, dense_top_k)
        self.sparse_top_k = max(1, sparse_top_k)
        self.final_top_k = max(1, final_top_k)
        self.dense_weight = max(0.0, dense_weight)
        self.sparse_weight = max(0.0, sparse_weight)
        if self.dense_weight + self.sparse_weight == 0.0:
            self.dense_weight = 0.5
            self.sparse_weight = 0.5

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        query_vec = self.retriever_model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vec = np.asarray(query_vec, dtype=np.float32)

        dense_scores_arr, dense_ids_arr = self.index.search(query_vec, self.dense_top_k)
        dense_scores_arr = dense_scores_arr[0]
        dense_ids_arr = dense_ids_arr[0]

        sparse_scores_arr = self.bm25.get_scores(tokenize(question))
        sparse_ids = np.argsort(sparse_scores_arr)[::-1][: self.sparse_top_k]

        dense_raw: dict[int, float] = {}
        sparse_raw: dict[int, float] = {}

        for rank, idx in enumerate(dense_ids_arr):
            idx = int(idx)
            if idx < 0 or idx >= self.chunk_count:
                continue
            dense_raw[idx] = float(dense_scores_arr[rank])

        for idx in sparse_ids:
            idx = int(idx)
            if idx < 0 or idx >= self.chunk_count:
                continue
            sparse_raw[idx] = float(sparse_scores_arr[idx])

        dense_norm = normalize_scores(dense_raw, higher_is_better=True)
        sparse_norm = normalize_scores(sparse_raw, higher_is_better=True)
        weight_sum = self.dense_weight + self.sparse_weight

        combined: list[dict[str, Any]] = []
        for idx in sorted(set(dense_raw.keys()) | set(sparse_raw.keys())):
            hybrid = (
                self.dense_weight * dense_norm.get(idx, 0.0)
                + self.sparse_weight * sparse_norm.get(idx, 0.0)
            ) / weight_sum
            combined.append(
                {
                    "chunk_idx": idx,
                    "text": self.chunk_texts[idx],
                    "metadata": self.chunk_metadata[idx],
                    "scores": {
                        "dense_raw": dense_raw.get(idx),
                        "sparse_raw": sparse_raw.get(idx),
                        "hybrid": hybrid,
                    },
                }
            )

        combined.sort(key=lambda x: x["scores"]["hybrid"], reverse=True)
        return combined[: self.final_top_k]

    def rerank_chunks(self, question: str, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not retrieved or self.reranker is None:
            return retrieved

        pairs = [(question, item.get("text", "")) for item in retrieved]
        scores = self.reranker.predict(pairs)

        for item, score in zip(retrieved, scores):
            item["rerank_score"] = float(score)

        retrieved.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
        return retrieved[: self.final_top_k]

    def select_evidence(self, question_vec: np.ndarray, sentences: list[str], top_k: int = 2) -> list[str]:
        if not sentences:
            return []
        sent_vecs = self.answer_ranker.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = np.dot(sent_vecs, question_vec)
        order = np.argsort(scores)[::-1]

        selected: list[str] = []
        seen: set[str] = set()
        for idx in order:
            sentence = sentences[int(idx)].strip()
            if not is_good_sentence(sentence):
                continue
            key = sentence[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append(sentence)
            if len(selected) >= top_k:
                break
        return selected

    def generate_structured_answer(self, evidence: list[str]) -> str:
        if not evidence:
            return "The retrieved documents do not contain sufficient information."

        answer = "The retrieved documents highlight several findings related to the question.\n\nKey points:\n"
        for sentence in evidence[:5]:
            line = sentence.strip()
            if not line.endswith("."):
                line += "."
            answer += f"- {line}\n"
        return answer.strip()

    def answer(self, question: str, grouped_evidence: dict[str, list[str]]) -> str:
        if not grouped_evidence:
            return "The retrieved documents do not provide enough information to answer the question."

        q_vec = self.answer_ranker.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        final_evidence: list[str] = []
        for _, sentences in grouped_evidence.items():
            if not sentences:
                continue
            best = self.select_evidence(q_vec, sentences, top_k=2)
            final_evidence.extend(best)

        # Final diversity pass across all selected evidence.
        if final_evidence:
            sent_vecs = self.answer_ranker.encode(
                final_evidence,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            final_evidence = select_diverse_sentences(
                q_vec,
                sent_vecs,
                final_evidence,
                top_k=5,
            )

        if not final_evidence:
            return "The retrieved context does not clearly answer the question."

        return self.generate_structured_answer(final_evidence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local AutoE-style QA pairs from existing vector DB")
    parser.add_argument("--dataset-dir", type=str, default="/storage/data/new_samples")
    parser.add_argument("--question-dir", type=str, default="/storage/qed/data_local_questions")
    parser.add_argument("--index-dir", type=str, default="/storage/hybrid_100_newdata")
    parser.add_argument("--fallback-index-dir", type=str, default="/storage/hybrid_test_100_newdata")
    parser.add_argument("--num-pairs", type=int, default=10)
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
    parser.add_argument("--output", type=str, default="/storage/qed/autoe_qa_pairs.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    question_dir = Path(args.question_dir)
    index_dir = resolve_index_dir(Path(args.index_dir), Path(args.fallback_index_dir))

    questions = load_questions(question_dir, limit=args.num_pairs)
    if len(questions) < args.num_pairs:
        raise RuntimeError(
            f"Need {args.num_pairs} questions, but only found {len(questions)} in {question_dir}"
        )

    title_to_text = build_title_to_text(dataset_dir)

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

    qa_pairs: list[dict[str, Any]] = []
    for pair_id, question in enumerate(questions[: args.num_pairs], start=1):
        retrieved = autoe.retrieve(question)
        retrieved = autoe.rerank_chunks(question, retrieved)

        grouped_evidence: dict[str, list[str]] = defaultdict(list)
        evidence_titles: list[str] = []
        for item in retrieved:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            title = metadata.get("title") if isinstance(metadata, dict) else None
            if not isinstance(title, str) or not title:
                title = "unknown"
            if isinstance(title, str) and title and title not in evidence_titles:
                evidence_titles.append(title)

            if isinstance(title, str) and title in title_to_text:
                grouped_evidence[title].extend(split_sentences(title_to_text[title])[:8])
            text = item.get("text", "") if isinstance(item, dict) else ""
            if isinstance(text, str) and text.strip():
                grouped_evidence[title].extend(split_sentences(text)[:8])

        # Deduplicate and cap per-title evidence pool for answer ranking speed.
        compact_grouped_evidence: dict[str, list[str]] = {}
        for title, sentences in grouped_evidence.items():
            deduped_sentences: list[str] = []
            seen_sentences: set[str] = set()
            for sentence in sentences:
                key = sentence.strip().lower()
                if key and key not in seen_sentences:
                    seen_sentences.add(key)
                    deduped_sentences.append(sentence)
            if deduped_sentences:
                compact_grouped_evidence[title] = deduped_sentences[:80]

        answer = autoe.answer(question, compact_grouped_evidence)
        qa_pairs.append(
            {
                "id": pair_id,
                "question": question,
                "answer": answer,
                "source_titles": evidence_titles[:5],
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "dataset_dir": str(dataset_dir),
            "question_dir": str(question_dir),
            "index_dir": str(index_dir),
            "embedding_model": args.embedding_model,
            "answer_model": args.answer_model,
            "num_pairs": args.num_pairs,
        },
        "qa_pairs": qa_pairs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(qa_pairs)} question-answer pairs to {output_path}")


if __name__ == "__main__":
    main()
