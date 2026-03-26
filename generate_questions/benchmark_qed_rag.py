import argparse, time, json, pickle, re, sys, types
import importlib.machinery
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import torch
import torch.nn.functional as F

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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def tokenize(text: str) -> list[str]:
	return re.findall(r"[a-z0-9]+", text.lower())


def normalize_scores(raw_scores: dict[int, float]) -> dict[int, float]:
	if not raw_scores:
		return {}
	values = np.asarray(list(raw_scores.values()), dtype=np.float32)
	min_val = float(np.min(values))
	max_val = float(np.max(values))
	if abs(max_val - min_val) < 1e-12:
		return {idx: 1.0 for idx in raw_scores}
	return {idx: (score - min_val) / (max_val - min_val) for idx, score in raw_scores.items()}


def softmax_from_scores(raw_scores: dict[int, float], temperature: float = 1.0) -> dict[int, float]:
	if not raw_scores:
		return {}
	temp = max(1e-6, float(temperature))
	idxs = list(raw_scores.keys())
	vals = np.asarray([raw_scores[i] for i in idxs], dtype=np.float64)
	vals = vals / temp
	vals = vals - np.max(vals)
	exp_vals = np.exp(vals)
	den = float(np.sum(exp_vals))
	if den <= 0:
		return {i: 1.0 / len(idxs) for i in idxs}
	probs = exp_vals / den
	return {i: float(p) for i, p in zip(idxs, probs)}


class FisherScorer:
	"""Compute Fisher-like document relevance scores from a local causal LM."""

	def __init__(self, model_name: str, cache_dir: str | None = None, device: str = "cpu") -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
		self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
		self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
		self.model = self.model.to(self.device)
		self.model.eval()

	def _last_token_logits(self, text: str) -> torch.Tensor:
		inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		with torch.no_grad():
			outputs = self.model(**inputs)
			logits = outputs.logits[:, -1, :]
		return logits

	def _entropy(self, logits: torch.Tensor) -> float:
		probs = F.softmax(logits, dim=-1)
		h = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
		return float(h.item())

	def entropy_reduction_scores(self, query: str, docs: dict[int, str]) -> dict[int, float]:
		if not docs:
			return {}
		h_q = self._entropy(self._last_token_logits(query))
		scores: dict[int, float] = {}
		for idx, doc in docs.items():
			ctx = f"Question: {query}\n\nDocument: {doc}"
			h_qd = self._entropy(self._last_token_logits(ctx))
			scores[idx] = h_q - h_qd
		return scores

	def gradient_norm_scores(self, query: str, docs: dict[int, str]) -> dict[int, float]:
		if not docs:
			return {}
		scores: dict[int, float] = {}
		for idx, doc in docs.items():
			text = f"Question: {query}\n\nDocument: {doc}"
			inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
			inputs = {k: v.to(self.device) for k, v in inputs.items()}
			self.model.zero_grad(set_to_none=True)
			outputs = self.model(**inputs)
			logits = outputs.logits[:, -1, :]
			probs = F.softmax(logits, dim=-1)
			loss = -torch.log(torch.max(probs) + 1e-9)
			loss.backward()

			grad_sq_sum = 0.0
			for p in self.model.parameters():
				if p.grad is not None:
					grad_sq_sum += float((p.grad.detach() ** 2).sum().item())
			scores[idx] = grad_sq_sum
		return scores


def dedupe_keep_order(items: list[str]) -> list[str]:
	out: list[str] = []
	seen: set[str] = set()
	for item in items:
		key = item.strip().lower()
		if key and key not in seen:
			seen.add(key)
			out.append(item.strip())
	return out


def read_questions_jsonl(path: Path) -> list[str]:
	questions: list[str] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			if isinstance(obj, dict):
				text = obj.get("question") or obj.get("text")
				if isinstance(text, str) and text.strip():
					questions.append(text.strip())
			elif isinstance(obj, str) and obj.strip():
				questions.append(obj.strip())
	return questions


def read_questions_json(path: Path) -> list[str]:
	data = json.loads(path.read_text(encoding="utf-8"))
	out: list[str] = []
	if isinstance(data, list):
		for item in data:
			if isinstance(item, str) and item.strip():
				out.append(item.strip())
			elif isinstance(item, dict):
				text = item.get("question") or item.get("text")
				if isinstance(text, str) and text.strip():
					out.append(text.strip())
	elif isinstance(data, dict):
		maybe_list = data.get("questions")
		if isinstance(maybe_list, list):
			for item in maybe_list:
				if isinstance(item, str) and item.strip():
					out.append(item.strip())
				elif isinstance(item, dict):
					text = item.get("question") or item.get("text") or item.get("output_question")
					if isinstance(text, str) and text.strip():
						out.append(text.strip())
	return out


def load_queries(args: argparse.Namespace) -> list[str]:
	queries: list[str] = []

	if args.query:
		queries.extend(args.query)

	if args.selected_questions_file:
		p = Path(args.selected_questions_file)
		if p.exists():
			data = json.loads(p.read_text(encoding="utf-8"))
			if isinstance(data, list):
				for item in data:
					if isinstance(item, str) and item.strip():
						queries.append(item.strip())

	if args.selected_questions_json_file:
		p = Path(args.selected_questions_json_file)
		if p.exists():
			data = json.loads(p.read_text(encoding="utf-8"))
			if isinstance(data, list):
				for item in data:
					if isinstance(item, dict):
						text = item.get("text") or item.get("question")
						if isinstance(text, str) and text.strip():
							queries.append(text.strip())

	if args.queries_file:
		p = Path(args.queries_file)
		if p.exists():
			if p.suffix == ".jsonl":
				queries.extend(read_questions_jsonl(p))
			elif p.suffix == ".json":
				queries.extend(read_questions_json(p))
			else:
				lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
				queries.extend([ln for ln in lines if ln])

	if not queries:
		p1 = Path("/storage/qed/output/data_local_questions/selected_questions.json")
		p2 = Path("/storage/qed/output/data_local_questions/selected_questions_text.json")
		p3 = Path("/storage/questions.jsonl")
		if p1.exists():
			args.selected_questions_json_file = str(p1)
			return load_queries(args)
		if p2.exists():
			args.selected_questions_file = str(p2)
			return load_queries(args)
		if p3.exists():
			args.queries_file = str(p3)
			return load_queries(args)

	queries = dedupe_keep_order(queries)
	if args.limit > 0:
		queries = queries[: args.limit]
	return queries


class LocalRAG:
	def __init__(
		self,
		db_dir: Path,
		embedding_model: str,
		reranker_model: str,
		dense_top_k: int,
		sparse_top_k: int,
		final_top_k: int,
		nprobe: int,
		dense_weight: float,
		sparse_weight: float,
		fisher_modes: list[str],
		fisher_model: str,
		fisher_top_k: int,
		cache_dir: str | None,
	) -> None:
		self.index = faiss.read_index(str(db_dir / "dense_ivfpq.faiss"))
		self.index.nprobe = nprobe

		payload = pickle.loads((db_dir / "sparse_bm25.pkl").read_bytes())
		self.token_corpus = payload["token_corpus"]
		self.chunk_texts = payload["chunk_texts"]
		self.chunk_metadata = payload["chunk_metadata"]
		self.bm25 = BM25Okapi(self.token_corpus)
		self.fisher_key = None

		self.chunk_count = min(
			int(self.index.ntotal),
			len(self.token_corpus),
			len(self.chunk_texts),
			len(self.chunk_metadata),
		)
		if self.chunk_count <= 0:
			raise RuntimeError("Empty vector DB")

		self.embedder = SentenceTransformer(embedding_model, device="cpu")
		self.reranker = CrossEncoder(reranker_model, device="cpu")
		self.dense_top_k = max(1, dense_top_k)
		self.sparse_top_k = max(1, sparse_top_k)
		self.final_top_k = max(1, final_top_k)
		self.dense_weight = max(0.0, dense_weight)
		self.sparse_weight = max(0.0, sparse_weight)
		if self.dense_weight + self.sparse_weight == 0:
			self.dense_weight = 0.5
			self.sparse_weight = 0.5
		self.fisher_modes = [m for m in fisher_modes if m != "none"]
		self.fisher_top_k = max(1, fisher_top_k)
		self._fisher_scorers: dict[str, FisherScorer] = {}
		for mode in self.fisher_modes:
			if mode in {"entropy", "gradient"}:
				self._fisher_scorers[mode] = FisherScorer(
					model_name=fisher_model,
					cache_dir=cache_dir,
					device="cpu",
				)

	def _build_items(
		self,
		idx_order: list[int],
		dense_raw: dict[int, float],
		sparse_raw: dict[int, float],
		fisher_raw: dict[int, float],
		rerank_raw: dict[int, float],
		score_map: dict[int, float],
		score_name: str,
		k: int,
	) -> list[dict[str, Any]]:
		items: list[dict[str, Any]] = []
		for idx in idx_order[:k]:
			items.append(
				{
					"chunk_idx": idx,
					"text": self.chunk_texts[idx],
					"metadata": self.chunk_metadata[idx],
					"scores": {
						"dense_raw": dense_raw.get(idx),
						"sparse_raw": sparse_raw.get(idx),
						"fisher_raw": fisher_raw.get(idx),
						"rerank_raw": rerank_raw.get(idx),
						score_name: score_map.get(idx),
					},
				}
			)
		return items

	def _compute_rerank_scores(self, question: str, candidate_ids: list[int]) -> dict[int, float]:
		if self.reranker is None or not candidate_ids:
			return {}

		pairs = [(question, str(self.chunk_texts[idx])) for idx in candidate_ids]
		scores = self.reranker.predict(pairs)
		return {idx: float(score) for idx, score in zip(candidate_ids, scores)}

	def _maybe_rerank_order(
		self,
		idx_order: list[int],
		rank_scores: dict[int, float],
		score_map: dict[int, float],
	) -> list[int]:
		if not rank_scores:
			return idx_order
		return sorted(
			idx_order,
			key=lambda i: (rank_scores.get(i, -1e9), score_map.get(i, 0.0)),
			reverse=True,
		)

	def retrieve_tracks(self, question: str) -> dict[str, Any]:
		q = self.embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
		q = np.asarray(q, dtype=np.float32)

		d_scores, d_ids = self.index.search(q, self.dense_top_k)
		d_scores = d_scores[0]
		d_ids = d_ids[0]

		s_scores_all = self.bm25.get_scores(tokenize(question))
		s_ids = np.argsort(s_scores_all)[::-1][: self.sparse_top_k]

		dense_raw: dict[int, float] = {}
		sparse_raw: dict[int, float] = {}
		fisher_raw: dict[int, float] = {}
		fisher_raw_by_mode: dict[str, dict[int, float]] = {}

		for rank, idx in enumerate(d_ids):
			idx = int(idx)
			if 0 <= idx < self.chunk_count:
				dense_raw[idx] = float(d_scores[rank])

		for idx in s_ids:
			idx = int(idx)
			if 0 <= idx < self.chunk_count:
				sparse_raw[idx] = float(s_scores_all[idx])

		union_ids = sorted(set(dense_raw.keys()) | set(sparse_raw.keys()))
		if "embedding" in self.fisher_modes:
			# Approximation: I_d ~ (1 - p(d|q))^2 * ||e_d||^2, with ||e_d||~1 after normalization.
			probs = softmax_from_scores(dense_raw)
			mode_scores: dict[int, float] = {}
			for idx in union_ids:
				p = probs.get(idx, 0.0)
				mode_scores[idx] = float((1.0 - p) ** 2)
			fisher_raw_by_mode["embedding"] = mode_scores

		if any(m in self.fisher_modes for m in {"entropy", "gradient"}) and union_ids:
			# Compute expensive Fisher scores on a narrowed candidate set once, then evaluate enabled modes.
			dense_norm_pre = normalize_scores(dense_raw)
			sparse_norm_pre = normalize_scores(sparse_raw)
			dense_sparse_pre = {
				idx: 0.5 * dense_norm_pre.get(idx, 0.0) + 0.5 * sparse_norm_pre.get(idx, 0.0)
				for idx in union_ids
			}
			pool_ids = sorted(union_ids, key=lambda i: dense_sparse_pre.get(i, 0.0), reverse=True)[: self.fisher_top_k]
			docs = {idx: str(self.chunk_texts[idx]) for idx in pool_ids}
			if "entropy" in self.fisher_modes and "entropy" in self._fisher_scorers:
				fisher_raw_by_mode["entropy"] = self._fisher_scorers["entropy"].entropy_reduction_scores(question, docs)
			if "gradient" in self.fisher_modes and "gradient" in self._fisher_scorers:
				fisher_raw_by_mode["gradient"] = self._fisher_scorers["gradient"].gradient_norm_scores(question, docs)

		# Combine Fisher scores from all requested modes by averaging normalized per-mode scores.
		fisher_norm_by_mode: dict[str, dict[int, float]] = {
			mode: normalize_scores(scores)
			for mode, scores in fisher_raw_by_mode.items()
			if scores
		}
		if fisher_norm_by_mode:
			for idx in union_ids:
				vals = [mode_norm.get(idx, 0.0) for mode_norm in fisher_norm_by_mode.values()]
				if vals:
					fisher_raw[idx] = float(sum(vals) / len(vals))

		dense_norm = normalize_scores(dense_raw)
		sparse_norm = normalize_scores(sparse_raw)
		fisher_norm = normalize_scores(fisher_raw) if fisher_raw else {}

		dense_sparse_50: dict[int, float] = {}
		for idx in union_ids:
			dense_sparse_50[idx] = 0.5 * dense_norm.get(idx, 0.0) + 0.5 * sparse_norm.get(idx, 0.0)

		weight_sum = self.dense_weight + self.sparse_weight
		dense_sparse_weighted: dict[int, float] = {}
		for idx in union_ids:
			dense_sparse_weighted[idx] = (
				self.dense_weight * dense_norm.get(idx, 0.0)
				+ self.sparse_weight * sparse_norm.get(idx, 0.0)
			) / weight_sum

		semantic_100 = {idx: dense_norm.get(idx, 0.0) for idx in union_ids}
		dense_100 = semantic_100.copy()
		sparse_100 = {idx: sparse_norm.get(idx, 0.0) for idx in union_ids}

		semantic_fisher_50: dict[int, float] = {}
		semantic_fisher_weighted: dict[int, float] = {}
		fisher_100: dict[int, float] = {}
		if fisher_norm:
			semantic_weight = float(getattr(self, "semantic_weight", 0.5))
			fisher_weight = float(getattr(self, "fisher_weight", 0.5))
			weight_sum_sf = semantic_weight + fisher_weight
			if weight_sum_sf <= 0:
				semantic_weight = 0.5
				fisher_weight = 0.5
				weight_sum_sf = 1.0
			for idx in union_ids:
				fisher_100[idx] = fisher_norm.get(idx, 0.0)
				semantic_fisher_50[idx] = 0.5 * dense_norm.get(idx, 0.0) + 0.5 * fisher_norm.get(idx, 0.0)
				semantic_fisher_weighted[idx] = (
					semantic_weight * dense_norm.get(idx, 0.0)
					+ fisher_weight * fisher_norm.get(idx, 0.0)
				) / weight_sum_sf

		order_semantic_100 = sorted(union_ids, key=lambda i: semantic_100.get(i, 0.0), reverse=True)
		order_fisher_100 = sorted(union_ids, key=lambda i: fisher_100.get(i, 0.0), reverse=True) if fisher_100 else []
		order_semantic_fisher_50 = sorted(union_ids, key=lambda i: semantic_fisher_50.get(i, 0.0), reverse=True) if semantic_fisher_50 else []
		order_semantic_fisher_weighted = (
			sorted(union_ids, key=lambda i: semantic_fisher_weighted.get(i, 0.0), reverse=True)
			if semantic_fisher_weighted
			else []
		)
		order_dense_sparse_50 = sorted(union_ids, key=lambda i: dense_sparse_50.get(i, 0.0), reverse=True)
		order_dense_sparse_weighted = sorted(union_ids, key=lambda i: dense_sparse_weighted.get(i, 0.0), reverse=True)
		order_dense_100 = sorted(union_ids, key=lambda i: dense_100.get(i, 0.0), reverse=True)
		order_sparse_100 = sorted(union_ids, key=lambda i: sparse_100.get(i, 0.0), reverse=True)

		rerank_raw = self._compute_rerank_scores(question, union_ids)
		order_semantic_100 = self._maybe_rerank_order(order_semantic_100, rerank_raw, semantic_100)
		order_fisher_100 = self._maybe_rerank_order(order_fisher_100, rerank_raw, fisher_100)
		order_semantic_fisher_50 = self._maybe_rerank_order(order_semantic_fisher_50, rerank_raw, semantic_fisher_50)
		order_semantic_fisher_weighted = self._maybe_rerank_order(order_semantic_fisher_weighted, rerank_raw, semantic_fisher_weighted)
		order_dense_sparse_50 = self._maybe_rerank_order(order_dense_sparse_50, rerank_raw, dense_sparse_50)
		order_dense_sparse_weighted = self._maybe_rerank_order(order_dense_sparse_weighted, rerank_raw, dense_sparse_weighted)
		order_dense_100 = self._maybe_rerank_order(order_dense_100, rerank_raw, dense_100)
		order_sparse_100 = self._maybe_rerank_order(order_sparse_100, rerank_raw, sparse_100)

		tracks = {
			"semantic_100": self._build_items(order_semantic_100, dense_raw, sparse_raw, fisher_raw, rerank_raw, semantic_100, "semantic_100", self.final_top_k),
			"fisher_100": self._build_items(order_fisher_100, dense_raw, sparse_raw, fisher_raw, rerank_raw, fisher_100, "fisher_100", self.final_top_k),
			"semantic_fisher_50_50": self._build_items(order_semantic_fisher_50, dense_raw, sparse_raw, fisher_raw, rerank_raw, semantic_fisher_50, "semantic_fisher_50_50", self.final_top_k),
			"semantic_fisher_weighted": self._build_items(order_semantic_fisher_weighted, dense_raw, sparse_raw, fisher_raw, rerank_raw, semantic_fisher_weighted, "semantic_fisher_weighted", self.final_top_k),
			"dense_sparse_50_50": self._build_items(order_dense_sparse_50, dense_raw, sparse_raw, fisher_raw, rerank_raw, dense_sparse_50, "dense_sparse_50_50", self.final_top_k),
			"dense_sparse_weighted": self._build_items(order_dense_sparse_weighted, dense_raw, sparse_raw, fisher_raw, rerank_raw, dense_sparse_weighted, "dense_sparse_weighted", self.final_top_k),
			"dense_100": self._build_items(order_dense_100, dense_raw, sparse_raw, fisher_raw, rerank_raw, dense_100, "dense_100", self.final_top_k),
			"sparse_100": self._build_items(order_sparse_100, dense_raw, sparse_raw, fisher_raw, rerank_raw, sparse_100, "sparse_100", self.final_top_k),
		}

		return {
			"fisher_available": bool(fisher_100),
			"fisher_key": self.fisher_key,
			"fisher_modes": self.fisher_modes,
			"fisher_mode_scores": fisher_norm_by_mode,
			"candidate_count": len(union_ids),
			"tracks": tracks,
		}

	def retrieve(self, question: str) -> list[dict[str, Any]]:
		# Backward-compatible single track for generation context.
		return self.retrieve_tracks(question)["tracks"]["dense_sparse_50_50"]


class LocalGenerator:
	def __init__(self, model_name: str, max_new_tokens: int, cache_dir: str | None = None) -> None:
		tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
		model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
		self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
		self.max_new_tokens = max(16, int(max_new_tokens))

	def answer(self, question: str, retrieved: list[dict[str, Any]]) -> str:
		if not retrieved:
			return ""
		context_blocks: list[str] = []
		for i, item in enumerate(retrieved, start=1):
			title = ""
			md = item.get("metadata", {})
			if isinstance(md, dict):
				title = str(md.get("title") or "")
			snippet = str(item.get("text", "")).strip()
			if title:
				context_blocks.append(f"[{i}] {title}\n{snippet}")
			else:
				context_blocks.append(f"[{i}] {snippet}")

		prompt = (
			"Answer the question using only the provided context. "
			"If context is insufficient, say so briefly.\n\n"
			f"Question: {question}\n\n"
			"Context:\n"
			+ "\n\n".join(context_blocks)
			+ "\n\nAnswer:"
		)

		out = self.pipe(
			prompt,
			max_new_tokens=self.max_new_tokens,
			do_sample=True,
			return_full_text=False,
			pad_token_id=self.pipe.tokenizer.eos_token_id,
		)
		text = out[0].get("generated_text", "")
		return text.strip()


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Local benchmark_qed_rag runner")
	p.add_argument("--db-dir", type=str, default="/storage/hybrid_test_100_newdata")
	p.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
	p.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-base")
	p.add_argument("--dense-top-k", type=int, default=20)
	p.add_argument("--sparse-top-k", type=int, default=20)
	p.add_argument("--final-top-k", type=int, default=5)
	p.add_argument("--dense-weight", type=float, default=0.5)
	p.add_argument("--sparse-weight", type=float, default=0.5)
	p.add_argument("--semantic-weight", type=float, default=0.5)
	p.add_argument("--fisher-weight", type=float, default=0.5)
	p.add_argument(
		"--fisher-mode",
		type=str,
		default=None,
		choices=["none", "embedding", "entropy", "gradient"],
		help="Deprecated single Fisher mode. Use --fisher-modes for one or many modes.",
	)
	p.add_argument(
		"--fisher-modes",
		type=str,
		nargs="+",
		default=["embedding"],
		choices=["none", "embedding", "entropy", "gradient"],
		help="One or more Fisher computation modes to run in the same execution.",
	)

	# Keep backward compatibility if the old single-mode arg is used by callers.
	p.add_argument(
		"--fisher-model",
		type=str,
		default="Qwen/Qwen2.5-1.5B-Instruct",
		help="Local model used when fisher-mode is entropy or gradient.",
	)
	p.add_argument(
		"--fisher-top-k",
		type=int,
		default=20,
		help="Top candidate count used for expensive Fisher computation modes (entropy/gradient).",
	)
	p.add_argument(
		"--final-retrieval-methods",
		type=str,
		nargs="+",
		default=["dense_sparse_50_50"],
		choices=[
			"dense_sparse_50_50",
			"dense_sparse_weighted",
			"dense_100",
			"sparse_100",
			"semantic_100",
			"fisher_100",
			"semantic_fisher_50_50",
			"semantic_fisher_weighted",
		],
		help=(
			"One or more methods used for final retrieval decision. Use dense_sparse_weighted "
			"with --dense-weight/--sparse-weight and semantic_fisher_weighted with "
			"--semantic-weight/--fisher-weight."
		),
	)
	p.add_argument("--nprobe", type=int, default=32)

	p.add_argument("--query", action="append", default=[])
	p.add_argument("--selected-questions-file", type=str)
	p.add_argument("--selected-questions-json-file", type=str)
	p.add_argument("--queries-file", type=str)
	p.add_argument("--limit", type=int, default=0)

	p.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
	p.add_argument("--max-new-tokens", type=int, default=128)
	p.add_argument("--cache-dir", type=str, default=None)

	p.add_argument("--output", type=str, required=True)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	fisher_modes = list(args.fisher_modes)
	if args.fisher_mode is not None:
		fisher_modes = [args.fisher_mode]
	# Deduplicate while preserving order.
	fisher_modes = list(dict.fromkeys(fisher_modes))
	if not fisher_modes:
		fisher_modes = ["none"]

	queries = load_queries(args)
	if not queries:
		raise RuntimeError("No queries found. Provide --query or one of the questions file options.")

	rag = LocalRAG(
		db_dir=Path(args.db_dir),
		embedding_model=args.embedding_model,
		reranker_model=args.reranker_model,
		dense_top_k=args.dense_top_k,
		sparse_top_k=args.sparse_top_k,
		final_top_k=args.final_top_k,
		nprobe=args.nprobe,
		dense_weight=args.dense_weight,
		sparse_weight=args.sparse_weight,
		fisher_modes=fisher_modes,
		fisher_model=args.fisher_model,
		fisher_top_k=args.fisher_top_k,
		cache_dir=args.cache_dir,
	)
	rag.semantic_weight = args.semantic_weight
	rag.fisher_weight = args.fisher_weight

	generator = LocalGenerator(
		model_name=args.llm_model,
		max_new_tokens=args.max_new_tokens,
		cache_dir=args.cache_dir,
	)

	results: list[dict[str, Any]] = []
	for question in queries:
		s = time.time()
		retrieval = rag.retrieve_tracks(question)
		print(f"Retrieved in {time.time() - s:.4f}s")
		selected_methods = list(dict.fromkeys(args.final_retrieval_methods))
		for method in selected_methods:
			if method in {"fisher_100", "semantic_fisher_50_50", "semantic_fisher_weighted"} and not retrieval["fisher_available"]:
				raise RuntimeError(
					f"Selected final retrieval method '{method}' requires Fisher scores, but no Fisher scores were computed. "
					f"Use --fisher-modes embedding|entropy|gradient."
				)

		primary_method = selected_methods[0]
		retrieved = retrieval["tracks"][primary_method]
		selected_retrievals = {method: retrieval["tracks"][method] for method in selected_methods}
		print(f"selected in {time.time() - s:.4f}s")
		answer = generator.answer(question, retrieved)
		results.append({
			"question": question,
			"retrieved": retrieved,
			"selected_retrievals": selected_retrievals,
			"final_retrieval_method": primary_method,
			"final_retrieval_methods": selected_methods,
			"fisher_modes": fisher_modes,
			"final_hybrid_weights": {
				"dense_weight": args.dense_weight,
				"sparse_weight": args.sparse_weight,
				"semantic_weight": args.semantic_weight,
				"fisher_weight": args.fisher_weight,
			},
			"retrieval_tracks": retrieval["tracks"],
			"fisher_available": retrieval["fisher_available"],
			"fisher_key": retrieval["fisher_key"],
			"answer": answer,
		})
		print(f"Answer generated in {time.time() - s:.4f}s")

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
	main()
