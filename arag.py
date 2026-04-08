'''
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Index
   ↓
User Question                   User Question
   ↓                               ↓
Query Rewriting                 Query Rewriting
   ↓                               ↓
Vector Retrieval                Initial Retrieval
   ↓                               ↓
                                Retrieval Evaluation
                                   ↓
                                Bad Retrieval? ── YES → Query Expansion → Retrieve Again
                                   ↓ NO
Context Compression             Context Compression
   ↓                               ↓
Prompt                          Answer Generation
   ↓                               ↓
Qwen2.5-0.5B-Instruct           Answer Grounding Check
   ↓                               ↓
                                Ungrounded? ── YES → Retrieve More Evidence
                                   ↓ NO
Answer                          Final Answer


What makes this "production-grade"
The system now includes:
Query rewriting --> improve retrieval
Retrieval evaluation --> detect bad context
Query expansion --> fix retrieval
Context compression --> reduce noise
Grounding verification --> prevent hallucinations

Production improvements (recommended)
1 Hybrid retrieval: Combine dense + keyword search using --> Elasticsearch / Pyserini
2 Add reranking: Use --> bge-reranker-base / This usually improves RAG accuracy 20–40%.
3 Retrieval evaluation: Evaluate your RAG with --> RAGAS / DeepEval
'''


import faiss, torch, re, pickle, os, sys, types, importlib
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import logging
logging.set_verbosity_warning()

os.environ.setdefault("TRANSFORMERS_NO_APEX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TRAINER", "1")

if "apex" not in sys.modules:
    apex_stub = types.ModuleType("apex")
    apex_stub.amp = object()
    apex_stub.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
    sys.modules["apex"] = apex_stub

from sentence_transformers import SentenceTransformer


# -------------------------
# Load models
# -------------------------

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_scores(raw_scores, higher_is_better=True):
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

# -------------------------
# Build FAISS index
# -------------------------

class VectorStore:

    def __init__(self, db_dir="/storage/araia_models1", nprobe=32, dense_weight=0.5, sparse_weight=0.5):
        self.db_dir = Path(db_dir)
        dense_path = self.db_dir / "dense_ivfpq.faiss"
        sparse_path = self.db_dir / "sparse_bm25.pkl"

        if not dense_path.exists() or not sparse_path.exists():
            raise FileNotFoundError(f"Missing index files in {self.db_dir}")

        self.index = faiss.read_index(str(dense_path))
        self.index.nprobe = nprobe

        with sparse_path.open("rb") as f:
            payload = pickle.load(f)

        self.token_corpus = payload.get("token_corpus", [])
        self.texts = payload.get("chunk_texts", [])
        self.metadata = payload.get("chunk_metadata", [])
        self.bm25 = BM25Okapi(self.token_corpus)
        self.chunk_count = min(int(self.index.ntotal), len(self.texts), len(self.metadata), len(self.token_corpus))

        self.dense_weight = max(0.0, dense_weight)
        self.sparse_weight = max(0.0, sparse_weight)
        if self.dense_weight + self.sparse_weight == 0.0:
            self.dense_weight = 0.5
            self.sparse_weight = 0.5

    def search(self, query, k=10):
        dense_k = max(k * 4, 20)
        sparse_k = max(k * 4, 20)

        query_emb = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        query_emb = np.asarray(query_emb, dtype=np.float32)

        dense_scores_arr, dense_ids_arr = self.index.search(query_emb, dense_k)
        dense_scores_arr = dense_scores_arr[0]
        dense_ids_arr = dense_ids_arr[0]

        sparse_scores_arr = self.bm25.get_scores(tokenize(query))
        sparse_ids = np.argsort(sparse_scores_arr)[::-1][:sparse_k]

        dense_raw = {}
        sparse_raw = {}

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
        candidates = sorted(set(dense_raw.keys()) | set(sparse_raw.keys()))
        rescored = []
        for idx in candidates:
            hybrid = (
                self.dense_weight * dense_norm.get(idx, 0.0)
                + self.sparse_weight * sparse_norm.get(idx, 0.0)
            ) / weight_sum

            rescored.append(
                {
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "scores": {
                        "dense_raw": dense_raw.get(idx),
                        "sparse_raw": sparse_raw.get(idx),
                        "hybrid": hybrid,
                    },
                }
            )

        rescored.sort(key=lambda x: x["scores"]["hybrid"], reverse=True)
        return rescored[:k]

# -------------------------
# RAG pipeline
# -------------------------
class RAG:
    def __init__(self, vectordb, embedder, generator, tokenizer):
        self.vectordb = vectordb
        self.embedder = embedder
        self.generator = generator
        self.tokenizer = tokenizer

    def ask(self, question):
        better_query = self._rewrite_query(question)
        answer = self.self_correcting_rag(better_query, self.vectordb)
        '''
        step 1 initial retrieval
        step 2 evaluate retrieval
        step 3 generate answer
        step 4 verify grounding
        '''
        return answer


    # -------------------------
    # Query rewriting
    # -------------------------

    def _rewrite_query(self, question):
        prompt = f"""Rewrite the user question to improve document retrieval.
                    Question:{question}
                    Search query:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )
        '''
        The top_p (nucleus sampling) parameter in self.generator.generate controls
        the diversity and coherence of text generation by limiting the model's
        token selection to a subset of the most likely words whose cumulative probability
        adds up to a specified value, P. It acts as a dynamic alternative to Top-K sampling,
        adapting the number of words considered based on the probability distribution at each step.
        Key Aspects of top_p: 
        Nucleus Sampling: The model selects the smallest set of top-ranked tokens
        whose cumulative probability exceeds the threshold P
        (e.g., P=0.9 considers tokens that make up 90% of the probability mass).
        Creativity vs. Focus: A lower top_p (e.g., 0.1-0.5) makes the output more deterministic
        and focused (narrower selection), while a higher top_p (e.g., 0.9-1.0) makes the output
        more creative and varied (broader selection).
        Default Value: It is typically a float between 0.0 and 1.0,
        often defaulting to 1.0 (sampling from the entire vocabulary).
        Usage: It is frequently used in combination with temperature to balance randomness
        and coherence in generated text. 
        top_p vs. Other Parameters:
        top_p vs top_k: top_p adapts the number of tokens based on probability,
        while top_k always selects a fixed number of tokens.
        top_p vs temperature: temperature reshapes the entire probability distribution
        (high temp flattens it), while top_p truncates the distribution by cutting off the
        "tail" of low-probability tokens
'''
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Original question: {question}")
        print(f"Rewritten query: {text}")
        return text.split("Search query:")[-1].strip()

    # -------------------------
    # Context compression
    # -------------------------

    def _compress_context(self, question, docs, max_sentences=6):
        sentences = []

        for d in docs:
            parts = re.split(r'(?<=[.!?]) +', d["text"])
            sentences.extend(parts)

        sentence_emb = self.embedder.encode(sentences)
        query_emb = self.embedder.encode([question])[0]
        scores = sentence_emb @ query_emb
        ranked = sorted(
            zip(sentences, scores),
            key=lambda x: x[1],
            reverse=True
        )
        selected = [s[0] for s in ranked[:max_sentences]]
        return "\n".join(selected)

    # -------------------------
    # Prompt builder
    # -------------------------

    def _build_prompt(self, question, context):

        return f"""You are a helpful assistant.
                    Answer ONLY using the context below.
                    Context: {context}
                    Question: {question}
                    Answer:"""

    # -------------------------
    # Generation
    # -------------------------

    def _generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            # do_sample=True,
            # temperature=0.3,
            # top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    # -------------------------
    # Self-correcting RAG extensions
    # -------------------------

    def _evaluate_retrieval(self, question, context):
        prompt = f"""Determine if the context is relevant enough to answer the question.
                    Question: {question}
                    Context: {context}
                    Answer YES or NO. """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "YES" in response.upper()

    def _expand_query(self, question):
        prompt = f"""Generate a better search query for retrieving documents.
                    Question: {question}
                    Improved search query: """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.split("Improved search query:")[-1].strip()

    def _verify_grounding(self, question, answer, context):
        prompt = f"""Is the answer supported by the context?
                    Question: {question}
                    Answer: {answer}
                    Context: {context}
                    Answer YES or NO. """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "YES" in text.upper()


    def self_correcting_rag(self, question, vectordb):
        # step 1 initial retrieval
        docs = vectordb.search(question, k=10)
        context = self._compress_context(question, docs)

        # step 2 evaluate retrieval
        if not self._evaluate_retrieval(question, context):
            improved_query = self._expand_query(question)
            docs = vectordb.search(improved_query, k=10)
            context = self._compress_context(question, docs)

        # step 3 generate answer
        prompt = self._build_prompt(question, context)
        answer = self._generate_answer(prompt)

        # step 4 verify grounding
        if not self._verify_grounding(question, answer, context):
            docs = vectordb.search(question, k=20)
            context = self._compress_context(question, docs)
            prompt = self._build_prompt(question, context)
            answer = self._generate_answer(prompt)

        return answer





# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    GEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    # EMBED_MODEL = "BAAI/bge-small-en-v1.5"

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    generator = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        device_map="cpu",
        torch_dtype=torch.float16
    )
    # Keep generation config consistent with greedy decoding to avoid noisy warnings.
    generator.generation_config.do_sample = False
    if hasattr(generator.generation_config, "temperature"):
        generator.generation_config.temperature = 1.0
    if hasattr(generator.generation_config, "top_p"):
        generator.generation_config.top_p = 1.0
    if hasattr(generator.generation_config, "top_k"):
        generator.generation_config.top_k = 50
    embedder = SentenceTransformer(EMBED_MODEL)
    vectordb = VectorStore(db_dir="/storage/araia_models1",
                            #    nprobe=64,
                            #    dense_weight=0.7,
                            #    sparse_weight=0.3,
                            )
    rag = RAG(vectordb, embedder, generator, tokenizer)
    question = "What factors influence the relationship between soil moisture and runoff?"
    response = rag.ask(question)
    print(response)