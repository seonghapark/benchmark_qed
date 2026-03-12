import asyncio
import json
import re
from typing import Any

import torch
import torch.nn.functional as F
from json_repair import repair_json
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline

from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.llm.type.base import BaseModelOutput, BaseModelResponse, Usage


class HuggingFaceLocalChat:
    """Local chat provider backed by a Hugging Face causal LM."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._model_name = llm_config.model
        self._semaphore = asyncio.Semaphore(max(1, llm_config.concurrent_requests))
        self._usage = Usage(model=self._model_name)

        init_args = dict(llm_config.init_args)
        trust_remote_code = bool(init_args.pop("trust_remote_code", True))
        requested_device = str(init_args.pop("device", "auto")).lower()
        if requested_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = requested_device

        torch_dtype = init_args.pop("torch_dtype", None)
        if isinstance(torch_dtype, str):
            if torch_dtype.lower() in {"float16", "fp16"}:
                torch_dtype = torch.float16
            elif torch_dtype.lower() in {"bfloat16", "bf16"}:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(self._model_name, **model_kwargs)

        if self._device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            device_arg: int = 0
        else:
            model = model.to("cpu")
            device_arg = -1

        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            device=device_arg,
        )

    def get_usage(self) -> dict[str, Any]:
        return self._usage.model_dump()

    def _extract_questions_from_text(self, content: str) -> list[str]:
        lines = [ln.strip(" -\t") for ln in content.splitlines() if ln.strip()]
        candidates = [ln for ln in lines if "?" in ln]
        if not candidates:
            sentences = re.split(r"(?<=[?.!])\s+", content)
            candidates = [s.strip() for s in sentences if s.strip().endswith("?")]
        deduped: list[str] = []
        seen: set[str] = set()
        for c in candidates:
            key = c.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped

    def _coerce_json_output(
        self, content: str, messages: list[dict[str, str]]
    ) -> str:
        wants_json = any("--OUTPUT--" in m.get("content", "") for m in messages)
        if not wants_json:
            return content

        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        expects_expanded_questions = "output_question" in last_user

        parsed: Any = None
        try:
            repaired = repair_json(content, return_objects=True)
            parsed = repaired
        except Exception:
            parsed = None

        if expects_expanded_questions:
            questions_payload: list[dict[str, Any]] = []
            if isinstance(parsed, dict):
                raw_qs = parsed.get("questions", [])
                if isinstance(raw_qs, list):
                    for item in raw_qs:
                        if isinstance(item, dict):
                            out_q = item.get("output_question") or item.get("question") or ""
                            if out_q:
                                normalized = {
                                    "input_question": item.get("input_question", out_q),
                                    "period": item.get("period", ""),
                                    "location": item.get("location", ""),
                                    "named_entities": item.get("named_entities", ""),
                                    "abstract_categories": item.get("abstract_categories", ""),
                                    "background_information": item.get(
                                        "background_information", ""
                                    ),
                                    "output_question": out_q,
                                }
                                questions_payload.append(normalized)
                        elif isinstance(item, str):
                            questions_payload.append(
                                {
                                    "input_question": item,
                                    "period": "",
                                    "location": "",
                                    "named_entities": "",
                                    "abstract_categories": "",
                                    "background_information": "",
                                    "output_question": item,
                                }
                            )
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        out_q = item.get("output_question") or item.get("question") or ""
                        if out_q:
                            questions_payload.append(
                                {
                                    "input_question": item.get("input_question", out_q),
                                    "period": item.get("period", ""),
                                    "location": item.get("location", ""),
                                    "named_entities": item.get("named_entities", ""),
                                    "abstract_categories": item.get("abstract_categories", ""),
                                    "background_information": item.get(
                                        "background_information", ""
                                    ),
                                    "output_question": out_q,
                                }
                            )
                    elif isinstance(item, str):
                        questions_payload.append(
                            {
                                "input_question": item,
                                "period": "",
                                "location": "",
                                "named_entities": "",
                                "abstract_categories": "",
                                "background_information": "",
                                "output_question": item,
                            }
                        )

            if not questions_payload:
                for q in self._extract_questions_from_text(content):
                    questions_payload.append(
                        {
                            "input_question": q,
                            "period": "",
                            "location": "",
                            "named_entities": "",
                            "abstract_categories": "",
                            "background_information": "",
                            "output_question": q,
                        }
                    )

            return json.dumps({"questions": questions_payload}, ensure_ascii=True)

        if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
            return json.dumps(parsed, ensure_ascii=True)

        extracted = self._extract_questions_from_text(content)
        return json.dumps(
            {
                "background_information": "",
                "questions": extracted,
            },
            ensure_ascii=True,
        )

    async def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> BaseModelResponse:
        call_kwargs = dict(kwargs)
        temperature = float(call_kwargs.pop("temperature", 0.0) or 0.0)
        max_new_tokens = int(call_kwargs.pop("max_new_tokens", call_kwargs.pop("max_tokens", 256)))

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            ) + "\nassistant:"

        def _generate() -> str:
            outputs = self._generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                return_full_text=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            text = outputs[0].get("generated_text", "")
            return text.strip()

        async with self._semaphore:
            content = await asyncio.to_thread(_generate)

        content = self._coerce_json_output(content, messages)

        prompt_tokens = len(self._tokenizer.encode(prompt, add_special_tokens=False))
        completion_tokens = len(self._tokenizer.encode(content, add_special_tokens=False))
        self._usage.add_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        history = [
            *messages,
            {"role": "assistant", "content": content},
        ]
        return BaseModelResponse(
            output=BaseModelOutput(content=content),
            history=history,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )


class HuggingFaceLocalEmbedding:
    """Local embedding provider backed by Hugging Face encoder models."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._model_name = llm_config.model
        self._usage = Usage(model=self._model_name)
        self._semaphore = asyncio.Semaphore(max(1, llm_config.concurrent_requests))

        init_args = dict(llm_config.init_args)
        requested_device = str(init_args.pop("device", "auto")).lower()
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested_device

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

    def get_usage(self) -> dict[str, Any]:
        return self._usage.model_dump()

    async def embed(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        batch_size = int(kwargs.get("batch_size", 64))
        normalize_embeddings = bool(kwargs.get("normalize_embeddings", False))

        def _encode() -> list[list[float]]:
            all_embeddings: list[list[float]] = []
            for start in range(0, len(text_list), batch_size):
                batch = text_list[start : start + batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    masked = token_embeddings * attention_mask
                    summed = masked.sum(dim=1)
                    counts = attention_mask.sum(dim=1).clamp(min=1)
                    embeddings = summed / counts
                    if normalize_embeddings:
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().tolist())
            return all_embeddings

        async with self._semaphore:
            embeddings = await asyncio.to_thread(_encode)

        estimated_tokens = sum(max(1, len(t.split())) for t in text_list)
        self._usage.add_usage(prompt_tokens=estimated_tokens)
        return embeddings
