from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import LLM_BACKEND, LLM_MAX_NEW_TOKENS, LLM_MODEL, LLM_TEMPERATURE


@dataclass(frozen=True)
class LLMResult:
    text: str
    raw: Any | None = None


class LocalHuggingFaceLLM:
    """
    Free, local LLM using Hugging Face transformers.

    Default model is a small instruction-tuned seq2seq model (FLAN-T5).
    """

    def __init__(self, model_name: str = LLM_MODEL, max_new_tokens: int = LLM_MAX_NEW_TOKENS) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._is_encoder_decoder: bool | None = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        cfg = AutoConfig.from_pretrained(self.model_name)
        self._is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self._is_encoder_decoder:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def generate(self, prompt: str) -> LLMResult:
        self._ensure_loaded()
        tok = self._tokenizer
        model = self._model

        # Defensive cap: some tokenizers set model_max_length to very large sentinel values
        max_in = getattr(tok, "model_max_length", 512) or 512
        if max_in and max_in > 4096:
            max_in = 512

        import torch

        with torch.no_grad():
            if self._is_encoder_decoder:
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in)
                gen_kwargs = {}
                if LLM_TEMPERATURE > 0:
                    gen_kwargs.update({"do_sample": True, "temperature": LLM_TEMPERATURE})
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    **gen_kwargs,
                )
                text = tok.decode(outputs[0], skip_special_tokens=True) if len(outputs) else ""
                return LLMResult(text=text.strip(), raw=None)

            # Causal LM path
            # Use chat template when available (improves instruction following)
            if hasattr(tok, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are a scientific assistant. Use only provided context."},
                    {"role": "user", "content": prompt},
                ]
                enc = tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if hasattr(enc, "shape"):
                    prompt_ids = enc
                    attention_mask = None
                else:
                    prompt_ids = enc["input_ids"]
                    attention_mask = enc.get("attention_mask")
                input_len = int(prompt_ids.shape[-1])
                gen_kwargs = {}
                if LLM_TEMPERATURE > 0:
                    gen_kwargs.update({"do_sample": True, "temperature": LLM_TEMPERATURE})
                outputs = model.generate(
                    prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    **gen_kwargs,
                )
                gen_ids = outputs[0][input_len:] if outputs is not None else []
                text = tok.decode(gen_ids, skip_special_tokens=True) if len(gen_ids) else ""
                return LLMResult(text=text.strip(), raw=None)

            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in)
            input_len = int(inputs["input_ids"].shape[-1])
            gen_kwargs = {}
            if LLM_TEMPERATURE > 0:
                gen_kwargs.update({"do_sample": True, "temperature": LLM_TEMPERATURE})
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                **gen_kwargs,
            )
            gen_ids = outputs[0][input_len:] if outputs is not None else []
            text = tok.decode(gen_ids, skip_special_tokens=True) if len(gen_ids) else ""
            return LLMResult(text=text.strip(), raw=None)


_DEFAULT_LLM: object | None = None


def get_llm():
    global _DEFAULT_LLM
    if _DEFAULT_LLM is None:
        if LLM_BACKEND.lower() == "ollama":
            from .ollama_llm import OllamaLLM

            _DEFAULT_LLM = OllamaLLM()
        else:
            _DEFAULT_LLM = LocalHuggingFaceLLM()
    return _DEFAULT_LLM

