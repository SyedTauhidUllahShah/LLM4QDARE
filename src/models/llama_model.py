"""
LLaMA-2-7B-Chat wrapper using HuggingFace Transformers.

Uses greedy decoding (do_sample=False) and the official LLaMA-2 chat
instruction template:

    <s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_message} [/INST]
"""

import logging
from typing import List, Optional

from .base_model import BaseModel

logger = logging.getLogger(__name__)

_LLAMA_SYSTEM_PROMPT = (
    "You are a precise and expert Requirements Engineering analyst. "
    "Follow the instructions exactly and respond with only the requested label."
)

_LLAMA_TEMPLATE = (
    "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
)


class LlamaModel(BaseModel):
    """
    LLaMA-2-7B-Chat (HuggingFace) for single-label QDA annotation.

    Requires a HuggingFace access token in the environment variable
    ``HUGGINGFACE_TOKEN`` (the LLaMA-2 model is gated).

    Parameters
    ----------
    model_cfg : dict
        Config block from ``config["models"]["llama2"]``.
    """

    def __init__(self, model_cfg: dict) -> None:
        super().__init__(model_cfg, model_name="LLaMA-2")
        self._pipeline = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        import os

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch accelerate"
            ) from exc

        token = os.getenv("HUGGINGFACE_TOKEN")
        model_id: str = self.model_cfg["name"]
        torch_dtype_str: str = self.model_cfg.get("torch_dtype", "float16")
        device_map: str = self.model_cfg.get("device_map", "auto")
        load_in_4bit: bool = bool(self.model_cfg.get("load_in_4bit", False))

        torch_dtype = (
            torch.float16 if torch_dtype_str == "float16" else torch.bfloat16
        )

        logger.info("Loading LLaMA-2 model: %s …", model_id)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            token=token,
        )
        model.eval()

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
        )
        logger.info("LLaMA-2 model loaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _format_prompt(self, instruction: str) -> str:
        return _LLAMA_TEMPLATE.format(
            system_prompt=_LLAMA_SYSTEM_PROMPT,
            instruction=instruction,
        )

    def predict(self, prompt: str) -> Optional[str]:
        self._load()
        return self._run_generation([prompt])[0]

    def predict_batch(self, prompts: List[str]) -> List[Optional[str]]:
        self._load()
        batch_size: int = int(self.model_cfg.get("batch_size", 8))
        results: List[Optional[str]] = []

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            results.extend(self._run_generation(batch))

        return results

    def _run_generation(self, prompts: List[str]) -> List[Optional[str]]:
        formatted = [self._format_prompt(p) for p in prompts]
        max_new_tokens: int = int(self.model_cfg.get("max_new_tokens", 50))

        try:
            outputs = self._pipeline(
                formatted,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._pipeline.tokenizer.pad_token_id,
                return_full_text=False,
            )
            results: List[Optional[str]] = []
            for out in outputs:
                if isinstance(out, list):
                    out = out[0]
                generated = out.get("generated_text", "")
                # Strip the [/INST] artefact if the model echoes it
                generated = generated.split("[/INST]")[-1]
                results.append(generated.strip() if generated else None)
            return results
        except Exception as exc:
            logger.error("LLaMA-2 generation error: %s", exc)
            return [None] * len(prompts)

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                import torch
                del self._pipeline.model
                del self._pipeline
                torch.cuda.empty_cache()
            except Exception:
                pass
            finally:
                self._pipeline = None
