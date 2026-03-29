"""
Mistral-7B-Instruct wrapper using HuggingFace Transformers.

Uses greedy decoding (do_sample=False) to match temperature=0 behaviour.
Applies the Mistral instruction template: <s>[INST] … [/INST]
"""

import logging
from typing import List, Optional

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Mistral-v0.2 instruction format
_MISTRAL_TEMPLATE = "<s>[INST] {instruction} [/INST]"


class MistralModel(BaseModel):
    """
    Mistral-7B-Instruct (HuggingFace) for single-label QDA annotation.

    Parameters
    ----------
    model_cfg : dict
        Config block from ``config["models"]["mistral"]``.
    """

    def __init__(self, model_cfg: dict) -> None:
        super().__init__(model_cfg, model_name="Mistral")
        self._pipeline = None   # lazy init

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the model and tokeniser (done once)."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for Mistral. "
                "Install with: pip install transformers torch accelerate"
            ) from exc

        model_id: str = self.model_cfg["name"]
        torch_dtype_str: str = self.model_cfg.get("torch_dtype", "float16")
        device_map: str = self.model_cfg.get("device_map", "auto")
        load_in_4bit: bool = bool(self.model_cfg.get("load_in_4bit", False))

        torch_dtype = (
            torch.float16 if torch_dtype_str == "float16" else torch.bfloat16
        )

        logger.info("Loading Mistral model: %s …", model_id)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
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
        )
        model.eval()

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
        )
        logger.info("Mistral model loaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _format_prompt(self, instruction: str) -> str:
        return _MISTRAL_TEMPLATE.format(instruction=instruction)

    def predict(self, prompt: str) -> Optional[str]:
        """Run a single prompt through Mistral and return the response text."""
        self._load()
        return self._run_generation([prompt])[0]

    def predict_batch(self, prompts: List[str]) -> List[Optional[str]]:
        """Batch inference for efficiency."""
        self._load()
        batch_size: int = int(self.model_cfg.get("batch_size", 8))
        results: List[Optional[str]] = []

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            results.extend(self._run_generation(batch))

        return results

    def _run_generation(self, prompts: List[str]) -> List[Optional[str]]:
        """
        Format prompts with the Mistral instruction template and run
        generation with greedy decoding.
        """
        formatted = [self._format_prompt(p) for p in prompts]
        max_new_tokens: int = int(self.model_cfg.get("max_new_tokens", 50))

        try:
            outputs = self._pipeline(
                formatted,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # greedy = temperature 0
                pad_token_id=self._pipeline.tokenizer.pad_token_id,
                return_full_text=False,    # return only the generated part
            )
            results: List[Optional[str]] = []
            for out in outputs:
                if isinstance(out, list):
                    out = out[0]
                generated = out.get("generated_text", "")
                results.append(generated.strip() if generated else None)
            return results
        except Exception as exc:
            logger.error("Mistral generation error: %s", exc)
            return [None] * len(prompts)

    def close(self) -> None:
        """Free GPU memory by deleting the pipeline and model."""
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
