"""
GPT-4 model wrapper using the OpenAI Chat Completions API.

Features:
- Temperature = 0 (greedy / deterministic decoding)
- Exponential back-off on rate-limit / transient errors
- Per-call timeout to prevent hanging
"""

import logging
import os
import time
from typing import Optional

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class GPT4Model(BaseModel):
    """
    Wraps the OpenAI GPT-4 Turbo API for single-label QDA annotation.

    Parameters
    ----------
    model_cfg : dict
        Config block from ``config["models"]["gpt4"]``.
    """

    def __init__(self, model_cfg: dict) -> None:
        super().__init__(model_cfg, model_name="GPT-4")
        self._client = None  # lazy init

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_client(self):
        """Initialise and cache the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for GPT-4. "
                    "Install with: pip install openai>=1.0.0"
                ) from exc

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Copy .env.example to .env and fill in your key."
                )
            self._client = OpenAI(api_key=api_key)
            logger.info(
                "OpenAI client initialised with model %s",
                self.model_cfg["name"],
            )
        return self._client

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, prompt: str) -> Optional[str]:
        """
        Send prompt to GPT-4 and return the raw text response.

        Retries up to ``model_cfg["max_retries"]`` times on transient errors
        using exponential back-off.
        """
        client = self._get_client()
        model_id: str = self.model_cfg["name"]
        temperature = self.model_cfg.get("temperature", None)
        max_tokens: int = int(self.model_cfg.get("max_tokens", 50))
        timeout: int = int(self.model_cfg.get("request_timeout", 30))
        max_retries: int = int(self.model_cfg.get("max_retries", 5))
        backoff_base: float = float(self.model_cfg.get("retry_backoff", 2.0))

        reasoning_effort = self.model_cfg.get("reasoning_effort", None)

        # Build API kwargs — omit temperature/reasoning_effort if null/None
        api_kwargs: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "timeout": timeout,
        }
        if temperature is not None:
            api_kwargs["temperature"] = float(temperature)
        if reasoning_effort is not None:
            api_kwargs["reasoning_effort"] = reasoning_effort

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**api_kwargs)
                raw = response.choices[0].message.content
                return raw.strip() if raw else None

            except Exception as exc:
                last_exc = exc
                exc_name = type(exc).__name__

                # Check if it is a rate-limit or server error
                is_rate_limit = "RateLimitError" in exc_name
                is_server_error = "APIStatusError" in exc_name or "500" in str(exc)
                is_timeout = "Timeout" in exc_name

                if not (is_rate_limit or is_server_error or is_timeout):
                    # Non-transient error — do not retry
                    logger.error("GPT-4 non-transient error: %s", exc)
                    return None

                wait = backoff_base ** attempt
                logger.warning(
                    "GPT-4 attempt %d/%d failed (%s). Retrying in %.1fs…",
                    attempt + 1,
                    max_retries,
                    exc_name,
                    wait,
                )
                time.sleep(wait)

        logger.error(
            "GPT-4 failed after %d retries. Last error: %s",
            max_retries,
            last_exc,
        )
        return None

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None:
            self._client.close()
            self._client = None
