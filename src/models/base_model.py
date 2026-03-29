"""
Abstract base class for LLM wrappers used in QDA annotation experiments.
"""

import abc
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """
    Common interface for all LLM backends.

    Subclasses implement :meth:`predict` and optionally override
    :meth:`predict_batch` for efficient batched inference.

    Parameters
    ----------
    model_cfg : dict
        Model-specific configuration block from ``config.yaml``
        (e.g. ``config["models"]["gpt4"]``).
    model_name : str
        Human-readable name used in logs and result files.
    """

    def __init__(self, model_cfg: dict, model_name: str) -> None:
        self.model_cfg = model_cfg
        self.model_name = model_name
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def predict(self, prompt: str) -> Optional[str]:
        """
        Send a single prompt to the model and return the raw text response.

        Parameters
        ----------
        prompt : str
            The fully-rendered prompt string.

        Returns
        -------
        str or None
            Raw model output string, or ``None`` on failure.
        """

    # ------------------------------------------------------------------
    # Default batched inference (sequential fallback)
    # ------------------------------------------------------------------

    def predict_batch(self, prompts: List[str]) -> List[Optional[str]]:
        """
        Run inference on a list of prompts.

        Subclasses may override this for true batched inference.
        The default implementation calls :meth:`predict` sequentially.

        Parameters
        ----------
        prompts : list of str

        Returns
        -------
        list of (str or None)
        """
        return [self.predict(p) for p in prompts]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name!r})"

    def close(self) -> None:
        """Release any held resources (GPU memory, connections, etc.)."""
