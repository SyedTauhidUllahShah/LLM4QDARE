"""
Robust extraction of categorical labels from raw LLM output.

LLMs may produce noisy responses such as:
  - "Catalog"                          (perfect)
  - "catalog"                          (wrong case)
  - "The label is: Catalog."           (embedded label)
  - "Catalog - this requirement …"     (label with explanation)
  - "I would assign 'Catalog' here."   (quoted label)
  - "CATALOG"                          (all caps)
  - "Notifications"                    (plural / morphological variant)

The extractor tries progressively looser matching and falls back to
fuzzy matching against the known label vocabulary.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional

logger = logging.getLogger(__name__)


class LabelExtractor:
    """
    Extracts a canonical label from raw model output.

    Parameters
    ----------
    known_labels : list of str
        The exhaustive set of valid labels for the dataset
        (e.g. ["Catalog", "Loan", "Notification", …]).
    fuzzy_threshold : float
        Minimum similarity ratio for fuzzy matching (0–1). Default 0.80.
    """

    def __init__(
        self,
        known_labels: List[str],
        fuzzy_threshold: float = 0.80,
    ) -> None:
        self.known_labels = known_labels
        self.fuzzy_threshold = fuzzy_threshold

        # Pre-build case-insensitive lookup: lower → canonical
        self._lower_map: dict = {lbl.lower(): lbl for lbl in known_labels}
        # Also map label without non-alphanumeric chars
        self._norm_map: dict = {
            re.sub(r"[^a-z0-9]", "", lbl.lower()): lbl for lbl in known_labels
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, raw_output: str) -> Optional[str]:
        """
        Parse ``raw_output`` and return the best matching canonical label,
        or ``None`` if no label could be reliably identified.

        Strategy (in order):
        1. Exact match (case-insensitive) of the entire cleaned string.
        2. Scan for a known label as a whole word within the response.
        3. Extract candidate tokens and match each against the vocabulary.
        4. Fuzzy match the cleaned first word/phrase against the vocabulary.
        5. Return ``None`` (unknown).
        """
        if not raw_output or not raw_output.strip():
            return None

        cleaned = self._clean(raw_output)

        # 1. Whole-string exact match
        result = self._exact_match(cleaned)
        if result:
            return result

        # 2. Scan full response for embedded label
        result = self._scan_for_label(cleaned)
        if result:
            return result

        # 3. Try each token (word) extracted from the response
        result = self._token_match(cleaned)
        if result:
            return result

        # 4. Fuzzy match
        result = self._fuzzy_match(cleaned)
        if result:
            logger.debug("Fuzzy match '%s' → '%s'", cleaned, result)
            return result

        logger.debug("No label extracted from: %r", raw_output[:120])
        return None

    def extract_batch(self, raw_outputs: List[str]) -> List[Optional[str]]:
        """Extract labels from a list of raw model outputs."""
        return [self.extract(r) for r in raw_outputs]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Remove common surrounding punctuation and whitespace."""
        text = text.strip()
        # Remove common wrapper characters
        text = text.strip("\"'`.,;:!?\n\r\t")
        # Collapse internal whitespace
        text = re.sub(r"\s+", " ", text)
        return text

    def _exact_match(self, text: str) -> Optional[str]:
        """Match if the entire text (case-insensitive) is a known label."""
        key = re.sub(r"[^a-z0-9]", "", text.lower())
        return self._norm_map.get(key) or self._lower_map.get(text.lower())

    def _scan_for_label(self, text: str) -> Optional[str]:
        """
        Scan for any known label appearing as a whole word (case-insensitive)
        in the response, preferring longer matches.
        """
        found = []
        for lbl in self.known_labels:
            pattern = r"\b" + re.escape(lbl) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                found.append(lbl)

        if not found:
            return None

        # Prefer the longest match (most specific)
        return max(found, key=len)

    def _token_match(self, text: str) -> Optional[str]:
        """
        Split response into tokens and try to match each token exactly
        against the vocabulary. Return the first successful match.
        """
        # Extract candidate tokens: words that start with a capital letter
        # (typical label format) or any word if none found
        tokens = re.findall(r"[A-Z][A-Za-z0-9]*|[a-zA-Z0-9]+", text)

        for token in tokens:
            key_lower = token.lower()
            key_norm = re.sub(r"[^a-z0-9]", "", key_lower)
            match = self._lower_map.get(key_lower) or self._norm_map.get(key_norm)
            if match:
                return match

        return None

    def _fuzzy_match(self, text: str) -> Optional[str]:
        """
        Compute SequenceMatcher ratio between the first ~30 chars of the
        response and each known label.  Return the best match above threshold.
        """
        candidate = text[:30].lower()
        best_label: Optional[str] = None
        best_ratio = 0.0

        for lbl in self.known_labels:
            ratio = SequenceMatcher(None, candidate, lbl.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = lbl

        if best_ratio >= self.fuzzy_threshold:
            return best_label

        return None
