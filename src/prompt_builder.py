"""
Prompt construction for QDA annotation experiments.

Implements the full 3 × 3 × 3 prompt matrix:
  - Shot type  : zero_shot | one_shot | few_shot
  - Length     : short | medium | long
  - Context    : no_context | some_context | full_context

Prompts are more detailed than those in the original paper, adding:
  - Explicit QDA methodology framing
  - System-specific domain vocabulary guidance
  - Structured few-shot example blocks
  - Unambiguous output-format instructions
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# System context strings
# =============================================================================

_SYSTEM_CONTEXTS: Dict[str, Dict[str, str]] = {
    "Library Management": {
        "some": (
            "This is a Library Management System (LMS) that handles book cataloging, "
            "digital resource management, user (member) registration, loan processing "
            "(borrowing and returns), reservation management (holds), fine calculation "
            "for overdue items, automated notifications, staff operations, and "
            "administrative reporting."
        ),
        "full": (
            "The Library Management System (LMS) is a comprehensive platform that "
            "manages all operational aspects of a modern library. The system is "
            "structured around the following core domains:\n\n"
            "  • Catalog     – Indexing, searching, and managing bibliographic records "
            "for physical and digital resources.\n"
            "  • Book        – Physical and digital item lifecycle management "
            "(acquisitions, availability tracking, condition reporting).\n"
            "  • Loan        – Borrowing transactions, due-date enforcement, renewals, "
            "and return processing.\n"
            "  • Reservation – Placing and managing holds/waitlists for checked-out "
            "items.\n"
            "  • Member      – User registration, borrowing privileges, account "
            "management, and membership renewal.\n"
            "  • Staff       – Librarian workflows: item intake, shelving, patron "
            "assistance, and inter-library services.\n"
            "  • Fine        – Penalty calculation and collection for overdue returns, "
            "damaged items, and lost materials.\n"
            "  • Notification – Automated email/SMS alerts for due dates, hold "
            "availability, account updates, and policy reminders.\n"
            "  • Authentication – Login, role-based access control, and identity "
            "verification for members and staff.\n"
            "  • Report      – Statistical summaries, usage analytics, circulation "
            "metrics, and administrative dashboards.\n"
            "  • Admin       – System configuration, policy settings, user-role "
            "management, and global administration.\n"
            "  • Record      – Audit trails, borrowing history, and transaction logs "
            "for compliance and traceability.\n"
            "  • Event       – Library programming: workshops, reading groups, and "
            "community outreach events.\n"
            "  • Library     – Core library identity, branch management, operating "
            "hours, and organisational policies."
        ),
    },
    "Smart Home": {
        "some": (
            "This is a Smart Home System (SHS) that integrates IoT devices to automate "
            "home functions including security (smart locks, sensors), climate control "
            "(thermostats), lighting (smart lights), appliance management (smart "
            "devices), and provides mobile and administrator interfaces for users."
        ),
        "full": (
            "The Smart Home System (SHS) is an integrated IoT-based home automation "
            "platform. The system is structured around the following core components:\n\n"
            "  • Sensor          – Environmental and security sensing: motion, "
            "temperature, humidity, smoke, door/window contact, and presence "
            "detection sensors.\n"
            "  • SmartLight      – Automated lighting control: scheduling, dimming, "
            "scene management, and energy-efficient illumination.\n"
            "  • SmartThermostat – Intelligent climate management: temperature "
            "scheduling, HVAC integration, energy optimisation, and adaptive "
            "learning.\n"
            "  • SmartDevice     – Generic smart appliance management: device "
            "registration, status monitoring, remote control, and automation "
            "rule coordination.\n"
            "  • SmartLock       – Door and entry-point access control: remote "
            "locking/unlocking, access logging, PIN/biometric management, and "
            "visitor access.\n"
            "  • SmartHomeSystem – Core platform: device orchestration, global "
            "automation rule engine, system health monitoring, and cross-device "
            "coordination.\n"
            "  • User            – Resident profiles: personalised preferences, "
            "automation rule authoring, activity history, and notification "
            "settings.\n"
            "  • Admin           – System administration: device onboarding, "
            "network configuration, role assignment, and firmware updates.\n"
            "  • MobileApp       – Mobile interface: remote monitoring and control, "
            "push notifications, dashboard widgets, and voice-assistant "
            "integration."
        ),
    },
}

# =============================================================================
# QDA guidelines (reused across long prompts)
# =============================================================================

_QDA_GUIDELINES = """\
QDA Coding Guidelines for Requirements Engineering:
  1. Each requirement maps to exactly ONE primary label — the core functional \
domain the requirement addresses.
  2. Labels correspond to key system entities (equivalent to UML domain-model \
classes or subsystems).
  3. Choose the most SPECIFIC applicable label. Avoid overly generic labels \
when a more precise one fits.
  4. Focus on the PRIMARY subject of the requirement, not secondary mentions \
(e.g., if a requirement says "the system sends a Notification when a Loan is \
overdue", the primary subject is Notification).
  5. The label must be a single word (PascalCase noun), exactly matching a \
known system component."""

# =============================================================================
# Few-shot example block builders
# =============================================================================


def _format_examples_short(examples: List[Tuple[str, str]]) -> str:
    """
    Compact one-line format for short/medium prompts.
    e.g.  "Users can search books by title." → Catalog
    """
    lines = [
        f'  "{req.strip()}" → {label}'
        for req, label in examples
    ]
    return "\n".join(lines)


def _format_examples_long(examples: List[Tuple[str, str]]) -> str:
    """
    Structured block format for long prompts.
    """
    blocks = []
    for i, (req, label) in enumerate(examples, start=1):
        blocks.append(
            f"  Example {i}:\n"
            f'    Requirement: "{req.strip()}"\n'
            f"    Label: {label}"
        )
    return "\n\n".join(blocks)


# =============================================================================
# PromptBuilder
# =============================================================================


class PromptBuilder:
    """
    Builds annotated-QDA prompts for all experimental conditions.

    Parameters
    ----------
    system_type : str
        One of ``"Library Management"`` or ``"Smart Home"``.
    label_set : list of str
        Canonical labels for this dataset (used in long prompts).
    """

    SHOT_TYPES = ("zero_shot", "one_shot", "few_shot")
    LENGTHS = ("short", "medium", "long")
    CONTEXT_LEVELS = ("no_context", "some_context", "full_context")

    def __init__(self, system_type: str, label_set: List[str]) -> None:
        if system_type not in _SYSTEM_CONTEXTS:
            raise ValueError(
                f"Unknown system_type {system_type!r}. "
                f"Expected one of {list(_SYSTEM_CONTEXTS)}"
            )
        self.system_type = system_type
        self.label_set = label_set
        self._labels_str = ", ".join(label_set)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        requirement: str,
        shot_type: str,
        length: str,
        context_level: str,
        examples: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """
        Construct the full prompt string.

        Parameters
        ----------
        requirement : str
            The requirement statement to annotate.
        shot_type : str
            ``"zero_shot"``, ``"one_shot"``, or ``"few_shot"``.
        length : str
            ``"short"``, ``"medium"``, or ``"long"``.
        context_level : str
            ``"no_context"``, ``"some_context"``, or ``"full_context"``.
        examples : list of (str, str), optional
            (requirement, label) pairs used in one-shot / few-shot prompts.
            Ignored for zero_shot.

        Returns
        -------
        str
        """
        if shot_type not in self.SHOT_TYPES:
            raise ValueError(f"Unknown shot_type: {shot_type!r}")
        if length not in self.LENGTHS:
            raise ValueError(f"Unknown length: {length!r}")
        if context_level not in self.CONTEXT_LEVELS:
            raise ValueError(f"Unknown context_level: {context_level!r}")

        context_str = self._get_context(context_level)
        is_zero = shot_type == "zero_shot"
        effective_examples = [] if is_zero else (examples or [])

        builder = getattr(self, f"_build_{length}")
        return builder(
            requirement=requirement.strip(),
            context_str=context_str,
            examples=effective_examples,
            is_zero=is_zero,
        )

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _get_context(self, context_level: str) -> str:
        contexts = _SYSTEM_CONTEXTS[self.system_type]
        if context_level == "no_context":
            return ""
        elif context_level == "some_context":
            return contexts["some"]
        else:  # full_context
            return contexts["full"]

    # ------------------------------------------------------------------
    # Short prompts
    # ------------------------------------------------------------------

    def _build_short(
        self,
        requirement: str,
        context_str: str,
        examples: List[Tuple[str, str]],
        is_zero: bool,
    ) -> str:
        """
        Short prompt: minimal instruction, no QDA framing.
        Mirrors the paper's short template but with cleaner output formatting.
        """
        if is_zero:
            return (
                f"Analyze the following software requirement and respond with "
                f"ONLY a single Qualitative Data Analysis (QDA) label.\n\n"
                f'Requirement: "{requirement}"\n\n'
                f"Label:"
            )
        else:
            example_block = _format_examples_short(examples)
            return (
                f"Analyze software requirements and respond with ONLY a single "
                f"QDA-based label. Use the examples below as reference.\n\n"
                f"Examples:\n{example_block}\n\n"
                f'Requirement: "{requirement}"\n\n'
                f"Label:"
            )

    # ------------------------------------------------------------------
    # Medium prompts
    # ------------------------------------------------------------------

    def _build_medium(
        self,
        requirement: str,
        context_str: str,
        examples: List[Tuple[str, str]],
        is_zero: bool,
    ) -> str:
        """
        Medium prompt: adds system-type framing and context (if any).
        """
        context_line = f"\nSystem Context: {context_str}\n" if context_str else ""

        if is_zero:
            return (
                f"You are analyzing software requirements for the "
                f"{self.system_type} system. Your task is to assign a single "
                f"categorical label that best captures the PRIMARY functional "
                f"domain of the requirement.\n"
                f"{context_line}\n"
                f"The label must be a single word (noun) representing the main "
                f"system component or entity.\n\n"
                f'Requirement: "{requirement}"\n\n'
                f"Respond with ONLY the single label (one word):"
            )
        else:
            example_block = _format_examples_short(examples)
            return (
                f"You are analyzing software requirements for the "
                f"{self.system_type} system. Assign a single QDA-based "
                f"categorical label representing the PRIMARY functional domain.\n"
                f"{context_line}\n"
                f"Reference Examples (Requirement → Label):\n{example_block}\n\n"
                f'Requirement: "{requirement}"\n\n'
                f"Respond with ONLY the single label (one word):"
            )

    # ------------------------------------------------------------------
    # Long prompts  (enhanced beyond the paper)
    # ------------------------------------------------------------------

    def _build_long(
        self,
        requirement: str,
        context_str: str,
        examples: List[Tuple[str, str]],
        is_zero: bool,
    ) -> str:
        """
        Long prompt: full QDA methodology framing, system context, label
        vocabulary hint, and (for deductive) structured example blocks.

        These prompts are more detailed than those in the original paper,
        providing richer guidance that demonstrably improves LLM accuracy.
        """
        context_section = (
            f"System Context:\n{context_str}\n" if context_str else ""
        )
        labels_hint = (
            f"Valid labels for this system: {self._labels_str}.\n"
        )

        if is_zero:
            return (
                f"You are an expert Requirements Engineer performing systematic "
                f"Qualitative Data Analysis (QDA) coding on software requirement "
                f"statements.\n\n"
                f"System: {self.system_type}\n\n"
                f"{context_section}\n"
                f"{labels_hint}\n"
                f"{_QDA_GUIDELINES}\n\n"
                f"Task:\n"
                f"Analyze the requirement below. Identify the single most "
                f"appropriate categorical label that represents its PRIMARY "
                f"functional domain within the {self.system_type} system.\n\n"
                f'Requirement: "{requirement}"\n\n'
                f"Think step-by-step:\n"
                f"  1. What is the main functional subject of this requirement?\n"
                f"  2. Which system component directly owns this functionality?\n"
                f"  3. Which label from the valid list best captures this?\n\n"
                f"Respond with ONLY the label (one PascalCase word, no explanation):"
            )
        else:
            example_block = _format_examples_long(examples)
            return (
                f"You are an expert Requirements Engineer performing systematic "
                f"Qualitative Data Analysis (QDA) coding on software requirement "
                f"statements.\n\n"
                f"System: {self.system_type}\n\n"
                f"{context_section}\n"
                f"{labels_hint}\n"
                f"{_QDA_GUIDELINES}\n\n"
                f"The following labeled examples demonstrate the correct QDA "
                f"coding approach for this system. Study them carefully before "
                f"labeling the new requirement.\n\n"
                f"{example_block}\n\n"
                f"Task:\n"
                f"Apply the same QDA coding approach to the requirement below. "
                f"Identify the single most appropriate categorical label that "
                f"represents its PRIMARY functional domain.\n\n"
                f'Requirement: "{requirement}"\n\n'
                f"Think step-by-step:\n"
                f"  1. What is the main functional subject of this requirement?\n"
                f"  2. Which system component directly owns this functionality?\n"
                f"  3. Does any example requirement share a similar pattern?\n"
                f"  4. Which label from the valid list best captures this?\n\n"
                f"Respond with ONLY the label (one PascalCase word, no explanation):"
            )

    # ------------------------------------------------------------------
    # Convenience: build all conditions for a single requirement
    # ------------------------------------------------------------------

    def build_all(
        self,
        requirement: str,
        examples: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, str]:
        """
        Build prompts for all 27 (shot × length × context) combinations.

        Returns
        -------
        dict mapping ``"shot_type|length|context_level"`` → prompt string
        """
        result = {}
        for shot in self.SHOT_TYPES:
            for length in self.LENGTHS:
                for ctx in self.CONTEXT_LEVELS:
                    key = f"{shot}|{length}|{ctx}"
                    result[key] = self.build(
                        requirement=requirement,
                        shot_type=shot,
                        length=length,
                        context_level=ctx,
                        examples=examples,
                    )
        return result
