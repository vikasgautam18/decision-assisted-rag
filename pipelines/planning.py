from __future__ import annotations

import re
from dataclasses import dataclass

from clients.decision import ChoiceResult
from ingestion.corpus import DOSSIERS

_HISTORY = re.compile(
    r"\b(?:histor(?:y|ical|ic)|supersed(?:ed|ing)|previous|earlier|older|old value|"
    r"prior revision|old revision|revision change|before the change|as it stood|"
    r"as of|effective on|applied on|on \d{1,2}\s+"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?))\b",
    re.IGNORECASE,
)
_VISUAL = re.compile(
    r"\b(?:show|display|image|picture|diagram|flowchart|callout|highlight|"
    r"original figure|figure\s+\d+\.\d+)\b",
    re.IGNORECASE,
)
_VISUAL_BRANCH = re.compile(
    r"\b(?:which|what)\s+(?:step|path|branch|outcome)\b|\bflow(?:chart)?\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|which (?:one|platform|variant) "
    r"(?:has|is|offers)|highest|lowest|most|least|across|all platforms)\b",
    re.IGNORECASE,
)
_PROCEDURE = re.compile(
    r"\b(?:how to|steps?|procedure|process|sequence|if .* then|"
    r"what happens when|which step)\b",
    re.IGNORECASE,
)
_CALCULATION = re.compile(
    r"\b(?:calculate|compute|total|sum|average|mean|percentage|percent|"
    r"ratio|difference|increase|decrease|convert|how many|count)\b",
    re.IGNORECASE,
)
_UNSUPPORTED = re.compile(
    r"\b(?:send (?:an? )?(?:email|message)|delete|edit (?:the )?(?:index|resource)|"
    r"execute (?:this )?command|run (?:this )?command|change (?:the )?deployment)\b",
    re.IGNORECASE,
)
_ALL_SCOPE = re.compile(
    r"\b(?:all (?:six )?(?:platforms|vehicles|models)|every platform|fleet[- ]wide|"
    r"across (?:all )?platforms|manufacturer[- ]wide|among the dossiers)\b",
    re.IGNORECASE,
)
_SPECIFICATION = re.compile(
    r"\b(?:power|torque|range|capacity|mass|weight|payload|towing|dimensions?|"
    r"engine[- ]only|system[- ]combined)\b",
    re.IGNORECASE,
)
_SECTION_RULES = (
    ("history", re.compile(r"\b(?:history|supersed|previous revision|revision change)\b", re.I)),
    ("overview", _SPECIFICATION),
    ("powertrain", re.compile(r"\b(?:engine|chassis|compression|injection)\b", re.I)),
    ("homologation", re.compile(r"\b(?:homologat|test cycle|cycle test)\b", re.I)),
    ("mounting", re.compile(r"\b(?:mount|callout|assembly|torque|install)\b", re.I)),
    ("suppliers", re.compile(r"\b(?:supplier|commercial terms|quality terms)\b", re.I)),
    ("warranty", re.compile(r"\b(?:warranty|claim|adjudicat|outcome code)\b", re.I)),
    ("thermal", re.compile(r"\b(?:thermal|temperature|depleted battery|derat)\b", re.I)),
    ("maintenance", re.compile(r"\b(?:maintenance|replacement interval|wear item)\b", re.I)),
)


@dataclass(frozen=True)
class DeterministicPlan:
    routing: dict[str, ChoiceResult]
    primary_section: str | None
    expansion_section: str
    require_figure: bool


def _explicit_scope(question: str) -> tuple[str, ...]:
    return tuple(
        platform
        for platform in DOSSIERS
        if re.search(rf"\b{re.escape(platform)}\b", question, re.IGNORECASE)
    )


def _latest_history_scope(history: list[str]) -> tuple[str, ...]:
    for previous in reversed(history[-4:]):
        scope = _explicit_scope(previous)
        if scope:
            return scope
        if _ALL_SCOPE.search(previous):
            return tuple(DOSSIERS)
    return ()


def deterministic_plan(question: str, history: list[str]) -> DeterministicPlan:
    """Apply one benchmark-agnostic, deterministic policy to B-E."""
    scope = _explicit_scope(question)
    if not scope and _ALL_SCOPE.search(question):
        scope = tuple(DOSSIERS)
    if not scope:
        scope = _latest_history_scope(history)

    history_intent = bool(_HISTORY.search(question))
    require_figure = bool(_VISUAL.search(question) or _VISUAL_BRANCH.search(question))
    if _UNSUPPORTED.search(question):
        intent = "unsupported"
    elif history_intent:
        intent = "history"
    elif require_figure:
        intent = "figure"
    elif _CALCULATION.search(question):
        intent = "calculation"
    elif _PROCEDURE.search(question):
        intent = "procedure"
    elif len(scope) > 1 or _COMPARISON.search(question):
        intent = "comparison"
    elif scope:
        intent = "lookup"
    else:
        intent = "clarify"

    temporal = "history" if history_intent else "current"
    primary_section = "overview" if _SPECIFICATION.search(question) else None
    expansion_section = next(
        (section for section, pattern in _SECTION_RULES if pattern.search(question)),
        "sufficient",
    )
    routing = {
        "intent": ChoiceResult(intent, None),
        "temporal_scope": ChoiceResult(temporal, None),
        "figure_requirement": ChoiceResult(
            "required" if require_figure else "not_required", None
        ),
        "comparison_basis": ChoiceResult(
            "declared_specifications" if primary_section else "other", None
        ),
        **{
            f"scope_{platform}": ChoiceResult(
                "include" if platform in scope else "exclude", None
            )
            for platform in DOSSIERS
        },
    }
    return DeterministicPlan(routing, primary_section, expansion_section, require_figure)
