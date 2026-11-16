from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Arm = Literal["A", "B", "C", "D", "E", "F", "G"]
ARMS: tuple[Arm, ...] = ("A", "B", "C", "D", "E", "F", "G")


@dataclass(frozen=True)
class ArmPolicy:
    arm: Arm
    routing: bool
    assessment: bool
    verification: bool
    ordinary_decisions: bool = False


POLICIES = {
    "B": ArmPolicy("B", False, False, False),
    "C": ArmPolicy("C", False, True, False),
    "D": ArmPolicy("D", False, False, True),
    "E": ArmPolicy("E", False, True, True),
    "F": ArmPolicy("F", True, True, True),
    "G": ArmPolicy("G", True, True, True, ordinary_decisions=True),
}


def policy_for(arm: str) -> ArmPolicy:
    if arm not in POLICIES:
        raise ValueError("custom application arm must be B, C, D, E, F or G")
    return POLICIES[arm]
