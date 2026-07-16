"""Typed dataclasses mirroring the annotation.store schema rows."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RiskProfile:
    id: int
    category: str
    name: str
    description: str
    severity_rank: int
    created_at: str
    updated_at: str


@dataclass
class RubricCell:
    id: Optional[int]
    criterion_id: int
    risk_profile_id: int
    expected_behavior: str
    risk_signals: str
    inherited_from_cell_id: Optional[int]
    is_override: bool


@dataclass
class RubricCriterion:
    id: Optional[int]
    order_index: int
    title: str
    description: str
    cells: List[RubricCell] = field(default_factory=list)


@dataclass
class Rubric:
    id: Optional[int]
    category: str
    unit_stable_id: str
    annotator: str
    status: str  # 'draft' | 'submitted' | 'confirmed'
    source: str  # 'llm_suggestion' | 'manual'
    created_at: str
    updated_at: str
    criteria: List[RubricCriterion] = field(default_factory=list)


@dataclass
class RubricCandidate:
    """One of the LLM-proposed rubric tables, not yet persisted as a Rubric."""
    criteria: List[dict]  # [{"title": ..., "description": ..., "cells": {risk_profile_id: {...}}}]
