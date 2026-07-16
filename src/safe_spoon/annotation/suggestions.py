"""LLM-generated guidelines candidates for a unit."""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from safe_spoon.annotation.models import RiskProfile

log = logging.getLogger(__name__)


def _extract_json_array(raw: str) -> Optional[list]:
    """Best-effort extraction of a JSON array from an LLM response that may
    include surrounding prose or markdown code fences."""
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_rubric_candidates(
    prompter,
    category: str,
    repr_texts: List[str],
    dominant_topic: str,
    topic_keywords: str,
    risk_profiles: List[RiskProfile],
    n_candidates: int = 3,
    max_retries: int = 3,
    prompt_path: Optional[str] = None,
) -> List[list]:
    """Return up to n_candidates independently-generated rubric candidates.

    Each candidate is a list of criteria dicts in the shape expected by
    annotation.rubrics.submit_rubric(): [{"title", "description", "cells": {risk_profile_id: {...}}}].
    A candidate that fails to parse after max_retries is skipped rather than
    raising, so one bad LLM response doesn't block the other candidates.
    """
    if prompt_path is None:
        from safe_spoon.prompting import _default_prompt_path
        prompt_path = _default_prompt_path("rubric_suggestion_dft.txt")
    prompt_template = Path(str(prompt_path)).read_text(encoding="utf-8")

    docs_str = "\n- " + "\n- ".join(repr_texts) if repr_texts else "(none)"
    risk_profiles_str = "\n".join(
        f"{rp.id}: {rp.name} (severity {rp.severity_rank}) — {rp.description}"
        for rp in risk_profiles
    )
    filled = prompt_template.format(
        category=category,
        dominant_topic=dominant_topic,
        topic_keywords=topic_keywords,
        docs=docs_str,
        risk_profiles=risk_profiles_str,
    )

    valid_profile_ids = {rp.id for rp in risk_profiles}
    candidates: List[list] = []
    for candidate_idx in range(n_candidates):
        parsed = None
        for attempt in range(max_retries):
            temperature = 0.3 + attempt * 0.2
            # to obtian diverse candidates, we increase temperature on each retry
            raw, _ = prompter.prompt(
                question=filled,
                system_prompt_template_path=None,
                temperature=temperature,
                seed=1234 + candidate_idx * max_retries + attempt,
            )
            parsed = _extract_json_array(raw) if raw else None
            if parsed:
                break
        if not parsed:
            log.warning("Rubric suggestion: LLM returned no parseable JSON after %d attempts", max_retries)
            continue

        criteria = []
        for crit in parsed:
            if not isinstance(crit, dict) or "title" not in crit:
                continue
            cells = {
                pid: cell for pid, cell in (crit.get("cells") or {}).items()
                if str(pid).isdigit() and int(pid) in valid_profile_ids
            }
            criteria.append({
                "title": crit["title"],
                "description": crit.get("description", ""),
                "cells": {int(pid): cell for pid, cell in cells.items()},
            })
        if criteria:
            candidates.append(criteria)

    return candidates
