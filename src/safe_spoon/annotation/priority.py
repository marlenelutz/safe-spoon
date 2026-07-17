"""Single scalar priority used to rank the annotation queue."""

from typing import List


def compute_unit_priority(
    unit: dict,
    pw_heterogeneity: float,
    pw_size: float,
    pw_balance: float,
) -> float:
    """Weighted combination of a unit's priority signals."""
    return round(
        pw_heterogeneity * unit["heterogeneity"]
        + pw_size * unit["size_norm"]
        + pw_balance * unit["merge_balance"],
        4,
    )


def assign_priorities(
    units: List[dict],
    pw_heterogeneity: float,
    pw_size: float,
    pw_balance: float,
) -> None:
    """Set priority and priority_rank on every unit dict, in place."""
    for u in units:
        u["priority"] = compute_unit_priority(u, pw_heterogeneity, pw_size, pw_balance)

    for rank, u in enumerate(sorted(units, key=lambda x: -x["priority"]), 1):
        u["priority_rank"] = rank
