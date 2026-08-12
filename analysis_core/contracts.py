"""
analysis_core/contracts.py

Phase 0 deliverable — the single typed shape every deterministic check
in analysis_core will return. No LLM, no string formatting, no free
text: a Finding is a fact, computed once, by one named method, with
its unit explicit so percentage-vs-ratio bugs (see README §9.4 / §10)
can't happen at the source.
"""

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Finding:
    check_name: str
    # e.g. "shape", "missingness", "cardinality", "skewness",
    # "describe", "value_counts"

    column: Optional[str]
    # Column this fact is about. None for dataset-level facts (e.g. shape).

    value: Union[int, float, dict]
    # The computed number(s) or nested computed numbers. Never a string —
    # the narration layer formats it, it does not compute it.

    unit: Optional[str]
    # "%", "count", "ratio", "skew", None. Resolves the 75.92 vs 0.7592
    # class of bug: the unit travels with the number, so nothing gets
    # multiplied by 100 twice, or not at all, by accident in prose.

    n_rows_used: int
    # Row count the computation actually saw. Lets a reviewer (or a test)
    # sanity-check the Finding against the dataset without re-running it.

    method: str
    # e.g. "pandas.Series.skew()", "pandas.Series.nunique()". Exists so
    # nothing is ever hand-derived by narration — every Finding traces
    # to one concrete, checkable call.


# --- Phase 0 mapping: current report sections -> Finding shape ---
#
# This is the "one-page mapping" exit criterion. Each row below is not
# code — it's the decision Phase 1 will implement one function at a
# time. Fill in / adjust column lists to match your actual dataset
# before starting Phase 1.

PHASE_0_MAPPING = [
    {
        "report_section": "Dataset shape",
        "check_name": "shape",
        "column": None,
        "value_shape": {"rows": "int", "cols": "int"},
        "method": "df.shape",
    },
    {
        "report_section": "Missing-value summary",
        "check_name": "missingness",
        "column": "<per column>",
        "value_shape": {
            "non_null": "int",
            "missing": "int",
            "pct_missing": "float (0-100, unit='%')",
        },
        "method": "df[col].isna().sum() / len(df)",
    },
    {
        "report_section": "Cardinality",
        "check_name": "cardinality",
        "column": "<per column>",
        "value_shape": {"n_unique": "int", "pct_unique": "float (0-100, unit='%')"},
        "method": "df[col].nunique()",
    },
    {
        "report_section": "Skewness",
        "check_name": "skewness",
        "column": "<per numeric column>",
        "value_shape": "float (unit='skew')",
        "method": "df[col].skew()",
    },
    {
        "report_section": "Descriptive statistics",
        "check_name": "describe",
        "column": "<per numeric column>",
        "value_shape": {
            "mean": "float", "std": "float", "min": "float",
            "25%": "float", "50%": "float", "75%": "float", "max": "float",
        },
        "method": "df[numeric_cols].describe()",
    },
    {
        "report_section": "Class balance / target distribution",
        "check_name": "value_counts",
        "column": "<target column>",
        "value_shape": {
            "counts": "dict[str, int]",
            "proportions": "dict[str, float] (unit='ratio')",
        },
        "method": "df[col].value_counts() / df[col].value_counts(normalize=True)",
    },
    {
        "report_section": "Potential identifier columns",
        "check_name": "id_candidate",
        "column": "<per column>",
        "value_shape": "bool",
        "method": "derived rule: cardinality.pct_unique > 90 "
                  "(NOT a separate computation — reuses the 'cardinality' Finding)",
    },
]


if __name__ == "__main__":
    # Smoke check: print the mapping so it's easy to eyeball against
    # the actual report before Phase 1 starts writing functions.
    for row in PHASE_0_MAPPING:
        print(f"{row['report_section']:<30} -> check_name={row['check_name']}")