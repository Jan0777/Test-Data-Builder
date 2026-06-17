"""
Empirical-first Replicator profiler.
Learns the real marginal AND joint distributions from source data.
Guiding principle: do NOT fit textbook distributions — capture the data's own shape.
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from backend.spec.models import (
    CardinalitySpec, ColumnConstraints, ColumnSpec,
    GenerationSpec, IntraRowConstraint, RelationshipSpec, TableSpec,
)
from backend.replicator.semantics import detect_semantic

logger = logging.getLogger(__name__)

# Maximum rows used for expensive learning steps (correlation, content-sniffing)
_PROFILE_SAMPLE_LIMIT = 50_000
# _other_ pool cap — real tail values stored for fallback generation
_OTHER_POOL_CAP = 500
# Timestamp bootstrap pool cap
_TS_POOL_CAP = 2_000
# Cardinality threshold for "categorical" detection
_CAT_RATIO_THRESHOLD = 0.10
_CAT_UNIQUE_THRESHOLD = 50
# Minimum mutual-information-like effect size to declare a conditional group
_COND_GROUP_ETA_SQ_THRESHOLD = 0.05
# Min columns in the copula block
_MIN_COPULA_COLS = 2

_ALLOWED_SEMANTIC_TYPES = frozenset({
    "id", "name", "email", "address", "city", "state", "zip", "country",
    "currency", "category", "date", "phone", "url", "company", "none",
})


# ────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ────────────────────────────────────────────────────────────────────────────

def profile_to_spec(
    frames: Dict[str, pd.DataFrame],
    semantic_labels: Optional[Dict[str, Dict[str, Any]]] = None,
    row_count_override: Optional[int] = None,
) -> GenerationSpec:
    """
    Profile DataFrames into a GenerationSpec using empirical learning.

    Parameters
    ----------
    frames : table_name → DataFrame
    semantic_labels : optional external overrides {table → {col → {semantic_type, faker_method}}}
    row_count_override : if set, override the generated row count for every table
    """
    tables: List[TableSpec] = []
    relationships: List[RelationshipSpec] = []
    table_specs_map: Dict[str, TableSpec] = {}

    for table_name, df in frames.items():
        sem = (semantic_labels or {}).get(table_name, {})
        n_rows = row_count_override if row_count_override else len(df)
        logger.info("Profiling table '%s' (%d rows, %d cols)", table_name, len(df), len(df.columns))
        table_spec = _profile_table(table_name, df, n_rows, sem)
        tables.append(table_spec)
        table_specs_map[table_name] = table_spec

    if len(frames) > 1:
        inferred_rels = _infer_relationships(frames, table_specs_map)
        # Learn cross-table conditioning for each relationship
        for rel in inferred_rels:
            parent_df = frames.get(rel.parent)
            child_df = frames.get(rel.child)
            if parent_df is not None and child_df is not None:
                pc = _learn_parent_conditioning(parent_df, child_df, rel.parent_key, rel.child_key)
                if pc:
                    rel.parent_conditioning = pc
        relationships.extend(inferred_rels)

    return GenerationSpec(version="1.0", tables=tables, relationships=relationships)


# ────────────────────────────────────────────────────────────────────────────
# Table-level profiling
# ────────────────────────────────────────────────────────────────────────────

def _profile_table(
    name: str,
    df: pd.DataFrame,
    n_rows: int,
    semantic_labels: Dict[str, Any],
) -> TableSpec:
    columns: List[ColumnSpec] = []
    primary_key: Optional[str] = None

    # Identify FK columns ahead of time so we can skip empirical learning on them
    # (FKs are handled by child-table generation path)
    known_fk_cols: set = set()  # populated later in relationship detection

    for col_name in df.columns:
        col_spec, is_pk = _profile_column(
            col_name, df[col_name], len(df), semantic_labels.get(col_name, {}),
        )
        columns.append(col_spec)
        if is_pk and primary_key is None:
            primary_key = col_name

    constraints: List[IntraRowConstraint] = _infer_arithmetic_constraints(df)

    # Learn joint model (copula + conditional groups)
    col_specs_map = {c.name: c for c in columns}
    joint_model = _learn_joint_model(df, list(df.columns), col_specs_map, primary_key)

    # Update column strategies to copula_member for copula participants
    if joint_model:
        copula_cols = set(joint_model.get("columns", []))
        for col in columns:
            if col.name in copula_cols:
                # Move the existing empirical marginal params into copula_member format
                old_gen = col.generation
                if old_gen.get("strategy") == "empirical_sample":
                    col.generation = {
                        "strategy": "copula_member",
                        "params": {"marginal": old_gen.get("params", {})},
                    }
                    logger.info("  col '%s': strategy → copula_member", col.name)

    return TableSpec(
        name=name,
        row_count=n_rows,
        primary_key=primary_key,
        columns=columns,
        intra_row_constraints=constraints,
        joint_model=joint_model,
    )


# ────────────────────────────────────────────────────────────────────────────
# Column-level profiling
# ────────────────────────────────────────────────────────────────────────────

def _profile_column(
    name: str,
    series: pd.Series,
    n_rows: int,
    semantic_label: Dict[str, Any],
) -> Tuple[ColumnSpec, bool]:
    null_pct = float(series.isna().mean())
    clean = series.dropna()
    n_clean = len(clean)
    unique_count = int(series.nunique())
    is_unique = (unique_count == len(series)) and len(series) > 0
    is_pk = is_unique and null_pct == 0.0

    # External semantic overrides from the LLM pass
    ext_semantic_type = semantic_label.get("semantic_type", "none")
    ext_faker_method = semantic_label.get("faker_method")

    constraints_extra: Dict[str, Any] = {}
    col_type = "string"
    generation: Dict[str, Any] = {}
    semantic_type = "none"

    if n_clean == 0:
        # All-null column — emit a stub
        logger.info("  col '%s': all-null, emitting stub", name)
        col_type = "string"
        generation = {
            "strategy": "empirical_categorical",
            "params": {"values": {"": 1.0}, "preserve_order": False},
        }

    elif pd.api.types.is_bool_dtype(series):
        col_type = "boolean"
        true_rate = float(clean.astype(bool).mean())
        generation = {
            "strategy": "empirical_categorical",
            "params": {
                "values": {"true": round(true_rate, 6), "false": round(1.0 - true_rate, 6)},
                "preserve_order": False,
            },
        }
        logger.info("  col '%s': boolean, true_rate=%.3f", name, true_rate)

    elif pd.api.types.is_numeric_dtype(series):
        vals = clean.astype(float)
        is_integer = (
            pd.api.types.is_integer_dtype(series)
            or (len(vals) > 0 and bool(np.allclose(vals % 1, 0, atol=1e-6)))
        )
        col_type = "integer" if is_integer else "float"
        constraints_extra = {"min": float(vals.min()), "max": float(vals.max())}

        if is_pk and is_integer:
            start_val = int(vals.min()) if len(vals) > 0 else 1
            generation = {"strategy": "sequential", "params": {"start": start_val, "step": 1}}
            logger.info("  col '%s': PK → sequential (start=%d)", name, start_val)
        else:
            marginal = _learn_numeric_marginal(vals, is_integer)
            generation = {"strategy": "empirical_sample", "params": marginal}
            logger.info(
                "  col '%s': numeric empirical_sample (is_int=%s, zero_inf=%.3f)",
                name, is_integer, marginal.get("zero_inflation", 0.0),
            )

    else:
        # Try datetime first
        try:
            parsed = pd.to_datetime(clean, infer_datetime_format=True, errors="raise")
            col_type = "datetime"
            marginal = _learn_datetime_marginal(parsed)
            generation = {"strategy": "empirical_datetime", "params": marginal}
            semantic_type = "date"
            logger.info("  col '%s': datetime empirical", name)
        except Exception:
            # String column — content sniff → pattern → categorical → fallback
            col_type, generation, semantic_type = _profile_string_column(
                name, clean, unique_count, ext_semantic_type, ext_faker_method,
                is_pk=is_pk,
            )

    # Apply external semantic override (but not for numeric types — Faker produces strings)
    if ext_semantic_type not in ("none", ""):
        semantic_type = ext_semantic_type
        if ext_faker_method and col_type not in ("integer", "float", "datetime"):
            generation = {"strategy": "semantic", "params": {"faker_method": ext_faker_method}}
    elif semantic_type == "none":
        # Name-based fallback for numeric/datetime cols
        fallback_st, _ = _guess_semantic_from_name(name)
        if fallback_st != "none":
            semantic_type = fallback_st

    # Clamp to allowed set
    if semantic_type not in _ALLOWED_SEMANTIC_TYPES:
        semantic_type = "none"

    return ColumnSpec(
        name=name,
        type=col_type,
        semantic_type=semantic_type,  # type: ignore[arg-type]
        generation=generation,
        constraints=ColumnConstraints(
            unique=is_unique,
            nullable=(null_pct > 0),
            min=constraints_extra.get("min"),
            max=constraints_extra.get("max"),
        ),
        empirical={
            "null_rate": round(null_pct, 6),
            "unique_count": unique_count,
            "is_pk": is_pk,
        },
    ), is_pk


def _profile_string_column(
    name: str,
    clean: pd.Series,
    unique_count: int,
    ext_semantic_type: str,
    ext_faker_method: Optional[str],
    is_pk: bool = False,
) -> Tuple[str, Dict[str, Any], str]:
    """Route a string column to the best strategy."""
    n_clean = len(clean)

    # 1. Cardinality check FIRST — low-cardinality columns always use empirical_categorical
    #    to preserve real observed values (e.g. country codes "US","UK","DE" not "United States").
    #    Semantic generation (Faker) is reserved for high-cardinality columns that need new values.
    cardinality_ratio = unique_count / max(n_clean, 1)
    is_low_cardinality = (cardinality_ratio <= _CAT_RATIO_THRESHOLD) or (unique_count <= _CAT_UNIQUE_THRESHOLD)

    # String PK columns: use empirical_categorical to replay exact source values
    if is_pk:
        gen = _learn_categorical_marginal(clean)
        semantic_type = ext_semantic_type if ext_semantic_type not in ("none", "") else "id"
        logger.info("  col '%s': empirical_categorical (string PK, %d unique vals)", name, unique_count)
        return "string", gen, semantic_type

    if is_low_cardinality:
        gen = _learn_categorical_marginal(clean)
        # Run semantic detection for the type label only (not for strategy)
        if ext_semantic_type in ("none", ""):
            semantic_type, _ = detect_semantic(name, clean)
        else:
            semantic_type = ext_semantic_type
        logger.info("  col '%s': empirical_categorical (%d unique vals)", name, unique_count)
        return "categorical", gen, semantic_type

    # 2. Content-based semantic sniffing (values, not name) — high-cardinality only
    if ext_semantic_type in ("none", ""):
        semantic_type, faker_method = detect_semantic(name, clean)
    else:
        semantic_type = ext_semantic_type
        faker_method = ext_faker_method

    if faker_method and semantic_type not in ("none", "id", "currency", "category", "date"):
        logger.info("  col '%s': semantic '%s' (content-sniffed)", name, semantic_type)
        return "string", {"strategy": "semantic", "params": {"faker_method": faker_method}}, semantic_type

    # 3. Try format mask (structured codes / IDs) — only for short structured strings
    mask_params = _try_learn_pattern(clean)
    if mask_params:
        logger.info("  col '%s': learned_pattern mask='%s'", name, mask_params["mask"])
        return "string", {"strategy": "learned_pattern", "params": mask_params}, semantic_type or "id"

    # 4. High-cardinality free text — empirical_categorical (by frequency)
    #    Cap the pool at top-N covering 99% mass, fold tail to _other_
    gen = _learn_categorical_marginal(clean, target_coverage=0.99)
    logger.info("  col '%s': empirical_categorical (high-card free text)", name)
    return "string", gen, semantic_type or "none"


# ────────────────────────────────────────────────────────────────────────────
# Empirical learners
# ────────────────────────────────────────────────────────────────────────────

def _learn_numeric_marginal(vals: pd.Series, is_integer: bool) -> Dict[str, Any]:
    """Learn a 101-point quantile grid + rounding info from numeric values."""
    arr = vals.astype(float).values
    n = len(arr)

    # 101-point quantile grid (0th … 100th percentile)
    q_grid = np.linspace(0, 1, 101)
    quantiles = np.quantile(arr, q_grid).tolist()

    # Zero-inflation rate
    zero_inflation = float((arr == 0).mean()) if n > 0 else 0.0

    # Rounding granularity
    rounding = _detect_rounding(arr, is_integer)

    # Multimodality flag (simple: count histogram peaks)
    is_multimodal = False
    if n >= 20:
        try:
            counts, _ = np.histogram(arr, bins=min(20, n // 5))
            # Count sign changes in gradient → peaks
            grad = np.diff(counts.astype(float))
            peaks = int(np.sum((grad[:-1] > 0) & (grad[1:] < 0)))
            is_multimodal = peaks >= 2
        except Exception:
            pass

    return {
        "quantiles": quantiles,
        "is_integer": is_integer,
        "zero_inflation": round(zero_inflation, 6),
        "is_multimodal": is_multimodal,
        "min": float(arr.min()),
        "max": float(arr.max()),
        **rounding,
    }


def _detect_rounding(arr: np.ndarray, is_integer: bool) -> Dict[str, Any]:
    """Detect rounding granularity: .99 endings, multiples of 5/10/100, decimal places."""
    if is_integer:
        return {"round_decimals": 0}

    non_zero = arr[arr != 0]
    if len(non_zero) < 5:
        return {}

    # Check .99 price endings
    decimals = non_zero % 1
    rounded_dec = np.round(decimals, 2)
    if len(rounded_dec) >= 5:
        cnt_99 = np.sum(np.abs(rounded_dec - 0.99) < 0.005)
        if cnt_99 / len(rounded_dec) > 0.50:
            return {"price_ending_99": True, "round_decimals": 2}

    # Check for step-granularity (multiples of fixed step)
    for step in (100.0, 50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
        rem = non_zero % step
        # Remainder is near 0 or near step
        near_zero = np.abs(rem) < step * 0.02
        near_step = np.abs(rem - step) < step * 0.02
        if (near_zero | near_step).mean() > 0.90:
            if step >= 1.0:
                return {"value_multiplicity": {"step": step}}
            else:
                dp = max(0, -int(np.floor(np.log10(step))))
                return {"round_decimals": dp}

    # Modal decimal place count
    sample_strs = [f"{v:.8f}".rstrip("0") for v in non_zero[:200]]
    decimal_places = [len(s.split(".")[1]) if "." in s else 0 for s in sample_strs]
    if decimal_places:
        modal_dp = max(set(decimal_places), key=decimal_places.count)
        return {"round_decimals": min(modal_dp, 6)}

    return {}


def _learn_categorical_marginal(
    series: pd.Series,
    target_coverage: float = 1.0,
) -> Dict[str, Any]:
    """
    Full value→probability map with _other_ tail bucket.
    target_coverage: keep categories until their cumulative prob reaches this threshold.
    """
    str_vals = series.astype(str)
    freq = str_vals.value_counts(normalize=True)

    if target_coverage >= 1.0:
        # Keep everything
        values = {str(k): round(float(v), 8) for k, v in freq.items()}
        other_pool: List[str] = []
    else:
        cumsum = 0.0
        values: Dict[str, float] = {}
        other_vals: List[str] = []
        for cat, prob in freq.items():
            if cumsum < target_coverage:
                values[str(cat)] = round(float(prob), 8)
                cumsum += float(prob)
            else:
                other_vals.append(str(cat))

        if other_vals:
            other_prob = round(1.0 - sum(values.values()), 8)
            if other_prob > 0:
                values["_other_"] = max(0.0, other_prob)
            other_pool = other_vals[:_OTHER_POOL_CAP]
        else:
            other_pool = []

    # Normalize to exactly 1.0
    total = sum(values.values())
    if total > 0:
        values = {k: round(v / total, 8) for k, v in values.items()}

    result: Dict[str, Any] = {"values": values, "preserve_order": False}
    if other_pool:
        result["_other_pool_"] = other_pool

    return {"strategy": "empirical_categorical", "params": result}


def _learn_datetime_marginal(parsed: pd.Series) -> Dict[str, Any]:
    """Learn temporal structure: DOW/hour/month weights, trend, anchors."""
    ts = pd.to_datetime(parsed)
    ts = ts.dropna().sort_values()
    n = len(ts)

    anchor_start = ts.min().isoformat()
    anchor_end = ts.max().isoformat()

    # Detect granularity
    if (ts.dt.second != 0).any():
        granularity = "second"
    elif (ts.dt.minute != 0).any():
        granularity = "minute"
    elif (ts.dt.hour != 0).any():
        granularity = "hour"
    else:
        granularity = "day"

    # DOW weights (0=Monday … 6=Sunday)
    dow_counts = np.zeros(7)
    dow_vals = ts.dt.dayofweek.values
    for d in dow_vals:
        dow_counts[d] += 1
    dow_weights = (dow_counts / dow_counts.sum()).tolist() if dow_counts.sum() > 0 else [1 / 7] * 7

    # Hour weights (0–23)
    hour_counts = np.zeros(24)
    if granularity in ("hour", "minute", "second"):
        for h in ts.dt.hour.values:
            hour_counts[h] += 1
        hour_weights = (hour_counts / hour_counts.sum()).tolist() if hour_counts.sum() > 0 else [1 / 24] * 24
    else:
        hour_weights = [1 / 24] * 24

    # Month weights (1–12)
    month_counts = np.zeros(12)
    for m in ts.dt.month.values:
        month_counts[m - 1] += 1
    month_weights = (month_counts / month_counts.sum()).tolist() if month_counts.sum() > 0 else [1 / 12] * 12

    # Linear trend detection (slope of event count over time)
    trend = "none"
    if n >= 10:
        try:
            time_floats = (ts - ts.min()).dt.total_seconds().values.astype(float)
            if time_floats.max() > 0:
                slope, _, r, _, _ = scipy_stats.linregress(
                    np.linspace(0, 1, n), time_floats
                )
                if abs(r) > 0.5:
                    trend = "linear"
        except Exception:
            pass

    # Detect format
    sample_str = str(ts.iloc[0]) if n > 0 else ""
    if "T" in sample_str or " " in sample_str and granularity != "day":
        fmt = "%Y-%m-%d %H:%M:%S"
    else:
        fmt = "%Y-%m-%d"

    # Bootstrap pool for small datasets
    bootstrap_pool: List[str] = []
    if n <= _TS_POOL_CAP:
        bootstrap_pool = [t.strftime(fmt) for t in ts]

    return {
        "anchor_start": anchor_start,
        "anchor_end": anchor_end,
        "dow_weights": dow_weights,
        "hour_weights": hour_weights,
        "month_weights": month_weights,
        "trend": trend,
        "granularity": granularity,
        "format": fmt,
        "bootstrap_pool": bootstrap_pool,
    }


def _try_learn_pattern(series: pd.Series, min_consistency: float = 0.80) -> Optional[Dict[str, Any]]:
    """
    Derive a character-level mask from a sample of values.
    Returns None if the column doesn't have a consistent structured format.

    A structured code must be:
    - Short (≤ 40 characters)
    - Mostly consistent length (≥ 80% same length)
    - At least 15% of positions must be genuinely varied (> 1 unique char)
      — this excludes natural language text where words are the same but look like "A" wildcards
    """
    samples = series.dropna().astype(str).head(500).tolist()
    if len(samples) < 5:
        return None

    lengths = [len(s) for s in samples]
    modal_length = max(set(lengths), key=lengths.count)

    # Cap: long strings are almost never structured codes
    if modal_length > 40:
        return None

    consistent = [s for s in samples if len(s) == modal_length]

    if len(consistent) < len(samples) * min_consistency:
        return None  # too many length variations

    # Build per-position character classes
    mask_chars: List[str] = []
    alphabets: Dict[str, List[str]] = {}

    for pos in range(modal_length):
        chars_at_pos = set(s[pos] for s in consistent if pos < len(s))

        if all(c.isdigit() for c in chars_at_pos):
            mask_chars.append("#")
        elif all(c.isalpha() for c in chars_at_pos):
            mask_chars.append("A")
        elif len(chars_at_pos) == 1:
            lit = next(iter(chars_at_pos))
            mask_chars.append(lit)
        else:
            mask_chars.append("?")

        alphabets[str(pos)] = sorted(chars_at_pos)

    mask = "".join(mask_chars)

    # Must have at least one wildcard character
    if not any(c in "#A?" for c in mask):
        return None

    # Key quality gate: at least 15% of positions must have genuine variety (> 1 unique char).
    # Without this, "Product description number 99" would be accepted as a learned_pattern
    # with "A" wildcards that are actually frozen words.
    varied_positions = sum(
        1 for pos, ch in enumerate(mask)
        if ch in "#A?" and len(alphabets.get(str(pos), [])) > 1
    )
    min_varied = max(1, int(modal_length * 0.15))
    if varied_positions < min_varied:
        return None

    # Compute prefix (leading literals)
    prefix = ""
    for c in mask:
        if c not in "#A?":
            prefix += c
        else:
            break

    return {
        "mask": mask,
        "alphabets": alphabets,
        "examples": consistent[:5],
        "preserve_prefixes": bool(prefix),
        "modal_length": modal_length,
    }


# ────────────────────────────────────────────────────────────────────────────
# Joint model (copula + conditional groups)
# ────────────────────────────────────────────────────────────────────────────

def _learn_joint_model(
    df: pd.DataFrame,
    col_names: List[str],
    col_specs_map: Dict[str, ColumnSpec],
    primary_key: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Learn a Gaussian-copula joint model for numeric columns plus conditional
    groups for categorical → target dependencies.
    """
    # Identify copula-eligible columns (numeric, not PK, not all-null)
    sample_df = df if len(df) <= _PROFILE_SAMPLE_LIMIT else df.sample(_PROFILE_SAMPLE_LIMIT, random_state=42)

    numeric_cols = [
        c for c in col_names
        if c != primary_key
        and c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() >= 10
    ]

    # Detect conditional groups from low-cardinality categorical columns
    conditional_groups = _detect_conditional_groups(df, col_names, numeric_cols, primary_key)

    # Columns that are targets of conditional groups don't need to join the copula
    cond_target_cols = set()
    for cg in conditional_groups:
        cond_target_cols.update(cg["targets"])

    copula_cols = [c for c in numeric_cols if c not in cond_target_cols]

    if len(copula_cols) < _MIN_COPULA_COLS:
        if not conditional_groups:
            return None
        # Return model with only conditional groups, no copula
        return {
            "type": "gaussian_copula",
            "columns": [],
            "correlation_matrix": [],
            "conditional_groups": conditional_groups,
        }

    # Compute Spearman rank correlation matrix
    try:
        sub = sample_df[copula_cols].dropna()
        if len(sub) < 10:
            if not conditional_groups:
                return None
            return {
                "type": "gaussian_copula",
                "columns": [],
                "correlation_matrix": [],
                "conditional_groups": conditional_groups,
            }
        corr_df = sub.rank().corr(method="pearson")
        corr_mat = corr_df.values
        corr_mat = _make_psd(corr_mat)
        logger.info(
            "  joint_model: copula over %d cols, %d cond_groups",
            len(copula_cols), len(conditional_groups),
        )
        return {
            "type": "gaussian_copula",
            "columns": copula_cols,
            "correlation_matrix": corr_mat.tolist(),
            "conditional_groups": conditional_groups,
        }
    except Exception as exc:
        logger.warning("  joint_model: copula failed (%s), no joint model", exc)
        if conditional_groups:
            return {
                "type": "gaussian_copula",
                "columns": [],
                "correlation_matrix": [],
                "conditional_groups": conditional_groups,
            }
        return None


def _detect_conditional_groups(
    df: pd.DataFrame,
    all_cols: List[str],
    numeric_cols: List[str],
    primary_key: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Find categorical columns that strongly determine other columns via
    conditional distribution divergence (eta-squared effect size).
    """
    groups: List[Dict[str, Any]] = []
    sample_df = df if len(df) <= _PROFILE_SAMPLE_LIMIT else df.sample(_PROFILE_SAMPLE_LIMIT, random_state=42)

    # Candidate "by" columns: low-cardinality, non-numeric, not PK
    cat_candidates = [
        c for c in all_cols
        if c != primary_key
        and c in df.columns
        and not pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() <= 20
        and df[c].nunique() >= 2
        and df[c].notna().sum() >= 10
    ]

    used_targets: set = set()

    for cat_col in cat_candidates:
        targets: List[str] = []

        # Check all other columns (numeric and categorical targets)
        for target_col in all_cols:
            if target_col == cat_col or target_col == primary_key:
                continue
            if target_col in used_targets:
                continue
            if target_col not in df.columns:
                continue

            try:
                target_series = sample_df[target_col].dropna()
                cat_series = sample_df[cat_col][target_series.index]

                if pd.api.types.is_numeric_dtype(df[target_col]):
                    total_var = float(target_series.var())
                    if total_var < 1e-10:
                        continue
                    grouped = target_series.groupby(cat_series)
                    grand_mean = float(target_series.mean())
                    between_ss = sum(
                        len(g) * (float(g.mean()) - grand_mean) ** 2
                        for _, g in grouped
                        if len(g) > 0
                    )
                    eta_sq = between_ss / (total_var * len(target_series))
                    if eta_sq >= _COND_GROUP_ETA_SQ_THRESHOLD:
                        targets.append(target_col)
                else:
                    # For categorical targets, use Cramér's V via scipy chi2_contingency
                    ct_raw = pd.crosstab(cat_series, target_series)
                    if ct_raw.shape[0] < 2 or ct_raw.shape[1] < 2:
                        continue
                    try:
                        from scipy.stats import chi2_contingency as _chi2_ct
                        chi2_val, _, _, _ = _chi2_ct(ct_raw.values)
                        n_ct = float(ct_raw.values.sum())
                        phi2 = chi2_val / max(n_ct, 1)
                        r, c = ct_raw.shape
                        cramers_v = float(np.sqrt(phi2 / max(min(r, c) - 1, 1)))
                    except Exception:
                        cramers_v = 0.0
                    if cramers_v >= 0.3:
                        targets.append(target_col)
            except Exception:
                continue

        if not targets:
            continue

        # Build per-group marginals for each target
        group_marginals: Dict[str, Dict[str, Any]] = {}
        cat_values = sample_df[cat_col].dropna().unique().tolist()

        for gval in cat_values:
            mask = sample_df[cat_col] == gval
            gdf = sample_df[mask]
            if len(gdf) < 2:
                continue
            gval_str = str(gval)
            group_marginals[gval_str] = {}

            for target_col in targets:
                gseries = gdf[target_col].dropna()
                if len(gseries) == 0:
                    continue
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    is_int = pd.api.types.is_integer_dtype(df[target_col])
                    marginal = _learn_numeric_marginal(gseries.astype(float), is_int)
                    group_marginals[gval_str][target_col] = {
                        "strategy": "empirical_sample",
                        "params": marginal,
                    }
                else:
                    gen = _learn_categorical_marginal(gseries.astype(str))
                    group_marginals[gval_str][target_col] = gen

        if group_marginals:
            groups.append({
                "by": [cat_col],
                "targets": targets,
                "group_marginals": group_marginals,
            })
            used_targets.update(targets)
            logger.info(
                "  cond_group: '%s' → %s", cat_col, targets
            )

    return groups


def _make_psd(matrix: np.ndarray) -> np.ndarray:
    """Make a symmetric matrix positive semi-definite via eigenvalue clipping."""
    # Symmetrize
    mat = (matrix + matrix.T) / 2.0
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
    except np.linalg.LinAlgError:
        # Fallback: return identity
        return np.eye(matrix.shape[0])
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Re-normalize diagonal to 1 (correlation matrix)
    d = np.sqrt(np.diag(psd))
    d = np.where(d > 1e-10, d, 1.0)
    psd = psd / np.outer(d, d)
    np.clip(psd, -1.0, 1.0, out=psd)
    np.fill_diagonal(psd, 1.0)
    return psd


# ────────────────────────────────────────────────────────────────────────────
# Arithmetic constraint detection (generalized)
# ────────────────────────────────────────────────────────────────────────────

def _infer_arithmetic_constraints(df: pd.DataFrame) -> List[IntraRowConstraint]:
    constraints: List[IntraRowConstraint] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return constraints

    found_rules: set = set()

    for i, col_c in enumerate(numeric_cols):
        if len(constraints) >= 3:
            break
        col_c_vals = df[col_c].dropna()

        for col_a in numeric_cols:
            if col_a == col_c:
                continue
            for col_b in numeric_cols:
                if col_b in (col_a, col_c):
                    continue

                rule = _check_arithmetic_rule(df, col_a, col_b, col_c)
                if rule and rule not in found_rules:
                    constraints.append(IntraRowConstraint(type="arithmetic", rule=rule))
                    found_rules.add(rule)
                    logger.info("  arithmetic constraint detected: %s", rule)
                    break
            if len(constraints) >= 3:
                break

    return constraints


def _check_arithmetic_rule(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    col_c: str,
    rtol: float = 0.02,
    min_rows: int = 10,
) -> Optional[str]:
    """Test if c = a*b, c = a+b, c = a-b, or c = a/b holds within tolerance."""
    try:
        idx = df[[col_a, col_b, col_c]].dropna().index
        if len(idx) < min_rows:
            return None
        a = df.loc[idx, col_a].astype(float).values
        b = df.loc[idx, col_b].astype(float).values
        c = df.loc[idx, col_c].astype(float).values

        scale = np.abs(c).mean() + 1e-9

        if np.allclose(a * b, c, rtol=rtol, atol=scale * 0.01):
            return f"{col_c} = {col_a} * {col_b}"
        if np.allclose(a + b, c, rtol=rtol, atol=scale * 0.01):
            return f"{col_c} = {col_a} + {col_b}"
        if np.allclose(a - b, c, rtol=rtol, atol=scale * 0.01):
            return f"{col_c} = {col_a} - {col_b}"

        # Division: only when b != 0
        safe = np.abs(b) > 1e-9
        if safe.sum() >= min_rows:
            if np.allclose(a[safe] / b[safe], c[safe], rtol=rtol, atol=scale * 0.01):
                return f"{col_c} = {col_a} / {col_b}"
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────────────────────
# Cross-table conditioning
# ────────────────────────────────────────────────────────────────────────────

def _learn_parent_conditioning(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    parent_key: str,
    child_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Learn per-group child marginals keyed by parent attribute values.
    Only activates when the parent has low-cardinality non-key columns.
    """
    if parent_key not in parent_df.columns or child_key not in child_df.columns:
        return None

    # Find non-key parent columns with low cardinality
    parent_cat_cols = [
        c for c in parent_df.columns
        if c != parent_key
        and not pd.api.types.is_numeric_dtype(parent_df[c])
        and parent_df[c].nunique() <= 20
        and parent_df[c].notna().sum() >= 5
    ]
    if not parent_cat_cols:
        return None

    # Find numeric child columns (potential targets)
    child_target_cols = [
        c for c in child_df.columns
        if c != child_key
        and pd.api.types.is_numeric_dtype(child_df[c])
        and child_df[c].notna().sum() >= 5
    ]
    if not child_target_cols:
        return None

    # Join child to parent attributes
    try:
        joined = child_df[[child_key] + child_target_cols].merge(
            parent_df[[parent_key] + parent_cat_cols],
            left_on=child_key,
            right_on=parent_key,
            how="left",
        )
    except Exception:
        return None

    parent_col = parent_cat_cols[0]  # Use the first low-cardinality parent attribute

    group_marginals: Dict[str, Dict[str, Any]] = {}
    for gval, gdf in joined.groupby(parent_col):
        gval_str = str(gval)
        group_marginals[gval_str] = {}
        for target_col in child_target_cols:
            gseries = gdf[target_col].dropna()
            if len(gseries) < 3:
                continue
            is_int = pd.api.types.is_integer_dtype(child_df[target_col])
            marginal = _learn_numeric_marginal(gseries.astype(float), is_int)
            group_marginals[gval_str][target_col] = {
                "strategy": "empirical_sample",
                "params": marginal,
            }

    if not group_marginals:
        return None

    return {
        "parent_columns": [parent_col],
        "child_targets": child_target_cols,
        "method": "group_empirical",
        "group_marginals": group_marginals,
    }


# ────────────────────────────────────────────────────────────────────────────
# Relationship detection (preserved from original, unchanged logic)
# ────────────────────────────────────────────────────────────────────────────

def _infer_relationships(
    frames: Dict[str, pd.DataFrame],
    table_specs: Dict[str, TableSpec],
) -> List[RelationshipSpec]:
    """
    Three-strategy relationship detection:
    Strategy 0: shared-PK / 1:1 (one table's PK ⊆ another's PK)
    Strategy 1: exact column-name match (child col name == parent PK name, overlap ≥ 0.5)
    Strategy 2: name-heuristic (_id suffix → strip and match parent table name, overlap ≥ 0.85)
    """
    rels: List[RelationshipSpec] = []
    table_names = list(frames.keys())
    seen: set = set()

    # Index PKs
    pks_by_col: Dict[str, List[Tuple[str, set]]] = {}
    pk_of_table: Dict[str, Optional[str]] = {}
    for tname in table_names:
        ts = table_specs[tname]
        pk_of_table[tname] = ts.primary_key
        if ts.primary_key:
            col = ts.primary_key
            if col not in pks_by_col:
                pks_by_col[col] = []
            pk_series = frames[tname][col].dropna()
            if len(pk_series) > 50_000:
                pk_series = pk_series.sample(50_000, random_state=42)
            pks_by_col[col].append((tname, set(pk_series.tolist())))

    # Strategy 0: shared-PK / 1:1
    for pk_col in list(pks_by_col.keys()):
        entries = pks_by_col[pk_col]
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(len(entries)):
                if i == j:
                    continue
                p_name, p_set = entries[i]
                c_name, c_set = entries[j]
                key = (p_name, c_name, pk_col)
                if key in seen:
                    continue
                if not c_set.issubset(p_set) and not c_set == p_set:
                    continue
                rev_key = (c_name, p_name, pk_col)
                if c_set == p_set and rev_key in seen:
                    continue
                child_vals = frames[c_name][pk_col].dropna()
                card = _profile_cardinality(p_set, child_vals)
                participation = min(1.0, float(len(c_set) / max(len(p_set), 1)))
                rels.append(RelationshipSpec(
                    parent=p_name,
                    child=c_name,
                    parent_key=pk_col,
                    child_key=pk_col,
                    cardinality=card,
                    participation=round(participation, 4),
                    conditional_correlations=[],
                    temporal=None,
                ))
                seen.add(key)

    for child_name, child_df in frames.items():
        child_pk = pk_of_table.get(child_name)

        for child_col in child_df.columns:
            if child_col == child_pk:
                continue

            child_vals = child_df[child_col].dropna()
            if len(child_vals) == 0:
                continue
            sample_vals = child_vals if len(child_vals) <= 50_000 else child_vals.sample(50_000, random_state=42)

            # Strategy 1: exact column-name match
            if child_col in pks_by_col:
                for parent_name, parent_pk_set in pks_by_col[child_col]:
                    if parent_name == child_name:
                        continue
                    key = (parent_name, child_name, child_col)
                    if key in seen:
                        continue
                    overlap = sample_vals.isin(parent_pk_set).mean()
                    if overlap >= 0.5:
                        card = _profile_cardinality(parent_pk_set, child_vals)
                        participation = min(1.0, float(child_vals.nunique() / max(len(parent_pk_set), 1)))
                        rels.append(RelationshipSpec(
                            parent=parent_name,
                            child=child_name,
                            parent_key=child_col,
                            child_key=child_col,
                            cardinality=card,
                            participation=round(participation, 4),
                            conditional_correlations=[],
                            temporal=None,
                        ))
                        seen.add(key)
                continue

            # Strategy 2: name-heuristic (_id suffix)
            if not (child_col.endswith("_id") or (child_col.endswith("id") and len(child_col) > 2)):
                continue

            col_base = re.sub(r"[_]?id$", "", child_col, flags=re.IGNORECASE).lower().strip("_")
            col_norm = re.sub(r"[_\-\s]+", "", col_base)

            for parent_name in table_names:
                if parent_name == child_name:
                    continue
                parent_pk = pk_of_table.get(parent_name)
                if not parent_pk:
                    continue
                key = (parent_name, child_name, child_col)
                if key in seen:
                    continue

                parent_norm = re.sub(r"[_\-\s]+", "", parent_name.lower())
                if not (col_norm in parent_norm or parent_norm in col_norm):
                    continue

                for pname, pset in pks_by_col.get(parent_pk, []):
                    if pname != parent_name:
                        continue
                    overlap = sample_vals.isin(pset).mean()
                    if overlap >= 0.85:
                        card = _profile_cardinality(pset, child_vals)
                        participation = min(1.0, float(child_vals.nunique() / max(len(pset), 1)))
                        rels.append(RelationshipSpec(
                            parent=parent_name,
                            child=child_name,
                            parent_key=parent_pk,
                            child_key=child_col,
                            cardinality=card,
                            participation=round(participation, 4),
                            conditional_correlations=[],
                            temporal=None,
                        ))
                        seen.add(key)

    return rels


def _profile_cardinality(parent_keys: set, child_fk_series: pd.Series) -> CardinalitySpec:
    counts_per_parent = child_fk_series.value_counts()
    if len(counts_per_parent) == 0:
        return CardinalitySpec(distribution="poisson", params={"mu": 3.0})

    mu = float(counts_per_parent.mean())
    var = float(counts_per_parent.var()) if len(counts_per_parent) > 1 else mu

    if abs(var - mu) < mu * 0.5:
        return CardinalitySpec(distribution="poisson", params={"mu": round(mu, 2)}, min_children=1)
    elif var > mu * 1.5:
        p = max(0.01, min(0.99, mu / max(var, mu + 1)))
        n_nb = max(1, int(mu * p / (1 - p)))
        return CardinalitySpec(
            distribution="negative_binomial",
            params={"n": n_nb, "p": round(p, 4)},
            min_children=1,
        )
    else:
        lo = max(0, int(counts_per_parent.min()))
        hi = int(counts_per_parent.max())
        return CardinalitySpec(
            distribution="uniform",
            params={"low": lo, "high": hi},
            min_children=lo,
        )


def _guess_semantic_from_name(col_name: str):
    from backend.replicator.semantics import _guess_from_name
    return _guess_from_name(col_name)
