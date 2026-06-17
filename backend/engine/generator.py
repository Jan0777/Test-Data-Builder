"""
Generation engine — samples DataFrames from a GenerationSpec.
Supports all legacy strategies AND the new empirical/copula strategies.
"""
from __future__ import annotations
import re
import math
import logging
import random
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import norm as _sci_norm

from backend.spec.models import (
    ColumnSpec, GenerationSpec, RelationshipSpec, TableSpec
)
from backend.engine.topological import (
    topological_sort, get_parent_relationships, get_child_relationships
)
from backend.engine.validator import validate_spec

logger = logging.getLogger(__name__)
fake = Faker()
Faker.seed(42)
np.random.seed(42)

# Max re-roll attempts for reject-sampling (uniqueness enforcement)
_MAX_RESAMPLE_ATTEMPTS = 8


# ────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ────────────────────────────────────────────────────────────────────────────

def generate(
    spec: GenerationSpec,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Top-level generation entry point.
    Returns a dict of table_name → DataFrame.
    Raises ValueError if spec validation fails.

    Parameters
    ----------
    spec : GenerationSpec
    progress_cb : optional callback(pct: float, message: str)
    seed : optional integer seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
        random.seed(seed)

    result = validate_spec(spec)
    if not result.valid:
        msgs = "; ".join(c.message for c in result.conflicts)
        raise ValueError(f"Invalid spec: {msgs}")

    order, _deferred = topological_sort(spec)
    table_map = {t.name: t for t in spec.tables}

    frames: Dict[str, pd.DataFrame] = {}
    total_tables = len(order)

    for idx, table_name in enumerate(order):
        if table_name not in table_map:
            continue
        table_spec = table_map[table_name]
        parent_rels = get_parent_relationships(spec, table_name)

        if progress_cb:
            pct = (idx / total_tables) * 80
            progress_cb(pct, f"Generating table '{table_name}'")

        if not parent_rels:
            df = _generate_root_table(table_spec, spec)
        else:
            df = _generate_child_table(table_spec, parent_rels, frames, spec)

        df = _enforce_constraints(df, table_spec)
        df = _apply_intra_row_constraints(df, table_spec)
        frames[table_name] = df

    if progress_cb:
        progress_cb(95, "Finalizing")

    return frames


# ────────────────────────────────────────────────────────────────────────────
# Table generation
# ────────────────────────────────────────────────────────────────────────────

def _generate_root_table(table: TableSpec, spec: GenerationSpec) -> pd.DataFrame:
    n = table.row_count
    data: Dict[str, Any] = {}

    # Joint model: generate copula + conditional-group columns together
    if table.joint_model:
        data = _generate_joint_columns(table, n, data)

    # Remaining columns — independent sampling
    for col in table.columns:
        if col.name in data:
            continue
        strategy = col.generation.get("strategy", "")
        if strategy == "foreign_key":
            continue
        data[col.name] = _sample_column(col, n, data, {})

    return pd.DataFrame(data)


def _generate_child_table(
    table: TableSpec,
    parent_rels: List[RelationshipSpec],
    frames: Dict[str, pd.DataFrame],
    spec: GenerationSpec,
) -> pd.DataFrame:
    rel = parent_rels[0]
    parent_df = frames.get(rel.parent)
    if parent_df is None:
        return _generate_root_table(table, spec)

    parent_keys = parent_df[rel.parent_key].values
    n_parents = len(parent_keys)

    participating_mask = np.random.random(n_parents) < rel.participation
    active_parents = parent_keys[participating_mask]

    child_counts = _sample_cardinality(rel, len(active_parents))
    total_children = int(child_counts.sum())

    if total_children == 0:
        return pd.DataFrame({col.name: pd.Series(dtype=_dtype_for(col)) for col in table.columns})

    fk_column = np.repeat(active_parents, child_counts)

    # Parent attributes repeated for each child row (used for parent conditioning)
    parent_attrs_repeated: Optional[pd.DataFrame] = None
    if rel.parent_conditioning and parent_df is not None:
        try:
            parent_attrs_repeated = parent_df.loc[
                np.repeat(parent_df.index[participating_mask], child_counts)
            ].reset_index(drop=True)
        except Exception:
            parent_attrs_repeated = None

    data: Dict[str, Any] = {}

    # Assign FK column first
    for col in table.columns:
        if col.name == rel.child_key:
            data[col.name] = fk_column
            break

    # Joint model generation for child table
    if table.joint_model:
        data = _generate_joint_columns(table, total_children, data)

    for col in table.columns:
        if col.name in data:
            continue
        strategy = col.generation.get("strategy", "foreign_key")
        if strategy == "foreign_key":
            continue

        # Parent conditioning override
        if (
            rel.parent_conditioning
            and parent_attrs_repeated is not None
            and col.name in rel.parent_conditioning.get("child_targets", [])
        ):
            group_col = rel.parent_conditioning.get("parent_columns", [None])[0]
            group_marginals = rel.parent_conditioning.get("group_marginals", {})
            if group_col and group_col in parent_attrs_repeated.columns:
                result = []
                for gval in parent_attrs_repeated[group_col]:
                    key = str(gval)
                    if key in group_marginals and col.name in group_marginals[key]:
                        val = _sample_from_marginal_dict(group_marginals[key][col.name], 1)
                        result.append(val[0] if val else None)
                    else:
                        result.extend(_sample_column(col, 1, data, {}))
                data[col.name] = result
                continue

        # Legacy conditional correlations
        if rel.conditional_correlations:
            cc = next(
                (c for c in rel.conditional_correlations
                 if (isinstance(c, dict) and c.get("child_column") == col.name) or
                    (hasattr(c, "child_column") and c.child_column == col.name)),
                None,
            )
            if cc:
                parent_col = cc.get("parent_column") if isinstance(cc, dict) else cc.parent_column
                if parent_col in parent_df.columns:
                    parent_vals = np.repeat(
                        parent_df[parent_col].values[participating_mask], child_counts
                    )
                    data[col.name] = _sample_column_conditioned(col, total_children, parent_vals, data)
                    continue

        data[col.name] = _sample_column(col, total_children, data, {})

    return pd.DataFrame(data)


# ────────────────────────────────────────────────────────────────────────────
# Joint model (copula + conditional groups)
# ────────────────────────────────────────────────────────────────────────────

def _generate_joint_columns(
    table: TableSpec,
    n: int,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate all columns that are part of the joint model.

    Precedence:
    1. Conditional groups (categorical-driven): generate "by" cols, then
       sample targets from per-group empirical marginals.
    2. Copula block: draw correlated normals, map through each column's
       inverse-CDF marginal.
    3. Columns already in data (e.g. FK) are left untouched.
    """
    joint_model = table.joint_model
    if not joint_model:
        return data

    col_map = {c.name: c for c in table.columns}
    conditional_groups = joint_model.get("conditional_groups", [])
    copula_cols: List[str] = joint_model.get("columns", [])
    corr_matrix_raw = joint_model.get("correlation_matrix", [])

    # ── Step 1: Conditional groups ──────────────────────────────────────
    cond_generated: set = set()
    for cg in conditional_groups:
        by_cols: List[str] = cg.get("by", [])
        target_cols: List[str] = cg.get("targets", [])
        group_marginals: Dict[str, Any] = cg.get("group_marginals", {})

        # Generate "by" column(s) if not yet done
        for by_col in by_cols:
            if by_col not in data and by_col in col_map:
                data[by_col] = _sample_column(col_map[by_col], n, data, {})
                cond_generated.add(by_col)

        # Sample each target conditioned on the "by" value
        primary_by = by_cols[0] if by_cols else None
        by_vals = data.get(primary_by, [None] * n) if primary_by else [None] * n

        for target_col in target_cols:
            if target_col in data:
                continue
            if target_col not in col_map:
                continue

            result: List[Any] = []
            for i, bv in enumerate(by_vals):
                gkey = str(bv) if bv is not None else "_null_"
                if gkey in group_marginals and target_col in group_marginals[gkey]:
                    sampled = _sample_from_marginal_dict(group_marginals[gkey][target_col], 1)
                    result.append(sampled[0] if sampled else None)
                else:
                    # Fallback: sample from column's own marginal
                    fallback = _sample_column(col_map[target_col], 1, data, {})
                    result.append(fallback[0] if fallback else None)

            data[target_col] = result
            cond_generated.add(target_col)

    # ── Step 2: Copula for remaining numeric columns ─────────────────────
    if not copula_cols or not corr_matrix_raw:
        return data

    remaining = [c for c in copula_cols if c not in data]
    if len(remaining) < 1:
        return data

    if len(remaining) == 1:
        col_name = remaining[0]
        if col_name in col_map:
            data[col_name] = _sample_column(col_map[col_name], n, data, {})
        return data

    try:
        corr_matrix = np.array(corr_matrix_raw, dtype=float)
        # Build sub-matrix for remaining columns
        all_col_idx = {c: i for i, c in enumerate(copula_cols)}
        valid_remaining = [c for c in remaining if c in all_col_idx]
        if len(valid_remaining) < 2:
            for col_name in remaining:
                if col_name in col_map and col_name not in data:
                    data[col_name] = _sample_column(col_map[col_name], n, data, {})
            return data

        indices = [all_col_idx[c] for c in valid_remaining]
        sub_corr = corr_matrix[np.ix_(indices, indices)]
        sub_corr = _make_psd(sub_corr)

        # Draw correlated standard normals
        Z = np.random.multivariate_normal(
            np.zeros(len(valid_remaining)), sub_corr, size=n
        )
        # Convert to uniforms via standard normal CDF
        U = _sci_norm.cdf(Z)

        for i, col_name in enumerate(valid_remaining):
            if col_name not in col_map or col_name in data:
                continue
            col_spec = col_map[col_name]
            strategy = col_spec.generation.get("strategy", "")

            if strategy == "copula_member":
                marginal = col_spec.generation.get("params", {}).get("marginal", {})
            elif strategy == "empirical_sample":
                marginal = col_spec.generation.get("params", {})
            else:
                marginal = {}

            quantiles = marginal.get("quantiles", [])
            if len(quantiles) == 101:
                vals = np.interp(U[:, i], np.linspace(0, 1, 101), quantiles)
                vals = _apply_numeric_post_processing(vals, marginal, col_spec)
                data[col_name] = vals.tolist()
            else:
                # Fallback: independent sampling
                data[col_name] = _sample_column(col_spec, n, data, {})

    except Exception as exc:
        logger.warning("Copula sampling failed: %s — falling back to independent", exc)
        for col_name in remaining:
            if col_name in col_map and col_name not in data:
                data[col_name] = _sample_column(col_map[col_name], n, data, {})

    return data


def _sample_from_marginal_dict(gen_dict: Dict[str, Any], n: int) -> List[Any]:
    """Sample n values from a generation dict (mini _sample_column without a ColumnSpec)."""
    strategy = gen_dict.get("strategy", "")
    params = gen_dict.get("params", {})

    if strategy == "empirical_sample":
        quantiles = params.get("quantiles", [])
        if len(quantiles) == 101:
            u = np.random.uniform(0, 1, n)
            vals = np.interp(u, np.linspace(0, 1, 101), quantiles)
            # Basic post-processing
            is_int = params.get("is_integer", False)
            rd = params.get("round_decimals")
            if rd is not None:
                vals = np.round(vals, int(rd))
            if is_int:
                return list(vals.astype(int))
            return list(vals.round(6))
        return [float(np.mean(quantiles)) if quantiles else 0.0] * n

    elif strategy == "empirical_categorical":
        values = params.get("values", {})
        other_pool = params.get("_other_pool_", [])
        if not values:
            return ["unknown"] * n
        cats = list(values.keys())
        probs = list(values.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        chosen = np.random.choice(len(cats), size=n, p=probs)
        result = []
        for idx in chosen:
            v = cats[idx]
            if v == "_other_" and other_pool:
                result.append(random.choice(other_pool))
            else:
                result.append(v)
        return result

    elif strategy == "categorical_sample":
        values = params.get("values", {})
        if not values:
            return ["unknown"] * n
        cats = list(values.keys())
        probs = list(values.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        return list(np.random.choice(cats, size=n, p=probs))

    return [None] * n


def _make_psd(matrix: np.ndarray) -> np.ndarray:
    """Make a symmetric matrix positive semi-definite via eigenvalue clipping."""
    mat = (matrix + matrix.T) / 2.0
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
    except np.linalg.LinAlgError:
        return np.eye(matrix.shape[0])
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    d = np.sqrt(np.diag(psd))
    d = np.where(d > 1e-10, d, 1.0)
    psd = psd / np.outer(d, d)
    np.clip(psd, -1.0, 1.0, out=psd)
    np.fill_diagonal(psd, 1.0)
    return psd


# ────────────────────────────────────────────────────────────────────────────
# Column-level sampling
# ────────────────────────────────────────────────────────────────────────────

def _sample_column(
    col: ColumnSpec,
    n: int,
    existing_data: Dict[str, Any],
    parent_row: Dict[str, Any],
) -> Any:
    strategy = col.generation.get("strategy", "")
    params = col.generation.get("params", {})

    # ── Legacy strategies ────────────────────────────────────────────────
    if strategy == "sequential":
        start = params.get("start", 1)
        step = params.get("step", 1)
        base = params.get("_offset", 0)
        return list(range(start + base, start + base + n * step, step))

    elif strategy == "numeric_distribution":
        return _sample_numeric(col, n, params)

    elif strategy == "categorical_sample":
        values = params.get("values", {})
        if not values:
            return ["unknown"] * n
        cats = list(values.keys())
        probs = list(values.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        return list(np.random.choice(cats, size=n, p=probs))

    elif strategy == "datetime_range":
        start_str = params.get("start", "2020-01-01")
        end_str = params.get("end", "2024-12-31")
        fmt = params.get("format", "%Y-%m-%d")
        try:
            start_dt = datetime.fromisoformat(start_str[:10])
            end_dt = datetime.fromisoformat(end_str[:10])
        except Exception:
            start_dt = datetime(2020, 1, 1)
            end_dt = datetime(2024, 12, 31)
        delta_days = max(1, (end_dt - start_dt).days)
        offsets = np.random.randint(0, delta_days + 1, size=n)
        return [(start_dt + timedelta(days=int(o))).strftime(fmt) for o in offsets]

    elif strategy == "semantic":
        faker_method = params.get("faker_method", "word")
        return [_call_faker(faker_method) for _ in range(n)]

    elif strategy == "semantic_record":
        return _sample_semantic_record(col, n, params)

    elif strategy == "regex_pattern":
        pattern = params.get("pattern", r"\w+")
        try:
            return [fake.bothify(text=_regex_to_bothify(pattern)) for _ in range(n)]
        except Exception:
            return [fake.lexify(text="????") for _ in range(n)]

    elif strategy == "derived":
        expression = params.get("expression", "0")
        return _evaluate_derived(expression, n, existing_data)

    elif strategy == "foreign_key":
        return [None] * n

    # ── New empirical strategies ─────────────────────────────────────────

    elif strategy == "empirical_sample":
        return _sample_empirical_numeric(col, n, params)

    elif strategy == "copula_member":
        # Standalone (outside of joint model path) — sample from marginal
        marginal = params.get("marginal", params)
        quantiles = marginal.get("quantiles", [])
        if len(quantiles) == 101:
            u = np.random.uniform(0, 1, n)
            vals = np.interp(u, np.linspace(0, 1, 101), quantiles)
            return list(_apply_numeric_post_processing(vals, marginal, col))
        return _sample_numeric(col, n, params)

    elif strategy == "empirical_categorical":
        return _sample_empirical_categorical(n, params)

    elif strategy == "learned_pattern":
        return _sample_learned_pattern(n, params, col.constraints.unique)

    elif strategy == "empirical_datetime":
        return _sample_empirical_datetime(n, params)

    # ── Type-based fallbacks ─────────────────────────────────────────────

    elif col.type == "boolean":
        return list(np.random.choice([True, False], size=n))

    elif col.type in ("integer", "float"):
        return _sample_numeric(col, n, params)

    elif col.type == "categorical":
        cats = params.get("categories", ["A", "B", "C"])
        return list(np.random.choice(cats, size=n))

    else:
        return [fake.word() for _ in range(n)]


# ────────────────────────────────────────────────────────────────────────────
# Empirical numeric: inverse-CDF from 101-point quantile grid (C1)
# ────────────────────────────────────────────────────────────────────────────

def _sample_empirical_numeric(col: ColumnSpec, n: int, params: Dict[str, Any]) -> list:
    quantiles = params.get("quantiles", [])
    if len(quantiles) != 101:
        return _sample_numeric(col, n, params)

    u = np.random.uniform(0, 1, n)
    vals = np.interp(u, np.linspace(0, 1, 101), quantiles)
    return list(_apply_numeric_post_processing(vals, params, col))


def _apply_numeric_post_processing(
    vals: np.ndarray,
    params: Dict[str, Any],
    col: ColumnSpec,
) -> np.ndarray:
    """Apply zero-inflation, rounding, price endings, and integer casting."""
    # Zero inflation
    zi = params.get("zero_inflation", 0.0)
    if zi > 0:
        zi_mask = np.random.random(len(vals)) < zi
        vals = vals.copy()
        vals[zi_mask] = 0.0

    # Price .99 endings
    if params.get("price_ending_99", False):
        vals = np.floor(vals) + 0.99

    # Step multiplicity
    vm = params.get("value_multiplicity")
    if isinstance(vm, dict):
        step = float(vm.get("step", 1.0))
        if step > 0:
            vals = np.round(vals / step) * step

    # Decimal rounding
    rd = params.get("round_decimals")
    if rd is not None:
        vals = np.round(vals, int(rd))

    # Integer cast
    if params.get("is_integer", False) or col.type == "integer":
        return vals.astype(int).astype(float)

    return vals


# ────────────────────────────────────────────────────────────────────────────
# Empirical categorical: full value→prob map (C2)
# ────────────────────────────────────────────────────────────────────────────

def _sample_empirical_categorical(n: int, params: Dict[str, Any]) -> list:
    values: Dict[str, float] = params.get("values", {})
    other_pool: List[str] = params.get("_other_pool_", [])

    if not values:
        return ["unknown"] * n

    cats = list(values.keys())
    probs = list(values.values())
    total = sum(probs)
    if total <= 0:
        return [cats[0]] * n
    probs = [p / total for p in probs]

    chosen_indices = np.random.choice(len(cats), size=n, p=probs)
    result: List[Any] = []
    for idx in chosen_indices:
        v = cats[idx]
        if v == "_other_" and other_pool:
            result.append(random.choice(other_pool))
        else:
            result.append(v)
    return result


# ────────────────────────────────────────────────────────────────────────────
# Learned pattern (C3)
# ────────────────────────────────────────────────────────────────────────────

def _sample_learned_pattern(n: int, params: Dict[str, Any], unique: bool) -> list:
    mask: str = params.get("mask", "")
    alphabets: Dict[str, List[str]] = params.get("alphabets", {})
    modal_length: int = params.get("modal_length", len(mask))

    if not mask:
        return [fake.lexify(text="????") for _ in range(n)]

    def _generate_one() -> str:
        chars = []
        for pos, ch in enumerate(mask):
            pos_str = str(pos)
            pool = alphabets.get(pos_str, [])
            if ch == "#":
                chars.append(random.choice(pool) if pool else str(random.randint(0, 9)))
            elif ch == "A":
                chars.append(random.choice(pool) if pool else random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            elif ch == "?":
                chars.append(random.choice(pool) if pool else random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
            else:
                chars.append(ch)
        return "".join(chars)

    if not unique:
        return [_generate_one() for _ in range(n)]

    # Unique: re-roll on collision (reject sampling)
    seen: set = set()
    result: List[str] = []
    for _ in range(n):
        for _attempt in range(_MAX_RESAMPLE_ATTEMPTS * 3):
            v = _generate_one()
            if v not in seen:
                seen.add(v)
                result.append(v)
                break
        else:
            # Exhausted: append a suffixed value
            v = _generate_one() + f"_{len(result)}"
            seen.add(v)
            result.append(v)
    return result


# ────────────────────────────────────────────────────────────────────────────
# Empirical datetime (C4)
# ────────────────────────────────────────────────────────────────────────────

def _sample_empirical_datetime(n: int, params: Dict[str, Any]) -> list:
    bootstrap_pool: List[str] = params.get("bootstrap_pool", [])
    fmt: str = params.get("format", "%Y-%m-%d")
    granularity: str = params.get("granularity", "day")
    trend: str = params.get("trend", "none")

    # Bootstrap from real timestamps when pool is available
    if bootstrap_pool and len(bootstrap_pool) >= 10:
        indices = np.random.randint(0, len(bootstrap_pool), size=n)
        return [bootstrap_pool[i] for i in indices]

    # Otherwise, use DOW/hour/month weights
    anchor_start_str: str = params.get("anchor_start", "2020-01-01")
    anchor_end_str: str = params.get("anchor_end", "2024-12-31")

    try:
        anchor_start = datetime.fromisoformat(anchor_start_str[:19])
        anchor_end = datetime.fromisoformat(anchor_end_str[:19])
    except Exception:
        anchor_start = datetime(2020, 1, 1)
        anchor_end = datetime(2024, 12, 31)

    delta_secs = max(1, int((anchor_end - anchor_start).total_seconds()))

    dow_weights = params.get("dow_weights", [1 / 7] * 7)
    hour_weights = params.get("hour_weights", [1 / 24] * 24)
    month_weights = params.get("month_weights", [1 / 12] * 12)

    # Normalize weights
    dow_w = np.array(dow_weights, dtype=float)
    hour_w = np.array(hour_weights, dtype=float)
    month_w = np.array(month_weights, dtype=float)
    dow_w /= dow_w.sum() or 1
    hour_w /= hour_w.sum() or 1
    month_w /= month_w.sum() or 1

    results: List[str] = []
    # Sample n random offsets, then accept/reject by DOW and month; fall back to uniform
    attempts = 0
    max_attempts = n * 20
    while len(results) < n and attempts < max_attempts:
        batch = min(n * 4, max_attempts - attempts)
        offsets = np.random.randint(0, delta_secs, size=batch)
        for offset in offsets:
            if len(results) >= n:
                break
            dt = anchor_start + timedelta(seconds=int(offset))
            dow_prob = dow_w[dt.weekday()]
            month_prob = month_w[dt.month - 1]
            hour_prob = hour_w[dt.hour] if granularity != "day" else 1.0
            accept_prob = (dow_prob / (1 / 7)) * (month_prob / (1 / 12))
            if granularity != "day":
                accept_prob *= hour_prob / (1 / 24)
            accept_prob = min(accept_prob, 1.0)
            if random.random() < accept_prob:
                results.append(dt.strftime(fmt))
        attempts += batch

    # Pad with uniform samples if rejection sampling was too slow
    while len(results) < n:
        offset = random.randint(0, delta_secs)
        dt = anchor_start + timedelta(seconds=offset)
        results.append(dt.strftime(fmt))

    return results[:n]


# ────────────────────────────────────────────────────────────────────────────
# Semantic record — locale-consistent multi-field bundle (C7)
# ────────────────────────────────────────────────────────────────────────────

def _sample_semantic_record(col: ColumnSpec, n: int, params: Dict[str, Any]) -> list:
    """
    Generate locale-consistent values using Faker profiles.
    The field name in params["field"] determines which profile attribute to return.
    """
    field = params.get("field", "name")
    locale = params.get("locale", "en_US")
    try:
        locale_fake = Faker(locale)
        Faker.seed(42)
    except Exception:
        locale_fake = fake

    results = []
    for _ in range(n):
        try:
            profile = locale_fake.simple_profile()
            val = profile.get(field, locale_fake.word())
        except Exception:
            val = locale_fake.word()
        results.append(str(val))
    return results


# ────────────────────────────────────────────────────────────────────────────
# Legacy numeric sampling
# ────────────────────────────────────────────────────────────────────────────

def _sample_numeric(col: ColumnSpec, n: int, params: Dict[str, Any]) -> list:
    c = col.constraints
    dist = params.get("dist", "uniform" if (c.min is not None and c.max is not None) else "normal")

    if dist == "normal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        vals = np.random.normal(loc=mean, scale=max(std, 1e-9), size=n)
    elif dist == "lognormal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        vals = np.random.lognormal(mean=mean, sigma=max(std, 1e-9), size=n)
    elif dist == "uniform":
        low = params.get("low", c.min if c.min is not None else 0.0)
        high = params.get("high", c.max if c.max is not None else 100.0)
        vals = np.random.uniform(low=float(low), high=float(high), size=n)
    elif dist == "skewnormal":
        from scipy.stats import skewnorm
        a = params.get("a", 0.0)
        loc = params.get("mean", 0.0)
        scale = params.get("std", 1.0)
        vals = skewnorm.rvs(a=a, loc=loc, scale=max(scale, 1e-9), size=n)
    else:
        vals = np.random.normal(size=n)

    if c.min is not None:
        vals = np.maximum(vals, c.min)
    if c.max is not None:
        vals = np.minimum(vals, c.max)

    if col.type == "integer":
        return list(vals.astype(int))
    return list(vals.round(4))


# ────────────────────────────────────────────────────────────────────────────
# Cardinality sampling
# ────────────────────────────────────────────────────────────────────────────

def _sample_cardinality(rel: RelationshipSpec, n_parents: int) -> np.ndarray:
    card = rel.cardinality
    if isinstance(card, dict):
        dist = card.get("distribution", "poisson")
        params = card.get("params", {"mu": 3.0})
        min_c = card.get("min_children", 0)
        max_c = card.get("max_children", None)
    else:
        dist = card.distribution
        params = card.params
        min_c = card.min_children
        max_c = card.max_children

    if dist == "poisson":
        mu = params.get("mu", 3.0)
        counts = np.random.poisson(lam=mu, size=n_parents)
    elif dist == "negative_binomial":
        n_param = params.get("n", 5)
        p_param = params.get("p", 0.5)
        counts = np.random.negative_binomial(n=n_param, p=p_param, size=n_parents)
    elif dist == "uniform":
        low = int(params.get("low", 1))
        high = int(params.get("high", 5))
        counts = np.random.randint(low, high + 1, size=n_parents)
    else:
        counts = np.random.poisson(lam=3.0, size=n_parents)

    counts = np.maximum(counts, min_c)
    if max_c is not None:
        counts = np.minimum(counts, max_c)

    return counts.astype(int)


# ────────────────────────────────────────────────────────────────────────────
# Constraint enforcement — resample instead of clamp (C6)
# ────────────────────────────────────────────────────────────────────────────

def _enforce_constraints(df: pd.DataFrame, table: TableSpec) -> pd.DataFrame:
    col_map = {c.name: c for c in table.columns}

    for col_name, col in col_map.items():
        if col_name not in df.columns:
            continue
        c = col.constraints

        # Null enforcement
        if not c.nullable:
            df[col_name] = df[col_name].fillna(_default_value(col))

        # Range enforcement — resample before clamping
        strategy = col.generation.get("strategy", "")
        if c.min is not None or c.max is not None:
            if strategy in ("empirical_sample", "copula_member", "numeric_distribution"):
                mask = pd.Series([False] * len(df))
                if c.min is not None:
                    mask |= pd.to_numeric(df[col_name], errors="coerce") < c.min
                if c.max is not None:
                    mask |= pd.to_numeric(df[col_name], errors="coerce") > c.max

                if mask.any():
                    n_bad = int(mask.sum())
                    resampled = False
                    for _attempt in range(_MAX_RESAMPLE_ATTEMPTS):
                        new_vals = _sample_column(col, n_bad, {}, {})
                        new_series = pd.to_numeric(pd.Series(new_vals), errors="coerce")
                        still_bad = pd.Series([False] * n_bad)
                        if c.min is not None:
                            still_bad |= new_series < c.min
                        if c.max is not None:
                            still_bad |= new_series > c.max
                        df.loc[mask, col_name] = new_vals
                        mask_new = pd.Series([False] * len(df))
                        if c.min is not None:
                            mask_new |= pd.to_numeric(df[col_name], errors="coerce") < c.min
                        if c.max is not None:
                            mask_new |= pd.to_numeric(df[col_name], errors="coerce") > c.max
                        mask = mask_new
                        if not mask.any():
                            resampled = True
                            break

                    if not resampled and mask.any():
                        # Final guard clamp — only as last resort, with warning
                        logger.warning(
                            "col '%s': %d values out of range after %d resample attempts — clamping",
                            col_name, int(mask.sum()), _MAX_RESAMPLE_ATTEMPTS,
                        )
                        if c.min is not None:
                            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").clip(lower=c.min)
                        if c.max is not None:
                            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").clip(upper=c.max)
            else:
                # Legacy clamp for non-empirical strategies
                if c.min is not None:
                    numeric = pd.to_numeric(df[col_name], errors="coerce")
                    df[col_name] = np.maximum(numeric, c.min)
                if c.max is not None:
                    numeric = pd.to_numeric(df[col_name], errors="coerce")
                    df[col_name] = np.minimum(numeric, c.max)

        # Uniqueness enforcement — re-roll duplicates, not mutate (C6)
        if c.unique and col_name in df.columns:
            seen: set = set()
            new_vals: List[Any] = []
            needs_fix: List[int] = []

            raw = df[col_name].tolist()
            for i, v in enumerate(raw):
                if v in seen:
                    needs_fix.append(i)
                else:
                    seen.add(v)
                new_vals.append(v)

            if needs_fix:
                for pos in needs_fix:
                    # Re-roll up to K times
                    for _att in range(_MAX_RESAMPLE_ATTEMPTS * 4):
                        candidate = _sample_column(col, 1, {}, {})[0]
                        if candidate not in seen:
                            seen.add(candidate)
                            new_vals[pos] = candidate
                            break
                    else:
                        # Truly exhausted — use a derived unique suffix
                        base = new_vals[pos]
                        for suffix in range(1, 10_000):
                            if col.type == "integer":
                                candidate = int(base or 0) + suffix + max(seen, default=0, key=lambda x: int(x) if isinstance(x, (int, float)) else 0)
                            elif col.type == "float":
                                candidate = float(base or 0.0) + suffix * 1e-9
                            else:
                                candidate = f"{base}_{suffix}"
                            if candidate not in seen:
                                seen.add(candidate)
                                new_vals[pos] = candidate
                                break
                df[col_name] = new_vals

    return df


# ────────────────────────────────────────────────────────────────────────────
# Intra-row constraint application
# ────────────────────────────────────────────────────────────────────────────

def _apply_intra_row_constraints(df: pd.DataFrame, table: TableSpec) -> pd.DataFrame:
    for constraint in table.intra_row_constraints:
        rule = constraint.rule
        if constraint.type == "arithmetic":
            try:
                df = _apply_arithmetic_rule(df, rule)
            except Exception as e:
                logger.warning("Could not apply arithmetic rule '%s': %s", rule, e)
        elif constraint.type == "conditional":
            try:
                df = _apply_conditional_rule(df, rule)
            except Exception as e:
                logger.warning("Could not apply conditional rule '%s': %s", rule, e)
    return df


def _apply_arithmetic_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = re.match(r"(\w+)\s*=\s*(.+)", rule.strip())
    if not m:
        return df
    target_col = m.group(1).strip()
    expression = m.group(2).strip()
    if target_col in df.columns:
        try:
            df[target_col] = df.eval(expression)
        except Exception:
            pass
    return df


def _apply_conditional_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    m = re.match(r"if\s+(.+?)\s+then\s+(.+)", rule.strip(), re.IGNORECASE)
    if not m:
        return df
    return df


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _sample_column_conditioned(
    col: ColumnSpec, n: int, parent_vals: np.ndarray, existing_data: Dict[str, Any]
) -> list:
    """Sample a child column with a legacy mean-shift conditional (kept for backward compat)."""
    base = _sample_column(col, n, existing_data, {})
    if col.type in ("integer", "float") and parent_vals is not None:
        try:
            base_arr = np.array(base, dtype=float)
            parent_arr = np.array(parent_vals, dtype=float)
            shift = (parent_arr - parent_arr.mean()) * 0.5
            shifted = base_arr + shift
            c = col.constraints
            if c.min is not None:
                shifted = np.maximum(shifted, c.min)
            if c.max is not None:
                shifted = np.minimum(shifted, c.max)
            if col.type == "integer":
                return list(shifted.astype(int))
            return list(shifted.round(4))
        except Exception:
            pass
    return base


def _call_faker(method: str) -> str:
    try:
        fn = getattr(fake, method, None)
        if fn and callable(fn):
            result = fn()
            return str(result)
    except Exception:
        pass
    return fake.word()


def _regex_to_bothify(pattern: str) -> str:
    result = re.sub(r"\d", "#", pattern)
    result = re.sub(r"\w", "?", result)
    return result[:50] if len(result) > 50 else result


def _evaluate_derived(expression: str, n: int, existing_data: Dict[str, Any]) -> list:
    try:
        local_df = pd.DataFrame({
            k: v for k, v in existing_data.items()
            if isinstance(v, (list, np.ndarray)) and len(v) == n
        })
        if not local_df.empty:
            result = local_df.eval(expression)
            return list(result)
    except Exception:
        pass
    return [0.0] * n


def _default_value(col: ColumnSpec) -> Any:
    if col.type == "integer":
        return 0
    if col.type == "float":
        return 0.0
    if col.type == "boolean":
        return False
    if col.type == "datetime":
        return "2020-01-01"
    return ""


def _dtype_for(col: ColumnSpec):
    if col.type == "integer":
        return int
    if col.type == "float":
        return float
    if col.type == "boolean":
        return bool
    return str
