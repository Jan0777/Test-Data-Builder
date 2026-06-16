from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from backend.spec.models import (
    CardinalitySpec, ColumnConstraints, ColumnSpec,
    GenerationSpec, IntraRowConstraint, RelationshipSpec, TableSpec
)

logger = logging.getLogger(__name__)


def profile_to_spec(
    frames: Dict[str, pd.DataFrame],
    semantic_labels: Optional[Dict[str, Dict[str, Any]]] = None,
    row_count_override: Optional[int] = None,
) -> GenerationSpec:
    """
    Profile DataFrames into a GenerationSpec.
    semantic_labels: {table_name: {col_name: {semantic_type, faker_method}}}
    """
    tables: List[TableSpec] = []
    relationships: List[RelationshipSpec] = []

    table_specs_map: Dict[str, TableSpec] = {}

    for table_name, df in frames.items():
        sem = (semantic_labels or {}).get(table_name, {})
        n_rows = row_count_override if row_count_override else len(df)
        table_spec = _profile_table(table_name, df, n_rows, sem)
        tables.append(table_spec)
        table_specs_map[table_name] = table_spec

    if len(frames) > 1:
        inferred_rels = _infer_relationships(frames, table_specs_map)
        relationships.extend(inferred_rels)

    return GenerationSpec(version="1.0", tables=tables, relationships=relationships)


def _profile_table(
    name: str,
    df: pd.DataFrame,
    n_rows: int,
    semantic_labels: Dict[str, Any],
) -> TableSpec:
    columns: List[ColumnSpec] = []
    primary_key: Optional[str] = None

    for col_name in df.columns:
        col_spec, is_pk = _profile_column(col_name, df[col_name], len(df), semantic_labels.get(col_name, {}))
        columns.append(col_spec)
        if is_pk and primary_key is None:
            primary_key = col_name

    constraints: List[IntraRowConstraint] = _infer_arithmetic_constraints(df)

    return TableSpec(
        name=name,
        row_count=n_rows,
        primary_key=primary_key,
        columns=columns,
        intra_row_constraints=constraints,
    )


def _profile_column(
    name: str,
    series: pd.Series,
    n_rows: int,
    semantic_label: Dict[str, Any],
) -> Tuple[ColumnSpec, bool]:
    null_pct = float(series.isna().mean())
    unique_count = int(series.nunique())
    is_unique = (unique_count == len(series)) and len(series) > 0
    is_pk = is_unique and null_pct == 0.0

    col_type, generation, constraints = _infer_generation(name, series, null_pct, unique_count)

    semantic_type = semantic_label.get("semantic_type", "none")
    faker_method = semantic_label.get("faker_method")

    if semantic_type == "none":
        semantic_type, faker_method = _guess_semantic(name, col_type)

    if semantic_type != "none" and faker_method:
        generation = {"strategy": "semantic", "params": {"faker_method": faker_method}}

    if is_pk and col_type == "integer":
        generation = {"strategy": "sequential", "params": {"start": int(series.dropna().min()) if len(series.dropna()) > 0 else 1, "step": 1}}

    return ColumnSpec(
        name=name,
        type=col_type,
        semantic_type=semantic_type,
        generation=generation,
        constraints=ColumnConstraints(
            unique=is_unique,
            nullable=(null_pct > 0),
            min=constraints.get("min"),
            max=constraints.get("max"),
        ),
    ), is_pk


def _infer_generation(
    name: str,
    series: pd.Series,
    null_pct: float,
    unique_count: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    clean = series.dropna()
    n = len(clean)
    constraints: Dict[str, Any] = {}

    if n == 0:
        return "string", {"strategy": "semantic", "params": {"faker_method": "word"}}, constraints

    if hasattr(series, "dtype"):
        if pd.api.types.is_bool_dtype(series):
            true_rate = float(clean.mean()) if n > 0 else 0.5
            return "boolean", {
                "strategy": "categorical_sample",
                "params": {"values": {"true": round(true_rate, 4), "false": round(1 - true_rate, 4)}}
            }, constraints

        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            try:
                vals = clean.astype(float)
                mn = float(vals.min())
                mx = float(vals.max())
                mean = float(vals.mean())
                std = float(vals.std()) if n > 1 else 1.0
                skewness = float(stats.skew(vals)) if n > 2 else 0.0
                constraints = {"min": mn, "max": mx}

                if pd.api.types.is_integer_dtype(series):
                    if abs(skewness) < 0.5:
                        gen = {"strategy": "numeric_distribution", "params": {"dist": "normal", "mean": round(mean, 4), "std": round(max(std, 0.1), 4)}}
                    else:
                        gen = {"strategy": "numeric_distribution", "params": {"dist": "skewnormal", "mean": round(mean, 4), "std": round(max(std, 0.1), 4), "a": round(skewness, 2)}}
                    return "integer", gen, constraints
                else:
                    if mn >= 0 and skewness > 1.0:
                        gen = {"strategy": "numeric_distribution", "params": {"dist": "lognormal", "mean": round(float(np.log(mean + 1e-9)), 4), "std": round(float(np.std(np.log(vals + 1e-9))), 4)}}
                    else:
                        gen = {"strategy": "numeric_distribution", "params": {"dist": "normal", "mean": round(mean, 4), "std": round(max(std, 0.1), 4)}}
                    return "float", gen, constraints
            except Exception:
                pass

    try:
        parsed = pd.to_datetime(clean, infer_datetime_format=True, errors="raise")
        mn = parsed.min().strftime("%Y-%m-%d")
        mx = parsed.max().strftime("%Y-%m-%d")
        return "datetime", {
            "strategy": "datetime_range",
            "params": {"start": mn, "end": mx, "format": "%Y-%m-%d"}
        }, {}
    except Exception:
        pass

    str_vals = clean.astype(str)
    cardinality_ratio = unique_count / max(n, 1)

    if cardinality_ratio <= 0.1 or unique_count <= 20:
        freq = str_vals.value_counts(normalize=True)
        values = {str(k): round(float(v), 6) for k, v in freq.head(50).items()}
        total = sum(values.values())
        if total > 0:
            values = {k: round(v / total, 6) for k, v in values.items()}
        remaining = 1.0 - sum(values.values())
        if abs(remaining) > 0.001:
            first_key = next(iter(values))
            values[first_key] = round(values[first_key] + remaining, 6)
        return "categorical", {
            "strategy": "categorical_sample",
            "params": {"values": values}
        }, {}

    sample = str_vals.iloc[0] if n > 0 else ""
    if _looks_like_email(sample):
        return "string", {"strategy": "semantic", "params": {"faker_method": "email"}}, {}
    if _looks_like_phone(sample):
        return "string", {"strategy": "semantic", "params": {"faker_method": "phone_number"}}, {}

    return "string", {"strategy": "semantic", "params": {"faker_method": "word"}}, {}


def _guess_semantic(name: str, col_type: str):
    name_lower = name.lower()
    if any(k in name_lower for k in ("email", "e_mail", "mail")):
        return "email", "email"
    if any(k in name_lower for k in ("name", "fullname", "full_name", "first_name", "last_name")):
        return "name", "name"
    if any(k in name_lower for k in ("phone", "mobile", "cell", "tel")):
        return "phone", "phone_number"
    if any(k in name_lower for k in ("address", "street", "city", "zip", "postal")):
        return "address", "address"
    if any(k in name_lower for k in ("price", "salary", "wage", "amount", "cost", "revenue", "income")):
        return "currency", None
    if any(k in name_lower for k in ("category", "status", "type", "tier", "level")):
        return "category", None
    if any(k in name_lower for k in ("date", "time", "created", "updated", "at")):
        return "date", None
    if name_lower in ("id", "_id") or name_lower.endswith("_id") or name_lower.endswith("id"):
        return "id", None
    return "none", None


def _infer_arithmetic_constraints(df: pd.DataFrame) -> List[IntraRowConstraint]:
    constraints = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        return constraints

    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            for col_c in numeric_cols:
                if col_c in (col_a, col_b):
                    continue
                try:
                    product = df[col_a] * df[col_b]
                    corr = product.corr(df[col_c])
                    if corr > 0.99:
                        rule = f"{col_c} = {col_a} * {col_b}"
                        constraints.append(IntraRowConstraint(type="arithmetic", rule=rule))
                        break
                except Exception:
                    continue
            if constraints:
                break

    return constraints[:2]


def _infer_relationships(
    frames: Dict[str, pd.DataFrame],
    table_specs: Dict[str, "TableSpec"],
) -> List[RelationshipSpec]:
    rels = []
    table_names = list(frames.keys())

    pks: Dict[str, Tuple[str, "pd.Series"]] = {}
    for tname in table_names:
        ts = table_specs[tname]
        if ts.primary_key:
            pks[tname] = (ts.primary_key, frames[tname][ts.primary_key])

    for child_name, child_df in frames.items():
        for col in child_df.columns:
            if not (col.endswith("_id") or col.endswith("id")):
                continue
            for parent_name, (pk_col, pk_series) in pks.items():
                if parent_name == child_name:
                    continue
                col_clean = col.replace("_id", "").replace("id", "")
                parent_clean = parent_name.lower()
                if not (col_clean in parent_clean or parent_clean in col_clean):
                    continue
                child_vals = child_df[col].dropna()
                parent_vals = set(pk_series.tolist())
                if len(child_vals) == 0:
                    continue
                overlap = child_vals.isin(parent_vals).mean()
                if overlap >= 0.85:
                    card = _profile_cardinality(parent_vals, child_df[col].dropna())
                    participation = float(len(child_df[col].dropna().unique()) / max(len(parent_vals), 1))
                    participation = min(1.0, participation)
                    rels.append(RelationshipSpec(
                        parent=parent_name,
                        child=child_name,
                        parent_key=pk_col,
                        child_key=col,
                        cardinality=card,
                        participation=round(participation, 4),
                        conditional_correlations=[],
                        temporal=None,
                    ))

    return rels


def _profile_cardinality(parent_keys: set, child_fk_series: "pd.Series") -> CardinalitySpec:
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


def _looks_like_email(s: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s))


def _looks_like_phone(s: str) -> bool:
    return bool(re.match(r"^[\d\s\-\+\(\)]{7,15}$", s))
