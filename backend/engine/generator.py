from __future__ import annotations
import re
import math
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

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


def generate(spec: GenerationSpec, progress_cb: Optional[Callable[[float, str], None]] = None) -> Dict[str, pd.DataFrame]:
    """
    Top-level generation entry point.
    Returns a dict of table_name → DataFrame.
    Raises ValueError if spec validation fails.
    """
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


def _generate_root_table(table: TableSpec, spec: GenerationSpec) -> pd.DataFrame:
    n = table.row_count
    data: Dict[str, Any] = {}

    for col in table.columns:
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

    data: Dict[str, Any] = {}
    for col in table.columns:
        if col.name == rel.child_key:
            data[col.name] = fk_column
            continue
        strategy = col.generation.get("strategy", "foreign_key")
        if strategy == "foreign_key":
            continue

        if rel.conditional_correlations:
            cc = next(
                (c for c in rel.conditional_correlations
                 if (isinstance(c, dict) and c.get("child_column") == col.name) or
                    (hasattr(c, "child_column") and c.child_column == col.name)),
                None
            )
            if cc:
                parent_col = cc.get("parent_column") if isinstance(cc, dict) else cc.parent_column
                if parent_col in parent_df.columns:
                    parent_vals = np.repeat(parent_df[parent_col].values[participating_mask], child_counts)
                    data[col.name] = _sample_column_conditioned(col, total_children, parent_vals, data)
                    continue

        data[col.name] = _sample_column(col, total_children, data, {})

    return pd.DataFrame(data)


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


def _sample_column(
    col: ColumnSpec,
    n: int,
    existing_data: Dict[str, Any],
    parent_row: Dict[str, Any],
) -> Any:
    strategy = col.generation.get("strategy", "")
    params = col.generation.get("params", {})
    c = col.constraints

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

    elif strategy == "regex_pattern":
        pattern = params.get("pattern", r"\w+")
        try:
            from faker.providers import BaseProvider
            results = []
            for _ in range(n):
                results.append(fake.bothify(text=_regex_to_bothify(pattern)))
            return results
        except Exception:
            return [fake.lexify(text="????") for _ in range(n)]

    elif strategy == "derived":
        expression = params.get("expression", "0")
        return _evaluate_derived(expression, n, existing_data)

    elif strategy == "foreign_key":
        return [None] * n

    elif col.type == "boolean":
        return list(np.random.choice([True, False], size=n))

    elif col.type in ("integer", "float"):
        return _sample_numeric(col, n, params)

    elif col.type == "categorical":
        cats = params.get("categories", ["A", "B", "C"])
        return list(np.random.choice(cats, size=n))

    else:
        return [fake.word() for _ in range(n)]


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


def _sample_column_conditioned(
    col: ColumnSpec, n: int, parent_vals: np.ndarray, existing_data: Dict[str, Any]
) -> list:
    """Sample a child column conditioned on parent values (shift by parent mean)."""
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


def _enforce_constraints(df: pd.DataFrame, table: TableSpec) -> pd.DataFrame:
    col_map = {c.name: c for c in table.columns}
    for col_name, col in col_map.items():
        if col_name not in df.columns:
            continue
        c = col.constraints

        if not c.nullable:
            df[col_name] = df[col_name].fillna(_default_value(col))

        if c.unique and col_name in df.columns:
            seen: set = set()
            new_vals = []
            counter = 0
            for v in df[col_name]:
                if v in seen:
                    while True:
                        counter += 1
                        candidate = f"{v}_{counter}" if col.type == "string" else (int(v) + counter if col.type == "integer" else v + counter * 0.001)
                        if candidate not in seen:
                            v = candidate
                            break
                seen.add(v)
                new_vals.append(v)
            df[col_name] = new_vals

    return df


def _apply_intra_row_constraints(df: pd.DataFrame, table: TableSpec) -> pd.DataFrame:
    for constraint in table.intra_row_constraints:
        rule = constraint.rule
        if constraint.type == "arithmetic":
            try:
                df = _apply_arithmetic_rule(df, rule)
            except Exception as e:
                logger.warning(f"Could not apply arithmetic rule '{rule}': {e}")
        elif constraint.type == "conditional":
            try:
                df = _apply_conditional_rule(df, rule)
            except Exception as e:
                logger.warning(f"Could not apply conditional rule '{rule}': {e}")
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
        local_df = pd.DataFrame({k: v for k, v in existing_data.items()
                                  if isinstance(v, (list, np.ndarray)) and len(v) == n})
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
