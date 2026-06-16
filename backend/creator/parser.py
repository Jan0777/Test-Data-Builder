from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.spec.models import GenerationSpec
from backend.engine.validator import validate_spec

logger = logging.getLogger(__name__)


async def parse_query(query: str, row_count: Optional[int] = None) -> Tuple[GenerationSpec, str, List[str]]:
    """
    Parse a natural-language query into a validated GenerationSpec.
    Returns (spec, summary_message, conflict_warnings).
    Raises ValueError if the resulting spec has blocking conflicts.
    """
    from backend.llm.client import parse_intent

    raw_spec = await parse_intent(query)

    if row_count is not None:
        for table in raw_spec.get("tables", []):
            if "row_count" not in table or table.get("row_count", 0) == 0:
                table["row_count"] = row_count

    spec = GenerationSpec(**raw_spec)

    spec = _apply_defaults(spec)

    validation = validate_spec(spec)

    conflict_messages = [f"{c.table or ''}.{c.column or ''}: {c.message}".strip(".") for c in validation.conflicts]
    warning_messages = validation.warnings

    if not validation.valid:
        raise ValueError(f"Spec conflicts: {'; '.join(conflict_messages)}")

    summary = _build_summary(spec, query)

    return spec, summary, warning_messages


def _apply_defaults(spec: GenerationSpec) -> GenerationSpec:
    """Gap-fill missing details with sensible defaults."""
    for table in spec.tables:
        if table.row_count <= 0:
            table.row_count = 100

        col_names = {c.name for c in table.columns}

        if table.primary_key and table.primary_key not in col_names:
            from backend.spec.models import ColumnSpec, ColumnConstraints
            pk_col = ColumnSpec(
                name=table.primary_key,
                type="integer",
                semantic_type="id",
                generation={"strategy": "sequential", "params": {"start": 1, "step": 1}},
                constraints=ColumnConstraints(unique=True, nullable=False),
            )
            table.columns.insert(0, pk_col)

        if not table.primary_key and not any(c.constraints.unique for c in table.columns):
            pass

        for col in table.columns:
            if col.constraints.min is not None and col.constraints.max is not None:
                if col.constraints.min > col.constraints.max:
                    col.constraints.min, col.constraints.max = col.constraints.max, col.constraints.min

    return spec


def _build_summary(spec: GenerationSpec, original_query: str) -> str:
    table_parts = []
    for t in spec.tables:
        col_count = len(t.columns)
        table_parts.append(f"'{t.name}' ({t.row_count} rows, {col_count} columns)")

    summary = f"Interpreted as {len(spec.tables)} table(s): {', '.join(table_parts)}."

    if spec.relationships:
        rel_parts = [f"{r.parent} → {r.child}" for r in spec.relationships]
        summary += f" Relationships: {', '.join(rel_parts)}."

    return summary
