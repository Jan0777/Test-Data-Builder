from __future__ import annotations
import math
from typing import List
from backend.spec.models import GenerationSpec, SpecConflict, ValidationResult, ColumnSpec


def validate_spec(spec: GenerationSpec) -> ValidationResult:
    """
    Validate a GenerationSpec BEFORE generation.
    Returns a ValidationResult with errors (blocking) and warnings (non-blocking).
    """
    conflicts: List[SpecConflict] = []
    warnings: List[str] = []

    table_names = {t.name for t in spec.tables}

    for table in spec.tables:
        col_names = {c.name for c in table.columns}

        if table.row_count <= 0:
            conflicts.append(SpecConflict(
                table=table.name,
                message=f"row_count must be > 0, got {table.row_count}",
            ))

        if table.primary_key and table.primary_key not in col_names:
            conflicts.append(SpecConflict(
                table=table.name,
                message=f"primary_key '{table.primary_key}' not found in columns",
            ))

        for col in table.columns:
            _validate_column(col, table.name, table.row_count, conflicts, warnings)

    for rel in spec.relationships:
        if rel.parent not in table_names:
            conflicts.append(SpecConflict(
                message=f"Relationship references unknown parent table '{rel.parent}'"
            ))
        if rel.child not in table_names:
            conflicts.append(SpecConflict(
                message=f"Relationship references unknown child table '{rel.child}'"
            ))
        if not (0.0 <= rel.participation <= 1.0):
            conflicts.append(SpecConflict(
                message=f"Relationship {rel.parent}→{rel.child}: participation must be 0-1"
            ))

    return ValidationResult(
        valid=len(conflicts) == 0,
        conflicts=conflicts,
        warnings=warnings,
    )


def _validate_column(
    col: ColumnSpec,
    table_name: str,
    row_count: int,
    conflicts: List[SpecConflict],
    warnings: List[str],
) -> None:
    strategy = col.generation.get("strategy", "")
    params = col.generation.get("params", {})
    c = col.constraints

    if c.min is not None and c.max is not None and c.min > c.max:
        conflicts.append(SpecConflict(
            table=table_name, column=col.name,
            message=f"min ({c.min}) > max ({c.max})",
        ))

    if col.type == "integer" and c.unique:
        if c.min is not None and c.max is not None:
            possible = int(c.max) - int(c.min) + 1
            if possible < row_count:
                conflicts.append(SpecConflict(
                    table=table_name, column=col.name,
                    message=(
                        f"Cannot generate {row_count} unique integers in range "
                        f"[{int(c.min)}, {int(c.max)}] — only {possible} values exist"
                    ),
                ))

    if strategy == "categorical_sample":
        values = params.get("values", {})
        if values:
            total = sum(values.values())
            if not math.isclose(total, 1.0, abs_tol=0.01):
                conflicts.append(SpecConflict(
                    table=table_name, column=col.name,
                    message=f"categorical_sample probabilities sum to {total:.4f}, must be 1.0",
                ))
            if col.constraints.unique and len(values) < row_count:
                conflicts.append(SpecConflict(
                    table=table_name, column=col.name,
                    message=f"Cannot generate {row_count} unique values from {len(values)} categories",
                ))

    if strategy == "numeric_distribution":
        dist = params.get("dist", "normal")
        if dist == "normal":
            std = params.get("std", 1.0)
            if std <= 0:
                conflicts.append(SpecConflict(
                    table=table_name, column=col.name,
                    message=f"numeric_distribution normal: std must be > 0, got {std}",
                ))

    if strategy == "sequential" and col.constraints.unique:
        pass

    if strategy == "foreign_key":
        ref_table = params.get("references_table")
        if not ref_table:
            warnings.append(
                f"{table_name}.{col.name}: foreign_key strategy missing references_table"
            )
