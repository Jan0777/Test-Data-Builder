from __future__ import annotations
import math
import numpy as np
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

        # Validate joint_model if present
        if table.joint_model:
            _validate_joint_model(table.joint_model, table.name, col_names, conflicts, warnings)

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


def _validate_joint_model(
    joint_model: dict,
    table_name: str,
    col_names: set,
    conflicts: List[SpecConflict],
    warnings: List[str],
) -> None:
    jtype = joint_model.get("type")
    if jtype != "gaussian_copula":
        warnings.append(f"{table_name}: unknown joint_model type '{jtype}', skipping")
        return

    jcols = joint_model.get("columns", [])
    corr = joint_model.get("correlation_matrix", [])

    if not jcols:
        warnings.append(f"{table_name}: joint_model has no columns, copula disabled")
        return

    # All copula columns must exist in the table
    missing = [c for c in jcols if c not in col_names]
    if missing:
        conflicts.append(SpecConflict(
            table=table_name,
            message=f"joint_model references unknown columns: {missing}",
        ))

    # Correlation matrix must be square and match column count
    n = len(jcols)
    if corr:
        try:
            mat = np.array(corr, dtype=float)
            if mat.shape != (n, n):
                conflicts.append(SpecConflict(
                    table=table_name,
                    message=(
                        f"joint_model correlation_matrix shape {mat.shape} "
                        f"does not match {n} columns"
                    ),
                ))
            else:
                # Check PSD — warn, do not error (engine auto-corrects)
                eigvals = np.linalg.eigvalsh(mat)
                if eigvals.min() < -1e-6:
                    warnings.append(
                        f"{table_name}: correlation_matrix is not PSD "
                        f"(min eigenvalue {eigvals.min():.4f}), engine will auto-correct"
                    )
        except Exception as exc:
            conflicts.append(SpecConflict(
                table=table_name,
                message=f"joint_model correlation_matrix is invalid: {exc}",
            ))


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

    elif strategy == "empirical_categorical":
        values = params.get("values", {})
        if values:
            total = sum(values.values())
            if not math.isclose(total, 1.0, abs_tol=0.02):
                conflicts.append(SpecConflict(
                    table=table_name, column=col.name,
                    message=(
                        f"empirical_categorical probabilities sum to {total:.4f}, must be ~1.0"
                    ),
                ))
            if col.constraints.unique:
                # Effective unique count = number of explicit categories + pool size for _other_
                pool_size = len(params.get("_other_pool_", []))
                n_cats = len([k for k in values if k != "_other_"]) + pool_size
                if n_cats < row_count:
                    warnings.append(
                        f"{table_name}.{col.name}: empirical_categorical may not have enough "
                        f"unique values ({n_cats}) for {row_count} rows"
                    )

    elif strategy == "empirical_sample":
        quantiles = params.get("quantiles", [])
        if len(quantiles) != 101:
            conflicts.append(SpecConflict(
                table=table_name, column=col.name,
                message=(
                    f"empirical_sample requires exactly 101 quantile values, "
                    f"got {len(quantiles)}"
                ),
            ))

    elif strategy == "copula_member":
        marginal = params.get("marginal", {})
        quantiles = marginal.get("quantiles", [])
        if not quantiles:
            warnings.append(
                f"{table_name}.{col.name}: copula_member has no marginal quantiles, "
                "will fall back to independent sampling"
            )

    elif strategy == "learned_pattern":
        mask = params.get("mask", "")
        if not mask:
            conflicts.append(SpecConflict(
                table=table_name, column=col.name,
                message="learned_pattern requires a non-empty mask",
            ))
        if col.constraints.unique:
            # Estimate value space using full theoretical alphabet per position type.
            # We use max(observed_chars, theoretical_size) so the check is not
            # overly conservative when the source only shows a subset of chars.
            mask = params.get("mask", "")
            alphabets = params.get("alphabets", {})
            space = 1
            for pos, ch in enumerate(mask):
                observed = len(alphabets.get(str(pos), []))
                if ch == "#":
                    space *= max(observed, 10)
                elif ch == "A":
                    space *= max(observed, 26)
                elif ch == "?":
                    space *= max(observed, 36)
                # literal characters contribute 1 (no change to product)
            if space < row_count:
                conflicts.append(SpecConflict(
                    table=table_name, column=col.name,
                    message=(
                        f"learned_pattern mask '{mask}' has value space ~{space} "
                        f"but {row_count} unique values needed"
                    ),
                ))

    elif strategy == "numeric_distribution":
        dist = params.get("dist", "normal")
        if dist == "normal":
            std = params.get("std", 1.0)
            if std <= 0:
                conflicts.append(SpecConflict(
                    table=table_name, column=col.name,
                    message=f"numeric_distribution normal: std must be > 0, got {std}",
                ))

    elif strategy == "foreign_key":
        ref_table = params.get("references_table")
        if not ref_table:
            warnings.append(
                f"{table_name}.{col.name}: foreign_key strategy missing references_table"
            )
