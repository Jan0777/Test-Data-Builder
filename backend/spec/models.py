from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Known generation strategies — both legacy and new empirical ones.
# The engine supports all of these; the profiler emits the empirical ones.
# ---------------------------------------------------------------------------
KNOWN_STRATEGIES: List[str] = [
    # Legacy / Creator-mode strategies (always supported)
    "sequential",
    "numeric_distribution",
    "categorical_sample",
    "datetime_range",
    "semantic",
    "regex_pattern",
    "derived",
    "foreign_key",
    # New empirical strategies (Replicator v2)
    "empirical_sample",       # bootstrap / inverse-CDF from 101-point quantile grid
    "empirical_categorical",  # full value→prob map with _other_ tail bucket
    "copula_member",          # participates in table-level Gaussian copula
    "learned_pattern",        # format mask inferred from source values
    "empirical_datetime",     # temporal-structure-preserving datetime sampler
    "semantic_record",        # locale-consistent multi-field bundle
]


class ColumnConstraints(BaseModel):
    unique: bool = False
    nullable: bool = True
    min: Optional[float] = None
    max: Optional[float] = None


class GenerationStrategy(BaseModel):
    strategy: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ColumnSpec(BaseModel):
    name: str
    type: Literal["integer", "float", "string", "categorical", "datetime", "boolean"]
    semantic_type: Literal[
        "id", "name", "email", "address", "city", "state", "zip", "country",
        "currency", "category", "date", "phone", "url", "company", "none"
    ] = "none"
    generation: Dict[str, Any] = Field(default_factory=dict)
    constraints: ColumnConstraints = Field(default_factory=ColumnConstraints)
    # Raw learned profile payload — marginals, null rates, rounding info, etc.
    empirical: Optional[Dict[str, Any]] = None


class IntraRowConstraint(BaseModel):
    type: Literal["arithmetic", "conditional"]
    rule: str


class TableSpec(BaseModel):
    name: str
    row_count: int
    primary_key: Optional[str] = None
    columns: List[ColumnSpec] = Field(default_factory=list)
    intra_row_constraints: List[IntraRowConstraint] = Field(default_factory=list)
    # Gaussian-copula joint model for correlated numeric columns, plus
    # conditional group specs for categorical → target relationships.
    # {
    #   "type": "gaussian_copula",
    #   "columns": [col names in copula],
    #   "correlation_matrix": [[...]],   # Spearman, PSD-corrected
    #   "conditional_groups": [
    #       {
    #           "by": ["country"],
    #           "targets": ["currency"],
    #           "group_marginals": {
    #               "<group_value>": {"<target_col>": {generation dict}, ...},
    #               ...
    #           }
    #       }
    #   ]
    # }
    joint_model: Optional[Dict[str, Any]] = None


class CardinalitySpec(BaseModel):
    distribution: str = "poisson"
    params: Dict[str, Any] = Field(default_factory=lambda: {"mu": 3.0})
    min_children: int = 0
    max_children: Optional[int] = None


class ConditionalCorrelation(BaseModel):
    parent_column: str
    child_column: str
    type: str = "shift"
    params: Dict[str, Any] = Field(default_factory=dict)


class TemporalConstraint(BaseModel):
    rule: str


class RelationshipSpec(BaseModel):
    parent: str
    child: str
    parent_key: str
    child_key: str
    cardinality: Union[CardinalitySpec, Dict[str, Any]] = Field(default_factory=CardinalitySpec)
    participation: float = 1.0
    conditional_correlations: List[Union[ConditionalCorrelation, Dict[str, Any]]] = Field(default_factory=list)
    temporal: Optional[Dict[str, Any]] = None
    # Per-group child marginals keyed by parent column values.
    # { "parent_columns": [...], "child_targets": [...], "method": "group_empirical",
    #   "group_marginals": { "<parent_val>": { "<child_col>": {generation dict} } } }
    parent_conditioning: Optional[Dict[str, Any]] = None


class GenerationSpec(BaseModel):
    # version "1.0" is accepted; empirical extensions are additive and optional.
    version: str = "1.0"
    tables: List[TableSpec] = Field(default_factory=list)
    relationships: List[RelationshipSpec] = Field(default_factory=list)


class SpecConflict(BaseModel):
    table: Optional[str] = None
    column: Optional[str] = None
    message: str
    severity: Literal["error", "warning"] = "error"


class ValidationResult(BaseModel):
    valid: bool
    conflicts: List[SpecConflict] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
