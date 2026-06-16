from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


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
    semantic_type: Literal["id", "name", "email", "address", "currency", "category", "date", "phone", "none"] = "none"
    generation: Dict[str, Any] = Field(default_factory=dict)
    constraints: ColumnConstraints = Field(default_factory=ColumnConstraints)


class IntraRowConstraint(BaseModel):
    type: Literal["arithmetic", "conditional"]
    rule: str


class TableSpec(BaseModel):
    name: str
    row_count: int
    primary_key: Optional[str] = None
    columns: List[ColumnSpec] = Field(default_factory=list)
    intra_row_constraints: List[IntraRowConstraint] = Field(default_factory=list)


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


class GenerationSpec(BaseModel):
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
