from __future__ import annotations
import json
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_anthropic_client = None


def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


SPEC_SCHEMA_SUMMARY = """
GenerationSpec:
  version: str
  tables: List[TableSpec]
  relationships: List[RelationshipSpec]

TableSpec:
  name: str
  row_count: int (>0)
  primary_key: str | null
  columns: List[ColumnSpec]
  intra_row_constraints: List[{type: "arithmetic"|"conditional", rule: str}]

ColumnSpec:
  name: str
  type: "integer"|"float"|"string"|"categorical"|"datetime"|"boolean"
  semantic_type: "id"|"name"|"email"|"address"|"currency"|"category"|"date"|"phone"|"none"
  generation: {
    strategy: "sequential"|"numeric_distribution"|"categorical_sample"|"datetime_range"|"regex_pattern"|"semantic"|"derived"|"foreign_key",
    params: {}
  }
  constraints: {unique: bool, nullable: bool, min: number|null, max: number|null}

Generation strategy params:
- sequential: {start: int, step: int}
- numeric_distribution: {dist: "normal"|"lognormal"|"uniform"|"skewnormal", mean: float, std: float, low: float, high: float, a: float}
- categorical_sample: {values: {value: probability}} (probs must sum to 1.0)
- datetime_range: {start: "ISO", end: "ISO", format: "%Y-%m-%d"}
- regex_pattern: {pattern: str}
- semantic: {faker_method: str} (e.g. "name", "email", "address", "phone_number", "company")
- derived: {expression: str} (e.g. "qty * unit_price")
- foreign_key: {references_table: str, references_column: str}

RelationshipSpec:
  parent: str
  child: str
  parent_key: str
  child_key: str
  cardinality: {distribution: "poisson"|"negative_binomial"|"uniform", params: {mu: float}|{n:int,p:float}|{low:int,high:int}, min_children: int, max_children: int|null}
  participation: float (0-1)
  conditional_correlations: []
  temporal: null
"""


async def parse_intent(query: str) -> Dict[str, Any]:
    """
    Convert a natural-language data description into a GenerationSpec JSON.
    Returns the spec dict, validated structure.
    """
    client = _get_client()
    prompt = f"""You are a data engineering expert. Convert this natural-language data description into a valid GenerationSpec JSON.

SCHEMA:
{SPEC_SCHEMA_SUMMARY}

RULES:
1. Return ONLY valid JSON — no markdown, no explanation, no code fences
2. All probabilities in categorical_sample must sum to exactly 1.0
3. Row counts must be positive integers
4. If the user describes relationships, model them in the relationships array with a FK column in the child table using strategy "foreign_key"
5. Always add a primary key column with strategy "sequential" to each table
6. Infer sensible semantic types from column names (email→email, name→name, etc.)
7. For currency/salary columns use strategy "numeric_distribution" with a realistic mean/std
8. Gap-fill: if row count not specified, use 100-1000 based on context
9. The spec version must be "1.0"

USER REQUEST: {query}

Respond with ONLY the JSON GenerationSpec object:"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    
    return json.loads(raw)


async def semantic_pass(profile: Dict[str, Any], sample_rows: list) -> Dict[str, Any]:
    """
    Given a column profile and sample rows, label semantic types and infer rules.
    Returns a dict of {column_name: {semantic_type, faker_method, notes}}.
    """
    client = _get_client()
    
    sample_text = json.dumps(sample_rows[:5], default=str)
    profile_text = json.dumps(profile, default=str)
    
    prompt = f"""You are a data engineering expert. Given this column profile and sample data, label semantic types and Faker methods.

COLUMN PROFILES:
{profile_text}

SAMPLE ROWS (first 5):
{sample_text}

For each column, determine:
- semantic_type: one of "id"|"name"|"email"|"address"|"currency"|"category"|"date"|"phone"|"none"
- faker_method: the Faker method name if semantic (e.g. "name", "email", "address", "phone_number"), or null

Return ONLY a JSON object mapping column_name → {{semantic_type, faker_method}}:"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    
    return json.loads(raw)
