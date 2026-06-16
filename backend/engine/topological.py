from __future__ import annotations
from typing import Dict, List, Set, Tuple
from backend.spec.models import GenerationSpec


def build_dependency_graph(spec: GenerationSpec) -> Dict[str, Set[str]]:
    """Build a map of table → set of tables it depends on (via FK relationships)."""
    deps: Dict[str, Set[str]] = {t.name: set() for t in spec.tables}
    for rel in spec.relationships:
        if rel.child in deps:
            deps[rel.child].add(rel.parent)
    return deps


def topological_sort(spec: GenerationSpec) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Return (ordered_table_names, deferred_circular_pairs).
    Tables are ordered parents-first. Circular FKs are detected and returned
    as deferred pairs to be backfilled after initial generation.
    """
    deps = build_dependency_graph(spec)
    table_names = [t.name for t in spec.tables]
    
    visited: Set[str] = set()
    in_stack: Set[str] = set()
    order: List[str] = []
    circular: List[Tuple[str, str]] = []

    def visit(name: str, path: List[str]) -> None:
        if name in visited:
            return
        if name in in_stack:
            cycle_start = path.index(name) if name in path else len(path) - 1
            cycle = path[cycle_start:] + [name]
            for i in range(len(cycle) - 1):
                pair = (cycle[i], cycle[i + 1])
                if pair not in circular:
                    circular.append(pair)
            return
        in_stack.add(name)
        for dep in deps.get(name, set()):
            visit(dep, path + [name])
        in_stack.discard(name)
        visited.add(name)
        order.append(name)

    for name in table_names:
        visit(name, [])

    deferred: List[Tuple[str, str]] = []
    circular_set = set(circular)
    for parent, child in circular_set:
        deferred.append((parent, child))

    return order, deferred


def get_parent_relationships(spec: GenerationSpec, table_name: str):
    """Return relationships where table_name is the child."""
    return [r for r in spec.relationships if r.child == table_name]


def get_child_relationships(spec: GenerationSpec, table_name: str):
    """Return relationships where table_name is the parent."""
    return [r for r in spec.relationships if r.parent == table_name]
