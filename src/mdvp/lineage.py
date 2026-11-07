"""Lineage graph: dataset version -> training run -> model version.

Nodes are string ids (dataset versions, training runs, model
versions). Edges carry a kind (``derived_from``, ``trained_on``,
``produced``). Column-level lineage maps
``(dataset version id, column)`` to the set of
``(source version id, source column)`` pairs it derives from, so a
single column can be traced back to its raw inputs. ``impact()``
walks downstream (BFS) to answer "what breaks if this version
changes?".
"""
from __future__ import annotations

from collections import deque

from mdvp.models import TrainingRun, TransformStep

Node = str


class LineageGraph:
    """Directed lineage graph with column-level provenance."""

    def __init__(self) -> None:
        self.edges: dict[Node, list[tuple[Node, str]]] = {}
        self.column_lineage: dict[tuple[str, str], set[tuple[str, str]]] = {}

    def add_edge(self, source: Node, target: Node, kind: str) -> None:
        self.edges.setdefault(source, [])
        if (target, kind) not in self.edges[source]:
            self.edges[source].append((target, kind))
        self.edges.setdefault(target, [])

    def record_transform(
        self, output_version_id: str, step: TransformStep
    ) -> None:
        """Record one transform step's version- and column-level lineage."""
        for input_id in step.input_version_ids:
            self.add_edge(input_id, output_version_id, "derived_from")
        for out_col, sources in step.column_lineage.items():
            key = (output_version_id, out_col)
            self.column_lineage.setdefault(key, set()).update(sources)

    def record_training_run(self, run: TrainingRun) -> None:
        """Link dataset version -> training run -> model version."""
        self.add_edge(run.dataset_version_id, run.run_id, "trained_on")
        self.add_edge(run.run_id, run.model_version, "produced")
        for step in run.transform_chain:
            for input_id in step.input_version_ids:
                self.add_edge(input_id, run.run_id, "trained_on")

    def column_sources(self, version_id: str, column: str) -> set[tuple[str, str]]:
        """Direct sources of one output column."""
        return set(self.column_lineage.get((version_id, column), set()))

    def column_origin(self, version_id: str, column: str) -> set[tuple[str, str]]:
        """Transitive raw sources of one column (walk the lineage map)."""
        origins: set[tuple[str, str]] = set()
        queue: deque[tuple[str, str]] = deque([(version_id, column)])
        seen: set[tuple[str, str]] = set()
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            sources = self.column_lineage.get(node, set())
            if not sources:
                origins.add(node)
            else:
                queue.extend(sources)
        return origins

    def impact(self, version_id: str) -> list[Node]:
        """All downstream nodes (BFS): derived versions, runs, models."""
        seen: set[Node] = set()
        queue: deque[Node] = deque([version_id])
        while queue:
            node = queue.popleft()
            for target, _kind in self.edges.get(node, []):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return sorted(seen)

    def upstream(self, node: Node) -> list[Node]:
        """All upstream nodes (BFS over reversed edges)."""
        reverse: dict[Node, list[Node]] = {}
        for source, targets in self.edges.items():
            for target, _kind in targets:
                reverse.setdefault(target, []).append(source)
        seen: set[Node] = set()
        queue: deque[Node] = deque([node])
        while queue:
            current = queue.popleft()
            for source in reverse.get(current, []):
                if source not in seen:
                    seen.add(source)
                    queue.append(source)
        return sorted(seen)
