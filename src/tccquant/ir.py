from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    initializers: dict[str, object] = field(default_factory=dict)


@dataclass
class Model:
    graph: Graph
