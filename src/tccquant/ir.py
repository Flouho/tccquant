from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, object] = field(default_factory=dict)


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    initializers: dict[str, object] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


@dataclass
class Model:
    graph: Graph


@dataclass
class TensorQuantState:
    scale: list[float]
    zero_point: list[int]
    bitwidth: int
    granularity: str
    scheme: str


@dataclass
class OpError:
    op_name: str
    op_type: str
    mse: float
    mae: float
    max_abs_error: float
