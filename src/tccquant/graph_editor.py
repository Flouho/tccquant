from __future__ import annotations

from dataclasses import dataclass
import json

from .config import QuantSpec, QuantScheme
from .ir import Model, Graph, Node
from .quant_math import calc_scale_zero_point


@dataclass
class NodeAnchor:
    node_name: str
    output_index: int = 0


class QDQGraphEditor:
    def __init__(self, model: Model):
        self.model = model

    @classmethod
    def from_path(cls, path: str) -> "QDQGraphEditor":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        nodes = [Node(**n) for n in payload["nodes"]]
        graph = Graph(nodes=nodes, initializers=payload.get("initializers", {}))
        return cls(Model(graph=graph))

    def save(self, path: str) -> None:
        payload = {
            "nodes": [
                {"name": n.name, "op_type": n.op_type, "inputs": n.inputs, "outputs": n.outputs}
                for n in self.model.graph.nodes
            ],
            "initializers": self.model.graph.initializers,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _find_node(self, node_name: str) -> Node:
        for node in self.model.graph.nodes:
            if node.name == node_name:
                return node
        raise KeyError(f"Node not found: {node_name}")

    def insert_qdq_after(self, anchor: NodeAnchor, calibration_data, spec: QuantSpec, prefix: str | None = None) -> tuple[str, str]:
        node = self._find_node(anchor.node_name)
        out_name = node.outputs[anchor.output_index]
        qdq_prefix = prefix or f"{node.name}_qdq"

        scale, zp = calc_scale_zero_point(calibration_data, spec)

        scale_name = f"{qdq_prefix}_scale"
        zp_name = f"{qdq_prefix}_zero_point"
        q_out = f"{qdq_prefix}_q"
        dq_out = f"{qdq_prefix}_dq"

        zp_dtype = "uint8" if spec.scheme == QuantScheme.ASYMMETRIC else "int8"
        self.model.graph.initializers[scale_name] = scale
        self.model.graph.initializers[zp_name] = {"dtype": zp_dtype, "value": zp}

        q_node = Node(
            name=f"{qdq_prefix}_QuantizeLinear",
            op_type="QuantizeLinear",
            inputs=[out_name, scale_name, zp_name],
            outputs=[q_out],
        )
        dq_node = Node(
            name=f"{qdq_prefix}_DequantizeLinear",
            op_type="DequantizeLinear",
            inputs=[q_out, scale_name, zp_name],
            outputs=[dq_out],
        )

        for consumer in self.model.graph.nodes:
            if consumer.name in (q_node.name, dq_node.name):
                continue
            consumer.inputs = [dq_out if inp == out_name else inp for inp in consumer.inputs]

        insert_idx = self.model.graph.nodes.index(node) + 1
        self.model.graph.nodes.insert(insert_idx, q_node)
        self.model.graph.nodes.insert(insert_idx + 1, dq_node)
        return q_node.name, dq_node.name

    def remove_qdq_pair(self, q_node_name: str, dq_node_name: str) -> None:
        q_node = self._find_node(q_node_name)
        dq_node = self._find_node(dq_node_name)
        src = q_node.inputs[0]
        dq_out = dq_node.outputs[0]
        for consumer in self.model.graph.nodes:
            consumer.inputs = [src if inp == dq_out else inp for inp in consumer.inputs]

        self.model.graph.nodes = [n for n in self.model.graph.nodes if n.name not in {q_node_name, dq_node_name}]

        removable = set(q_node.inputs[1:] + dq_node.inputs[1:])
        still_used = {inp for node in self.model.graph.nodes for inp in node.inputs}
        for name in removable:
            if name not in still_used and name in self.model.graph.initializers:
                del self.model.graph.initializers[name]

    def strip_all_qdq(self) -> None:
        pairs = []
        for node in self.model.graph.nodes:
            if node.op_type == "QuantizeLinear":
                for n2 in self.model.graph.nodes:
                    if n2.op_type == "DequantizeLinear" and n2.inputs[0] == node.outputs[0]:
                        pairs.append((node.name, n2.name))
        for q_name, dq_name in pairs:
            self.remove_qdq_pair(q_name, dq_name)
