from __future__ import annotations

from .ir import Graph, Model, Node


class OnnxDependencyError(RuntimeError):
    pass


def parse_onnx_model(path: str) -> Model:
    """Parse ONNX file into lightweight IR.

    Requires `onnx` package at runtime.
    """
    try:
        import onnx  # type: ignore
        from onnx import numpy_helper  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by dedicated dependency test
        raise OnnxDependencyError("onnx dependency is required to parse ONNX models") from exc

    model = onnx.load(path)
    graph = model.graph
    nodes = []
    for idx, node in enumerate(graph.node):
        name = node.name or f"{node.op_type}_{idx}"
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        nodes.append(Node(name=name, op_type=node.op_type, inputs=list(node.input), outputs=list(node.output), attrs=attrs))

    inits = {}
    for init in graph.initializer:
        inits[init.name] = numpy_helper.to_array(init).tolist()

    g = Graph(
        nodes=nodes,
        initializers=inits,
        inputs=[i.name for i in graph.input],
        outputs=[o.name for o in graph.output],
    )
    return Model(graph=g)
