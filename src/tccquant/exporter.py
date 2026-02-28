from __future__ import annotations

from .ir import Model
from .onnx_parser import OnnxDependencyError


def _infer_shape(data):
    if isinstance(data, list):
        if not data:
            return [0]
        return [len(data)] + _infer_shape(data[0])
    return []


def _flatten(data):
    if isinstance(data, list):
        out = []
        for x in data:
            out.extend(_flatten(x))
        return out
    return [float(data)]


def export_model_to_onnx(model: Model, output_path: str, opset: int = 13) -> None:
    """Export internal IR model to ONNX.

    Quantized ops such as QuantLinear / QuantRMSNorm are exported as custom ops in domain `tccquant`.
    """
    try:
        import onnx  # type: ignore
        from onnx import TensorProto, helper  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise OnnxDependencyError("onnx dependency is required to export ONNX models") from exc

    inits = []
    value_infos = {}
    for name, val in model.graph.initializers.items():
        shape = _infer_shape(val)
        vals = _flatten(val)
        init = helper.make_tensor(name=name, data_type=TensorProto.FLOAT, dims=shape, vals=vals)
        inits.append(init)
        value_infos[name] = helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    nodes = []
    for node in model.graph.nodes:
        attrs = node.attrs or {}
        op_domain = "tccquant" if node.op_type.startswith("Quant") else ""
        onnx_node = helper.make_node(
            node.op_type,
            inputs=node.inputs,
            outputs=node.outputs,
            name=node.name,
            domain=op_domain,
            **attrs,
        )
        nodes.append(onnx_node)

    graph_inputs = []
    for name in model.graph.inputs:
        graph_inputs.append(value_infos.get(name, helper.make_tensor_value_info(name, TensorProto.FLOAT, None)))

    graph_outputs = []
    for name in model.graph.outputs:
        graph_outputs.append(value_infos.get(name, helper.make_tensor_value_info(name, TensorProto.FLOAT, None)))

    graph = helper.make_graph(nodes, "tccquant_export", graph_inputs, graph_outputs, initializer=inits)
    opsets = [helper.make_opsetid("", opset), helper.make_opsetid("tccquant", 1)]
    model_proto = helper.make_model(graph, producer_name="tccquant", opset_imports=opsets)
    onnx.save(model_proto, output_path)
