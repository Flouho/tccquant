import json
import random

from tccquant.config import QuantGranularity, QuantScheme, QuantSpec
from tccquant.graph_editor import NodeAnchor, QDQGraphEditor
from tccquant.ir import Graph, Model, Node


def _rand_2d(h, w):
    return [[random.uniform(-1, 1) for _ in range(w)] for _ in range(h)]


def _build_small_model() -> Model:
    return Model(
        graph=Graph(
            nodes=[
                Node(name="MatMul_0", op_type="MatMul", inputs=["x", "w"], outputs=["mm_out"]),
                Node(name="Relu_0", op_type="Relu", inputs=["mm_out"], outputs=["y"]),
            ],
            initializers={"w": _rand_2d(4, 4)},
        )
    )


def _build_llm_like_model() -> Model:
    return Model(
        graph=Graph(
            nodes=[
                Node(name="QProj", op_type="MatMul", inputs=["x", "wq"], outputs=["q_out"]),
                Node(name="Act", op_type="Gelu", inputs=["q_out"], outputs=["a_out"]),
                Node(name="OProj", op_type="MatMul", inputs=["a_out", "wo"], outputs=["y"]),
            ],
            initializers={"wq": _rand_2d(16, 16), "wo": _rand_2d(16, 16)},
        )
    )


def _save_model(path, model: Model):
    payload = {
        "nodes": [
            {"name": n.name, "op_type": n.op_type, "inputs": n.inputs, "outputs": n.outputs}
            for n in model.graph.nodes
        ],
        "initializers": model.graph.initializers,
    }
    path.write_text(json.dumps(payload))


def test_e2e_small_model_insert_and_remove_qdq(tmp_path):
    model = _build_small_model()
    p = tmp_path / "small.json"
    _save_model(p, model)

    editor = QDQGraphEditor.from_path(str(p))
    spec = QuantSpec(bits=8, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.ASYMMETRIC)
    q_name, dq_name = editor.insert_qdq_after(NodeAnchor("MatMul_0"), _rand_2d(4, 4), spec)
    assert any(n.name == q_name for n in editor.model.graph.nodes)
    assert any(n.name == dq_name for n in editor.model.graph.nodes)

    editor.remove_qdq_pair(q_name, dq_name)
    assert not any(n.op_type == "QuantizeLinear" for n in editor.model.graph.nodes)
    assert not any(n.op_type == "DequantizeLinear" for n in editor.model.graph.nodes)


def test_e2e_llm_like_dual_insertion(tmp_path):
    model = _build_llm_like_model()
    p = tmp_path / "llm.json"
    _save_model(p, model)

    editor = QDQGraphEditor.from_path(str(p))
    spec_w4 = QuantSpec(bits=4, granularity=QuantGranularity.PER_GROUP, scheme=QuantScheme.SYMMETRIC, axis=0, group_size=8)
    spec_a8 = QuantSpec(bits=8, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.ASYMMETRIC)

    editor.insert_qdq_after(NodeAnchor("QProj"), _rand_2d(1, 16), spec_a8, prefix="qproj_act")
    editor.insert_qdq_after(NodeAnchor("OProj"), _rand_2d(16, 16), spec_w4, prefix="oproj_weight")

    qdq_count = sum(n.op_type in {"QuantizeLinear", "DequantizeLinear"} for n in editor.model.graph.nodes)
    assert qdq_count == 4

    editor.strip_all_qdq()
    qdq_count = sum(n.op_type in {"QuantizeLinear", "DequantizeLinear"} for n in editor.model.graph.nodes)
    assert qdq_count == 0
