import pytest

from tccquant.exporter import export_model_to_onnx
from tccquant.ir import Graph, Model
from tccquant.onnx_parser import OnnxDependencyError


def test_export_onnx_requires_onnx_dependency(tmp_path):
    model = Model(graph=Graph(nodes=[], initializers={}, inputs=["x"], outputs=["y"]))
    out = tmp_path / "quant.onnx"
    with pytest.raises(OnnxDependencyError):
        export_model_to_onnx(model, str(out))
