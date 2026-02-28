import pytest

from tccquant.onnx_parser import OnnxDependencyError, parse_onnx_model


def test_onnx_parser_raises_dependency_error_without_onnx(tmp_path):
    dummy = tmp_path / "dummy.onnx"
    dummy.write_bytes(b"not a real onnx")
    with pytest.raises(OnnxDependencyError):
        parse_onnx_model(str(dummy))
