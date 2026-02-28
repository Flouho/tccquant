import json
import subprocess
import sys


def test_e2e_script_reports_missing_onnx_dependency(tmp_path):
    model = tmp_path / "dummy.onnx"
    calib = tmp_path / "calib.json"
    report = tmp_path / "report.json"
    qonnx = tmp_path / "quant.onnx"

    model.write_bytes(b"dummy")
    calib.write_text(json.dumps({"x": [[0.1, -0.2]]}))

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/e2e_quant_test.py",
            "--model",
            str(model),
            "--calibration-json",
            str(calib),
            "--error-report",
            str(report),
            "--quantized-onnx",
            str(qonnx),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "onnx dependency is required" in proc.stderr
