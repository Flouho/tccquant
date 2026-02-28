# TCCQuant

面向客户交付的量化工程（基于 PPQ 风格 Pass 机制），支持：

- 删除 PPQ 原工程中的非必要文件（可配置）
- 解析 ONNX 模型并转换为内部 IR
- 在图指定位置插入/删除 Q-DQ 节点
- 量化粒度：per-tensor / per-channel / per-group / per-block
- 对称 / 非对称量化
- 预设策略：W4A16、W4A8、W8A8、W8A16
- 基于 calibration data 的完整 W8A8 量化流程（weight per-channel + activation per-tensor）
- 全网 OP 级量化误差分析（MSE / MAE / MaxAbs）

## PPQ 风格 Pass 流程

- `CalibrationPass`：统计 calibration tensor 范围
- `W8A8QuantPass`：执行 W8A8 量化参数生成
- `ErrorAnalysisPass`：逐 OP 计算量化误差

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
pytest -q
```

## CLI

- 删除 PPQ 冗余文件（演练）：

```bash
python scripts/tccquant_cli.py prune-ppq /path/to/ppq --dry-run
```

- 指定节点后插入 Q-DQ（IR/JSON）：

```bash
python scripts/tccquant_cli.py insert-qdq \
  --model model.json \
  --output model_qdq.json \
  --node MatMul_0 \
  --bits 8 \
  --granularity per-channel \
  --scheme symmetric \
  --axis 0
```

- 执行 W8A8 量化+误差分析（ONNX）：

```bash
python scripts/tccquant_cli.py run-w8a8 \
  --model model.onnx \
  --calibration-json calibration.json \
  --report error_report.json
```

## 目录

- `src/tccquant/onnx_parser.py`：ONNX 解析
- `src/tccquant/passes.py`：PPQ 风格 pass 机制
- `src/tccquant/pipeline.py`：W8A8 管线与误差报告
- `src/tccquant/config.py`：量化配置
- `src/tccquant/policies.py`：预设量化策略
- `src/tccquant/quant_math.py`：scale/zero-point、fake quant 和误差计算
- `src/tccquant/graph_editor.py`：Q-DQ 图编辑
- `src/tccquant/ppq_cleanup.py`：PPQ 冗余文件清理
- `tests/`：基础与端到端测试
