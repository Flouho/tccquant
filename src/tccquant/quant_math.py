from __future__ import annotations

from .config import QuantGranularity, QuantScheme, QuantSpec


def _is_number(x) -> bool:
    return isinstance(x, (int, float))


def flatten(data):
    if _is_number(data):
        return [float(data)]
    out = []
    for x in data:
        out.extend(flatten(x))
    return out


def shape2d(data):
    rows = len(data)
    cols = len(data[0]) if rows else 0
    return rows, cols


def per_channel_min_max_2d(data, axis):
    rows, cols = shape2d(data)
    if axis == 0:
        mins = [min(data[r][c] for r in range(rows)) for c in range(cols)]
        maxs = [max(data[r][c] for r in range(rows)) for c in range(cols)]
    else:
        mins = [min(row) for row in data]
        maxs = [max(row) for row in data]
    return mins, maxs


def group_extrema_2d(data, axis, group_size):
    mins, maxs = per_channel_min_max_2d(data, axis)
    grouped_min, grouped_max = [], []
    for i in range(0, len(mins), group_size):
        grouped_min.append(min(mins[i : i + group_size]))
        grouped_max.append(max(maxs[i : i + group_size]))
    return grouped_min, grouped_max


def block_extrema_2d(data, block_size):
    bh, bw = block_size
    rows, cols = shape2d(data)
    mins, maxs = [], []
    for r in range(0, rows, bh):
        row_mins, row_maxs = [], []
        for c in range(0, cols, bw):
            block = [data[i][j] for i in range(r, min(r + bh, rows)) for j in range(c, min(c + bw, cols))]
            row_mins.append(min(block))
            row_maxs.append(max(block))
        mins.append(row_mins)
        maxs.append(row_maxs)
    return mins, maxs


def _to_list(x):
    return x if isinstance(x, list) else [x]


def calc_scale_zero_point(data, spec: QuantSpec):
    qmax = 2**spec.bits - 1
    if spec.scheme == QuantScheme.SYMMETRIC:
        qmin = -(2 ** (spec.bits - 1))
        qmax_signed = 2 ** (spec.bits - 1) - 1
    else:
        qmin = 0
        qmax_signed = qmax

    if spec.granularity == QuantGranularity.PER_TENSOR:
        flat = flatten(data)
        dmin, dmax = min(flat), max(flat)
    elif spec.granularity == QuantGranularity.PER_CHANNEL:
        dmin, dmax = per_channel_min_max_2d(data, spec.axis)
    elif spec.granularity == QuantGranularity.PER_GROUP:
        dmin, dmax = group_extrema_2d(data, spec.axis, int(spec.group_size))
    elif spec.granularity == QuantGranularity.PER_BLOCK:
        dmin, dmax = block_extrema_2d(data, tuple(spec.block_size))
    else:
        raise ValueError(f"Unknown granularity: {spec.granularity}")

    dmin_list, dmax_list = flatten(_to_list(dmin)), flatten(_to_list(dmax))
    scales, zps = [], []
    for mn, mx in zip(dmin_list, dmax_list):
        if spec.scheme == QuantScheme.SYMMETRIC:
            absmax = max(abs(mn), abs(mx))
            scale = max(absmax / qmax_signed, 1e-8)
            zp = 0
        else:
            scale = max((mx - mn) / (qmax_signed - qmin), 1e-8)
            zp = int(round(qmin - mn / scale))
            zp = max(qmin, min(qmax_signed, zp))
        scales.append(scale)
        zps.append(zp)
    return scales, zps


def fake_quant_dequant_per_tensor(data, scale: float, zp: int, bits: int, symmetric: bool):
    if symmetric:
        qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    else:
        qmin, qmax = 0, 2**bits - 1

    def _q(x):
        q = int(round(x / scale + zp))
        q = max(qmin, min(qmax, q))
        return (q - zp) * scale

    if _is_number(data):
        return _q(float(data))
    return [fake_quant_dequant_per_tensor(x, scale, zp, bits, symmetric) for x in data]


def tensor_error(original, quantized):
    o = flatten(original)
    q = flatten(quantized)
    if len(o) != len(q) or not o:
        return 0.0, 0.0, 0.0
    abs_err = [abs(a - b) for a, b in zip(o, q)]
    mse = sum((a - b) ** 2 for a, b in zip(o, q)) / len(o)
    mae = sum(abs_err) / len(o)
    max_abs = max(abs_err)
    return mse, mae, max_abs
