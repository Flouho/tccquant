import random

from tccquant.config import QuantGranularity, QuantScheme, QuantSpec
from tccquant.quant_math import calc_scale_zero_point


def _rand_2d(h, w):
    return [[random.uniform(-1, 1) for _ in range(w)] for _ in range(h)]


def test_quant_math_supports_all_granularity_modes():
    x = _rand_2d(8, 8)
    specs = [
        QuantSpec(bits=8, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.SYMMETRIC),
        QuantSpec(bits=8, granularity=QuantGranularity.PER_CHANNEL, scheme=QuantScheme.ASYMMETRIC, axis=0),
        QuantSpec(bits=4, granularity=QuantGranularity.PER_GROUP, scheme=QuantScheme.SYMMETRIC, axis=0, group_size=4),
        QuantSpec(bits=8, granularity=QuantGranularity.PER_BLOCK, scheme=QuantScheme.SYMMETRIC, block_size=(4, 4)),
    ]

    for spec in specs:
        scale, zp = calc_scale_zero_point(x, spec)
        assert all(s > 0 for s in scale)
        assert all(isinstance(v, int) for v in zp)
