from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantGranularity(str, Enum):
    PER_TENSOR = "per-tensor"
    PER_CHANNEL = "per-channel"
    PER_GROUP = "per-group"
    PER_BLOCK = "per-block"


class QuantScheme(str, Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    granularity: QuantGranularity = QuantGranularity.PER_TENSOR
    scheme: QuantScheme = QuantScheme.SYMMETRIC
    axis: int = 0
    group_size: int | None = None
    block_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.bits not in (4, 8, 16):
            raise ValueError(f"Unsupported bitwidth: {self.bits}")
        if self.granularity == QuantGranularity.PER_GROUP and not self.group_size:
            raise ValueError("group_size is required for per-group quantization")
        if self.granularity == QuantGranularity.PER_BLOCK and not self.block_size:
            raise ValueError("block_size is required for per-block quantization")
