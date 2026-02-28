from __future__ import annotations

from dataclasses import dataclass

from .config import QuantGranularity, QuantScheme, QuantSpec


@dataclass(frozen=True)
class QuantPolicy:
    name: str
    weight: QuantSpec
    activation: QuantSpec


PRESET_POLICIES: dict[str, QuantPolicy] = {
    "W4A16": QuantPolicy(
        name="W4A16",
        weight=QuantSpec(bits=4, granularity=QuantGranularity.PER_CHANNEL, scheme=QuantScheme.SYMMETRIC, axis=0),
        activation=QuantSpec(bits=16, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.ASYMMETRIC),
    ),
    "W4A8": QuantPolicy(
        name="W4A8",
        weight=QuantSpec(bits=4, granularity=QuantGranularity.PER_GROUP, scheme=QuantScheme.SYMMETRIC, group_size=64),
        activation=QuantSpec(bits=8, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.ASYMMETRIC),
    ),
    "W8A8": QuantPolicy(
        name="W8A8",
        weight=QuantSpec(bits=8, granularity=QuantGranularity.PER_CHANNEL, scheme=QuantScheme.SYMMETRIC, axis=0),
        activation=QuantSpec(bits=8, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.ASYMMETRIC),
    ),
    "W8A16": QuantPolicy(
        name="W8A16",
        weight=QuantSpec(bits=8, granularity=QuantGranularity.PER_BLOCK, scheme=QuantScheme.SYMMETRIC, block_size=(32, 32)),
        activation=QuantSpec(bits=16, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.ASYMMETRIC),
    ),
}


def policy_from_name(name: str) -> QuantPolicy:
    key = name.upper()
    if key not in PRESET_POLICIES:
        raise KeyError(f"Unknown policy: {name}. Available: {', '.join(PRESET_POLICIES)}")
    return PRESET_POLICIES[key]
