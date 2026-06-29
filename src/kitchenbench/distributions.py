"""Probability distributions for KitchenBench task-instance setups.

A task instance (see :mod:`kitchenbench.instances`) specifies a *stochastic*
environment: each setup variable is a :class:`Distribution`. A *realization*
samples every variable to produce one concrete environment. These types are pure
NumPy with no RoboLens dependency.

Distributions mirror the methodology's notation
(:doc:`reference/physical-automation-methodology-docs`):

- :class:`Uniform` — ``Uniform[a, b]`` (continuous)
- :class:`Categorical` — ``Categorical({…})`` (uniform or weighted over a finite set)
- :class:`Normal` — ``N(μ, σ²)`` (Gaussian; a 2-D jitter is two of these)
- :class:`Constant` — a fixed (non-random) part of a setup

Every ``sample`` returns a **builtin** ``float``/``int``/``str`` (never a NumPy
scalar) so realizations are JSON-native and mypy-strict clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

Scalar = float | int | str


@runtime_checkable
class Distribution(Protocol):
    """A sampleable, self-describing setup variable."""

    def sample(self, rng: np.random.Generator) -> Any: ...

    def describe(self) -> str: ...


@dataclass(frozen=True)
class Uniform:
    """Continuous uniform on ``[low, high]``."""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def describe(self) -> str:
        return f"Uniform[{_num(self.low)}, {_num(self.high)}]"


@dataclass(frozen=True)
class Categorical:
    """Uniform (or weighted) choice over a finite set of values.

    Samples **by index** so the value's type is preserved — ``rng.choice`` over a
    numeric/mixed tuple would coerce to a string array.
    """

    values: tuple[Scalar, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Categorical needs at least one value")
        if self.weights is not None and len(self.weights) != len(self.values):
            raise ValueError("weights must match values in length")

    def sample(self, rng: np.random.Generator) -> Scalar:
        probs = None if self.weights is None else np.asarray(self.weights, dtype=np.float64)
        if probs is not None:
            probs = probs / probs.sum()
        idx = int(rng.choice(len(self.values), p=probs))
        return self.values[idx]

    def describe(self) -> str:
        body = ", ".join(_num(v) if isinstance(v, int | float) else str(v) for v in self.values)
        return f"Categorical({{{body}}})"


@dataclass(frozen=True)
class Normal:
    """Gaussian ``N(mean, std²)``."""

    mean: float
    std: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.std))

    def describe(self) -> str:
        return f"N({_num(self.mean)}, {_num(self.std)}²)"


@dataclass(frozen=True)
class Constant:
    """A fixed, non-random setup value."""

    value: Scalar

    def sample(self, rng: np.random.Generator) -> Scalar:
        return self.value

    def describe(self) -> str:
        return repr(self.value)


def _num(x: float) -> str:
    """Render a number without a trailing ``.0`` for whole values."""
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
