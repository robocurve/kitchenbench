"""Task instances — the unit of the physical-automation methodology.

A **task instance** is a self-contained scenario: a *stochastic environment
setup* (named :class:`~kitchenbench.distributions.Distribution` variables) plus a
*goal* (a natural-language success criterion that may reference sampled
variables). A **realization** samples every variable to produce one concrete
environment; the methodology runs ``K_REALIZATIONS`` realizations of each of
``K_INSTANCES`` instances per task and estimates the instance success probability
as the mean binary outcome.

This module is pure (no RoboLens import); :mod:`kitchenbench.tasks` maps instances
onto RoboLens ``Scene``/``Epochs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kitchenbench.distributions import Distribution, Scalar

#: Methodology recommendations (see the reference docs).
K_REALIZATIONS = 5  # independent rollouts per instance -> success probability
K_INSTANCES = 5  # validated instances per task
K_EXPERTS = 3  # validators per instance


@dataclass(frozen=True)
class Validation:
    """Per-expert validation ratings carried with an instance.

    Empty until the human commissioning pipeline is run; :attr:`validated` follows
    the methodology accept rule (``K_EXPERTS`` ratings, all >= 4 on both axes).
    """

    representativeness: tuple[int, ...] = ()
    quality: tuple[int, ...] = ()
    difficulty: tuple[int, ...] = ()  # optional ("nice to have" in the methodology)
    source: str = "opus-draft"  # provenance: "opus-draft" (AI) vs "human"

    @property
    def validated(self) -> bool:
        return (
            len(self.representativeness) >= K_EXPERTS
            and len(self.quality) >= K_EXPERTS
            and all(r >= 4 for r in self.representativeness)
            and all(q >= 4 for q in self.quality)
        )


@dataclass(frozen=True)
class Realization:
    """One concrete environment sampled from a :class:`TaskInstance`."""

    seed: int
    values: dict[str, Scalar]
    instruction: str
    setup_lines: tuple[str, ...]


@dataclass(frozen=True)
class TaskInstance:
    """A stochastic scenario: named distributions + a goal."""

    instance_id: str
    goal: str  # template; ``{var}`` placeholders must be in ``language_vars``
    setup: dict[str, Distribution]
    target_kind: str
    language_vars: tuple[str, ...] = ()
    static: dict[str, Any] = field(default_factory=dict)
    validation: Validation = field(default_factory=Validation)

    def realize(self, seed: int) -> Realization:
        """Sample every setup variable (sorted-key order) for one concrete environment."""
        rng = np.random.default_rng(seed)
        values: dict[str, Scalar] = {key: self.setup[key].sample(rng) for key in sorted(self.setup)}
        instruction = self.goal.format(**{k: values[k] for k in self.language_vars})
        setup_lines = tuple(f"{key} = {values[key]}" for key in sorted(self.setup))
        return Realization(
            seed=seed, values=values, instruction=instruction, setup_lines=setup_lines
        )

    def setup_spec(self) -> dict[str, str]:
        """A JSON-native description of the setup (``{var: distribution.describe()}``)."""
        return {key: self.setup[key].describe() for key in sorted(self.setup)}
