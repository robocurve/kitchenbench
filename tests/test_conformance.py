"""The mock embodiment must pass the Inspect Robots adapter conformance kit.

This is the one-test CI enforcement the adapter authoring guide recommends:
conformant declarations are what keep the embodiment guardrail-ready and
agent-ready (dim-labeled tooling, derived safety limits).
"""

from inspect_robots.conformance import assert_embodiment_conformant

from kitchenbench.embodiment import KitchenEmbodiment


def test_kitchen_embodiment_is_conformant() -> None:
    embodiment = KitchenEmbodiment()
    try:
        assert_embodiment_conformant(embodiment.info)
    finally:
        embodiment.close()
