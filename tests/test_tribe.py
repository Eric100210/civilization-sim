"""
Calibration tests for Tribe behaviour.

These tests verify behavioural invariants rather than exact values,
since the simulation is stochastic. Each test documents an expected
demographic or territorial property that should hold across any seed.

Run with:  pytest tests/
"""

import pytest
import random
import numpy as np
from data.map import World
from data.tribe import Tribe
from data.resources import ResourceType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# TODO: create a small world (50x50) with a fixed seed and one tribe
# spawned on a known plain tile. Reuse across tests with @pytest.fixture.
@pytest.fixture
def world():
    # fix a determinist behaviour for randomness (for tests reproductability)
    random.seed(42)
    np.random.seed(42)
    size = 100
    w = World(width=size, height=size, seed=1090)
    w.generate(n_rivers=5, river_randomness=3)
    return w


@pytest.fixture
def plain_tribe(world):
    tribe = Tribe(world)
    x = 23
    y = 55
    tribe.territory = {(x, y)}
    habit = world.habitability_map[y, x]
    tribe.population = np.random.lognormal(mean=3.0 + 0.1 * habit, sigma=0.4)
    return tribe


# ---------------------------------------------------------------------------
# Territorial invariants
# ---------------------------------------------------------------------------


def test_territory_never_shrinks_after_sedentarisation():
    """Once sedentary, territory should only grow — expand() never removes tiles."""
    pass


def test_expand_never_adds_ocean_tile():
    """expand() must only add land tiles — never ocean or out-of-bounds."""
    pass


def test_expand_only_adds_adjacent_tiles():
    """Every new tile added by expand() must be adjacent (8-connected) to the existing territory."""
    pass


def test_border_tiles_are_all_adjacent_to_territory():
    """_border_tiles() should return only tiles that are direct neighbours of the territory."""
    pass


# ---------------------------------------------------------------------------
# Population invariants
# ---------------------------------------------------------------------------


def test_population_never_negative():
    """population should never go below 0 after eat() or population_growth()."""
    pass


def test_population_stabilises_on_good_land():
    """
    A tribe on fertile plains should stabilise between 200 and 5000 inhabitants
    after 300 years — not explode infinitely nor collapse to 1.
    """
    pass


def test_population_declines_on_bad_land():
    """
    A tribe forcibly placed on a desert tile (low food, low water) should
    decline or stay very small over 100 years.
    """
    pass


def test_eat_does_not_kill_tribe_instantly():
    """
    A single bad year (water=0) should not immediately trigger extinction —
    the population penalty should be proportional, not lethal in one step.
    """
    pass


# ---------------------------------------------------------------------------
# Resources invariants
# ---------------------------------------------------------------------------


def test_consumables_reset_each_year():
    """water and food should be reset to 0 at the start of each get_resources() call."""
    pass


def test_accumulables_never_decrease_without_technology_unlock():
    """stone, iron, gold, wood should only increase year over year (unless spent on tech)."""
    pass


def test_get_resources_returns_nonzero_on_plains():
    """A tribe on a plains tile should collect positive food and water each year."""
    pass


# ---------------------------------------------------------------------------
# Technology invariants
# ---------------------------------------------------------------------------


def test_technology_requires_resources_to_unlock():
    """A tribe with 0 resources should never advance beyond era 0."""
    pass


def test_technology_consumes_resources_on_unlock():
    """After unlocking an era, the required resources should be deducted from the tribe's stock."""
    pass


def test_technology_only_advances_one_era_at_a_time():
    """Even with excess resources, the tribe should advance one era per step, not multiple."""
    pass


def test_technology_reduces_death_rate():
    """death_rate with era=2 should be strictly lower than with era=0, all else equal."""
    pass


# ---------------------------------------------------------------------------
# Migration invariants
# ---------------------------------------------------------------------------


def test_migration_stops_after_sedentarisation():
    """Once population >= HAB_THRESHOLD, migrate() should never reset territory to 1 tile."""
    pass


def test_migration_never_moves_to_ocean():
    """migrate() should only move the tribe to land tiles."""
    pass


def test_migration_moves_toward_better_habitability():
    """
    Given a clear habitability gradient, the tribe should statistically
    move toward higher habitability over 20 migration steps.
    """
    pass


# ---------------------------------------------------------------------------
# Exploration invariants
# ---------------------------------------------------------------------------


def test_exploration_path_never_crosses_territory():
    """The exploration walk should never step on a tile already in self.territory."""
    pass


def test_exploration_path_length_bounded_by_length_exploration():
    """The number of tiles visited during exploration should not exceed length_exploration."""
    pass


def test_exploration_returns_zero_harvest_if_no_border():
    """A tribe completely surrounded (no border tiles) should return an empty harvest."""
    pass


# ---------------------------------------------------------------------------
# Extinction invariants
# ---------------------------------------------------------------------------


def test_extinct_tribe_does_not_step():
    """A tribe with alive=False should not modify any state when step() is called."""
    pass


def test_extinction_triggered_below_threshold():
    """_check_extinction() should set alive=False when population < 5."""
    pass
