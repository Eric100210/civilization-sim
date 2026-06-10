"""
Calibration tests for Tribe behaviour.

These tests verify behavioural invariants rather than exact values,
since the simulation is stochastic. Each test documents an expected
demographic or territorial property that should hold across any seed.

Run with:  pytest tests/
"""

import pytest
import random
from unittest.mock import patch
import numpy as np
from data.map import World
from data.tribe import Tribe, HAB_THRESHOLD
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
    w.generate(n_rivers=5, river_randomness=0.3)
    return w


@pytest.fixture
def plain_tribe(world):
    tribe = Tribe(world)
    x = 23
    y = 55
    tribe.territory = {(x, y)}
    habit = world.habitability_map[y, x]
    tribe.population = 100
    return tribe


@pytest.fixture
def mountain_tribe(world):
    tribe = Tribe(world)
    x = 95
    y = 28
    tribe.territory = {(x, y)}
    tribe.population = 100
    return tribe


@pytest.fixture
def snow_tribe(world):
    tribe = Tribe(world)
    x = 92
    y = 42
    tribe.territory = {(x, y)}
    tribe.population = 100
    return tribe


@pytest.fixture
def full_resources():
    resources = {r.value: 1000 for r in ResourceType}
    return resources


@pytest.fixture
def no_resources():
    resources = {r.value: 0 for r in ResourceType}
    return resources


# ---------------------------------------------------------------------------
# Territorial invariants
# ---------------------------------------------------------------------------


def test_expand_never_adds_ocean_tile(world, plain_tribe, full_resources):
    """expand() must only add land tiles — never ocean or out-of-bounds."""
    plain_tribe.population = HAB_THRESHOLD * 10
    plain_tribe.resources = full_resources

    for year in range(100):
        plain_tribe.expand()

    for x, y in plain_tribe.territory:
        assert 0 <= x < world.width, f"Tile ({x},{y}) is out of bounds"
        assert 0 <= y < world.height, f"Tile ({x},{y}) is out of bounds"
        assert world.is_land[y, x], f"Tile ({x},{y}) is ocean — should never be added"


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


def test_consumables_reset_each_year(plain_tribe):
    """water and food should be reset to 0 at the start of each get_resources() call."""
    # Artificially inflate consumables as if they had accumulated
    plain_tribe.resources[ResourceType.WATER.value] = 9999
    plain_tribe.resources[ResourceType.FOOD.value] = 9999

    plain_tribe.get_resources()

    # After get_resources(), water and food should reflect only this year's harvest,
    # not the 9999 leftover — i.e. they were reset to 0 before collection
    assert plain_tribe.resources[ResourceType.WATER.value] < 9999
    assert plain_tribe.resources[ResourceType.FOOD.value] < 9999


def test_accumulables_never_decrease_without_technology_unlock():
    """stone, iron, gold, wood should only increase year over year (unless spent on tech)."""
    pass


def test_get_resources_returns_nonzero_on_plains(plain_tribe):
    """A tribe on a plains tile should collect positive food and water each year."""
    plain_tribe.get_resources()

    assert plain_tribe.resources[ResourceType.FOOD.value] > 0
    assert plain_tribe.resources[ResourceType.WATER.value] > 0


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


def test_extinction_triggered_below_threshold(plain_tribe):
    """_check_extinction() should set alive=False when population < 5."""
    plain_tribe.population = 4.9
    plain_tribe._check_extinction()

    assert plain_tribe.alive is False
    assert plain_tribe.population == 0.0
    assert plain_tribe.territory == set()


def test_war_extinction_absorbs_population(world):
    """When a tribe dies during a war, the victor must absorb 20-40% of the
    remaining population (regression: absorbed was computed after zeroing pop)."""
    winner = Tribe(world)
    loser = Tribe(world)
    winner.territory = {(10, 10)}
    winner.population = 1000.0
    loser.territory = {(12, 10)}
    loser.population = 4.0
    for tribe, opp in ((winner, loser), (loser, winner)):
        tribe.at_war = True
        tribe.war_enemy = opp
        tribe.war_pop_start = tribe.population
        tribe.war_ter_start = len(tribe.territory)

    loser._check_extinction()

    assert loser.alive is False
    assert winner.population > 1000.0  # absorbed 0.8–1.6 people, never 0
    assert winner.at_war is False
    assert winner.war_enemy is None


def test_relocate_updates_tile_ownership(world):
    """Nomadic migration must move tile ownership along with the territory
    (regression: the old tile stayed owned forever, the new one never was)."""
    tribe = Tribe(world)
    x, y = tribe.spawn()

    nx, ny = x + 1, y
    tribe._relocate(nx, ny)

    assert tribe not in world.tiles[x][y].owner
    assert tribe in world.tiles[nx][ny].owner
    assert tribe.territory == {(nx, ny)}


def test_expand_triggers_before_full_saturation(world, plain_tribe):
    """A tribe nearing (but below) carrying capacity should still bud off new
    tiles — waiting for pressure > 1.0 deadlocks growth against density deaths."""
    tribe = plain_tribe
    x, y = next(iter(tribe.territory))
    world.tiles[x][y].owner.add(tribe)
    K = tribe._carrying_capacity()
    tribe.population = max(HAB_THRESHOLD, K * 0.9)

    for _ in range(20):
        tribe.expand()

    assert len(tribe.territory) > 1


def test_expand_targets_missing_era_resources(world):
    """A tribe lacking iron for its next era should eventually colonize an
    adjacent iron-bearing tile even though it is less habitable."""
    iron_map = world.resource_maps[ResourceType.IRON.value]
    tribe = None
    for y in range(world.height):
        for x in range(world.width):
            if iron_map[y, x] > 0 and world.is_land[y, x]:
                for nx, ny in world.tiles[x][y].neighbors(world):
                    if world.is_land[ny, nx] and iron_map[ny, nx] == 0:
                        tribe = Tribe(world)
                        tribe.territory = {(nx, ny)}
                        world.tiles[nx][ny].owner.add(tribe)
                        tribe.population = float(HAB_THRESHOLD)
                        tribe.hist_eras = 1  # next era (Bronze) requires iron
                        break
            if tribe:
                break
        if tribe:
            break
    assert tribe is not None, "fixture world has no iron tile with a land neighbour"

    for _ in range(100):
        tribe._cached_border = None
        tribe.expand()

    assert any(iron_map[ty, tx] > 0 for tx, ty in tribe.territory)


def test_one_sided_war_ends_in_capitulation(world):
    """A hopeless war (one side bleeding, the other intact) must end in
    capitulation well before the 40-year timeout, not grind to annihilation."""
    strong = Tribe(world)
    weak = Tribe(world)
    strong.territory = {(20 + dx, 20 + dy) for dx in range(10) for dy in range(10)}
    weak.territory = {(30 + dx, 20 + dy) for dx in range(5) for dy in range(5)}
    for t in (strong, weak):
        for tx, ty in t.territory:
            world.tiles[tx][ty].owner.add(t)
    strong.population = 100_000.0
    weak.population = 6_000.0
    for tribe, opp in ((strong, weak), (weak, strong)):
        tribe.at_war = True
        tribe.war_enemy = opp
        tribe.war_pop_start = tribe.population
        tribe.war_ter_start = len(tribe.territory)

    weak_pop_start = weak.population
    for _ in range(20):
        strong.war([strong, weak])
        weak.war([strong, weak])
        if not strong.at_war:
            break

    assert strong.at_war is False, "war should have ended by capitulation"
    assert weak.at_war is False
    assert strong.truce_timer > 0 and weak.truce_timer > 0
    assert weak.population < weak_pop_start


def test_extinct_tribe_does_not_step(plain_tribe):
    """A tribe with alive=False should return immediately without calling any method."""
    plain_tribe.alive = False

    with (
        patch.object(plain_tribe, "migrate") as mock_migrate,
        patch.object(plain_tribe, "get_resources") as mock_resources,
        patch.object(plain_tribe, "population_growth") as mock_growth,
        patch.object(plain_tribe, "expand") as mock_expand,
    ):
        plain_tribe.step(year=0, all_tribes=[])

        mock_migrate.assert_not_called()
        mock_resources.assert_not_called()
        mock_growth.assert_not_called()
        mock_expand.assert_not_called()
