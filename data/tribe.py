import random
import numpy as np
from .map import World


class Tribe:
    def __init__(self, world: World):
        self.world = world
        self.population: float | None = None
        self.resources: float | None = None
        self.territory: set[tuple[int, int]] | None = None
        self.technology: float | None = None
        self.aggressiveness: float | None = None
        self.alive: bool = True

    def display_territory(self) -> None:
        for x, y in self.territory:
            self.world.tiles[x][y].biome = "village"

    def spawn(self) -> tuple[int, int]:
        # argwhere on (height, width) returns (row, col) = (y, x)
        land_positions = np.argwhere(self.world.is_land)
        y, x = random.choice(land_positions)
        self.territory = {(x, y)}
        # initial population : lognormal distribution
        habit = self.world.habitability_map[y, x]
        self.population = np.random.lognormal(mean=3.5 + habit, sigma=0.3)
        return x, y

    def step(self, year: int, all_tribes: list["Tribe"]) -> None:
        """Called once per year. Order matters: resources first, then decisions."""
        if not self.alive:
            return
        self.migrate()
        self.population_growth()
        self.death()
        # self.trade(all_tribes)   # à brancher plus tard
        # self.war(all_tribes)     # à brancher plus tard

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def migrate(self) -> None:
        """
        Two-phase migration model inspired by Ammerman & Cavalli-Sforza (1984):

        1. EXPLORE — scout a radius of `vision` tiles and find the best spot.
           Vision grows with population (larger groups scout further).
        2. DECIDE — migrate only if the best spot is significantly better
           than current location AND the tribe isn't already well-settled.
           The threshold prevents endless restlessness in already-good zones.
        """
        x, y = next(iter(self.territory))
        habit_map = self.world.habitability_map
        current_habit = habit_map[y, x]

        # Vision radius: grows logarithmically with population (3–8 tiles)
        pop = max(1, self.population or 1)
        vision = int(np.clip(2 + np.log(pop / 50 + 1) * 2, 3, 8))

        # Scan all tiles within vision radius
        best_h, bx, by = current_habit, x, y
        for dy in range(-vision, vision + 1):
            for dx in range(-vision, vision + 1):
                if dx * dx + dy * dy > vision * vision:
                    continue  # circular radius
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                    h = habit_map[ny, nx]
                    if h > best_h:
                        best_h, bx, by = h, nx, ny

        # Migrate with a probability that depends on:
        #   - relative gain  : how much better is the destination?
        #   - attachment     : harder to leave already-good land
        #
        # p = sigmoid-like curve:
        #   gain < 0        → never migrate (destination is worse)
        #   gain = 0.2      → ~30% chance on average land
        #   gain = 0.5      → ~60% chance
        #   gain = 1.0+     → ~90% chance
        # Divided by (1 + current_habit * 0.3) so well-settled tribes
        # need a proportionally larger gain to bother moving.
        gain = best_h - current_habit
        if gain <= 0:
            return
        attachment = 1.0 + current_habit * 0.3
        p = min(1.0, (gain / attachment) * 1.5)
        if random.random() < p:
            self.territory = {(bx, by)}
            self.display_territory()

    def population_growth(self) -> None:
        """See logistic growth equation (Verhulst 1838)"""
        pass

    def death(self) -> None:
        pass

    def trade(self, all_tribes: list["Tribe"]) -> None:
        pass

    def war(self, all_tribes: list["Tribe"]) -> None:
        pass
