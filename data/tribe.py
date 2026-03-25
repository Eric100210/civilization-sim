import random
import numpy as np
from .map import World

HAB_THRESHOLD = 500


class Tribe:
    def __init__(self, world: World):
        self.world = world
        self.population: float | None = None
        self.resources: float | None = None
        self.territory: set[tuple[int, int]] | None = None
        self.technology: float | None = None
        self.aggressiveness: float | None = None
        self.alive: bool = True

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
        self.expand()
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

        Only active during the nomadic phase (population < SEDENTARY_THRESHOLD).
        Once sedentary, expand() takes over territorial growth.
        """
        if self.population >= HAB_THRESHOLD:
            return

        x, y = next(iter(self.territory))
        habit_map = self.world.habitability_map
        current_habit = habit_map[y, x]

        # Vision radius: grows logarithmically with population (3–8 tiles) : or with technology ?
        # With population / technology growing, they will explore further and migrate more, until war
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

    def _carrying_capacity(self) -> float:
        """Sum of habitability over owned tiles, scaled by HAB_THRESHOLD."""
        return max(
            1.0,
            sum(self.world.habitability_map[y, x] for x, y in self.territory)
            * HAB_THRESHOLD,
        )

    def _border_tiles(self) -> list[tuple[int, int]]:
        """All land tiles adjacent to the territory but not yet owned."""
        candidates = set()
        for x, y in self.territory:
            for nx, ny in self.world.tiles[x][y].neighbors(self.world):
                if self.world.is_land[ny, nx] and (nx, ny) not in self.territory:
                    candidates.add((nx, ny))
        return list(candidates)

    def population_growth(self) -> None:
        n = len(self.territory)
        avg_hab = np.mean(
            [self.world.habitability_map[y, x] for x, y in self.territory]
        )
        density = self.population / max(1, n)

        # Natality: favorable environment + resources (so commerce) + peace (no war)
        # implement resources and technology logic to have a real population model
        birth_rate = 0.002 + avg_hab * 0.01

        # Mortality : density + technology reducing it + extreme climate
        tech = self.technology or 0.0
        death_rate = 0.001 + density * 0.00005  # - tech * 0.001

        r = birth_rate - death_rate
        self.population = max(1.0, self.population * (1 + r))

    def expand(self) -> None:
        """
        Two mechanisms of territorial expansion:

        1. PRESSURE — overpopulation (P > K): the tribe must expand to survive.
           Number of new tiles ~ Poisson(pressure - 1).

        2. OPPORTUNISTIC — even without pressure, the tribe slowly colonizes
           adjacent tiles that are significantly more fertile than its current average.
        """
        if not self.territory:
            return

        if self.population < HAB_THRESHOLD:
            return

        K = self._carrying_capacity()
        pressure = self.population / K
        border = self._border_tiles()

        if not border:
            return

        # 1. Pressure-driven expansion
        if pressure > 1.0:
            n_tiles = np.random.poisson(lam=pressure - 1.0)
            for _ in range(n_tiles):
                if not border:
                    break
                chosen = random.choice(border)
                self.territory.add(chosen)
                border.remove(chosen)

        # 2. Opportunistic expansion toward more fertile land
        avg_hab = K / (len(self.territory) * HAB_THRESHOLD)
        for nx, ny in border:
            tile_hab = self.world.habitability_map[ny, nx]
            if tile_hab > avg_hab and random.random() < 0.01:
                self.territory.add((nx, ny))

    def get_resources(self):
        """Ressources sur les terres + exploration des zones alentours, surtout les montagnes pour les minerais"""
        pass

    def get_technology(self):
        """Dépend des ressources trouvées et du savoir-faire"""
        pass

    def trade(self, all_tribes: list["Tribe"]) -> None:
        pass

    def war(self, all_tribes: list["Tribe"]) -> None:
        pass
