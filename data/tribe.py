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

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main yearly step — orchestrates everything a tribe does in one year
    # ------------------------------------------------------------------

    def step(self, year: int, all_tribes: list["Tribe"]) -> None:
        """Called once per year. Order matters: resources first, then decisions."""
        if not self.alive:
            return
        self.migrate()
        self.reproduce()
        self.death()
        # self.trade(all_tribes)   # à brancher plus tard
        # self.war(all_tribes)     # à brancher plus tard

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def migrate(self) -> None:
        x, y = next(iter(self.territory))
        habit_map = self.world.habitability_map
        tile = self.world.tiles[x][y]
        current_habit = habit_map[y, x]
        best = (current_habit, x, y)

        for nx, ny in tile.neighbors(self.world):
            h = habit_map[ny, nx]
            if h > best[0]:
                best = (h, nx, ny)

        best_h, bx, by = best
        if best_h > current_habit:
            # do not migrate if minor habitability change : attachment to the land
            p = min(1, (best_h - current_habit) * 5)
            if random.random() < p:
                self.territory = {(bx, by)}
                self.display_territory()

    def reproduce(self) -> None:
        pass

    def death(self) -> None:
        pass

    def trade(self, all_tribes: list["Tribe"]) -> None:
        pass

    def war(self, all_tribes: list["Tribe"]) -> None:
        pass
