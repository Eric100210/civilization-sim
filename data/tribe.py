import random
import numpy as np
import math
from .map import World


class Tribe:
    def __init__(self, world: World):
        self.world = world
        self.population = None
        self.resources = None
        self.territory = None
        self.technology = None
        self.aggressiveness = None

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

    def reproduction(self):
        pass

    def death(self):
        pass

    def hunt(self):
        pass
