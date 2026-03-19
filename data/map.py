import numpy as np
from perlin_noise import PerlinNoise
import random
import math
from scipy.ndimage import distance_transform_edt, maximum_filter


colors = {
    "ocean": (66, 110, 225),
    "plains": (26, 176, 25),
    "desert": (227, 193, 5),
    "mountains": (140, 140, 140),
    "snow": (250, 250, 250),
    "river": (66, 120, 241),
    "village": (102, 57, 9),
}

# Biome thresholds
ELEVATION_SEA = 0.0
ELEVATION_MOUNTAIN = 0.4
TEMP_DESERT = 0.7
HUMIDITY_DESERT = 0.15


class Tile:
    """
    Lightweight object for biome/river state. Scalar terrain values
    (elevation, humidity, temperature, habitability) are stored in numpy
    arrays on World and NOT duplicated here.
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.river = 0
        self.biome = None

    def compute_biome(
        self, elevation: float, temperature: float, humidity: float
    ) -> str:
        if elevation < ELEVATION_SEA:
            return "ocean"
        if elevation > ELEVATION_MOUNTAIN:
            return "snow" if temperature < 0 else "mountains"
        if temperature > TEMP_DESERT and humidity < HUMIDITY_DESERT:
            return "desert"
        if self.river == 1:
            return "river"
        return "plains"

    def neighbors(self, world: "World") -> list[tuple[int, int]]:
        """Returns coordinates of valid 8-neighbours."""
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < world.width and 0 <= ny < world.height:
                    result.append((nx, ny))
        return result


class World:
    """
    All value maps are numpy arrays of shape (height, width), indexed [y, x].
    self.tiles[x][y] holds Tile objects (indexed [x][y] for convenience).
    """

    def __init__(self, width: int, height: int, seed: int | None = None):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 100000)
        self.humidity_seed = random.randint(0, 100000)

        # Tile objects grid — kept as list-of-lists indexed [x][y]
        self.tiles = [[Tile(x, y) for y in range(height)] for x in range(width)]

        # All value maps: shape (height, width), indexed [y, x]
        self.elevation_map: np.ndarray = np.zeros((height, width))
        self.humidity_map: np.ndarray = np.zeros((height, width))
        self.temperature_map: np.ndarray = np.zeros((height, width))
        self.habitability_map: np.ndarray = np.zeros((height, width))
        self.river_flow: np.ndarray = np.zeros((height, width))
        self.distance_map: np.ndarray | None = None
        self.is_land: np.ndarray | None = None
        self.is_water: np.ndarray | None = None

    def display_habitability(self) -> None:
        import matplotlib.pyplot as plt

        # habitability_map is already (height, width) — no transpose needed
        plt.imshow(self.habitability_map, origin="lower", cmap="magma")
        plt.colorbar(label="Habitability")
        plt.show()

    def elevation(self) -> None:
        """Single pass over the grid to compute all elevation noise layers at once."""
        continent_noise = PerlinNoise(octaves=2, seed=self.seed + 1000)
        noise1 = PerlinNoise(octaves=3, seed=self.seed)
        noise2 = PerlinNoise(octaves=3, seed=self.seed + 1)
        noise3 = PerlinNoise(octaves=3, seed=self.seed + 2)
        ridge_noise = PerlinNoise(octaves=1, seed=self.seed + 100)

        elev = np.empty((self.height, self.width))
        ridge_raw = np.empty((self.height, self.width))

        for y in range(self.height):
            ny = y / self.height
            for x in range(self.width):
                nx = x / self.width
                continent = continent_noise([nx * 0.8, ny * 0.8])
                e = noise1([nx * 2, ny * 2])
                e += 0.5 * noise2([nx * 4, ny * 4])
                e += 0.25 * noise3([nx * 8, ny * 8])
                elev[y, x] = e * 0.75 + continent + 0.05
                ridge_raw[y, x] = ridge_noise([nx * 2, ny * 2])

        # Mountain ridges — vectorised post-pass
        ridge = 1 - np.abs(ridge_raw)
        land_mask = elev > 0
        strong_ridge = ridge > 0.8
        elev += np.where(land_mask & strong_ridge, (ridge - 0.8) * 0.8, 0.0)

        self.elevation_map = elev

    def compute_distance_to_water(self) -> None:
        # water_mask shape (height, width) — True where water
        water_mask = self.elevation_map < ELEVATION_SEA
        self.distance_map = distance_transform_edt(~water_mask)
        self.is_water = self.distance_map == 0
        self.is_land = self.distance_map > 0

    def humidity(self) -> None:
        """Single pass to compute humidity noise, then vectorised water effect."""
        noise = PerlinNoise(octaves=3, seed=self.humidity_seed)

        noise_grid = np.empty((self.height, self.width))
        for y in range(self.height):
            ny = y / self.height
            for x in range(self.width):
                noise_grid[y, x] = noise([x / self.width, ny])

        noise_grid *= 0.15
        water_effect = np.exp(-self.distance_map * 0.2)
        self.humidity_map = np.clip(noise_grid + water_effect, 0.0, 1.0)

    def temperature(self) -> None:
        self.temperature_map = 1 - np.abs(self.elevation_map * 1.8)

    def compute_habitability(self) -> None:
        land = self.is_land  # bool (height, width)

        # River contribution: build a (height, width) bool array from tiles
        river_grid = np.zeros((self.height, self.width))
        for x in range(self.width):
            for y in range(self.height):
                river_grid[y, x] = self.tiles[x][y].river

        habit = np.where(
            land,
            (
                2 * np.exp(-self.distance_map * 0.1)
                + river_grid
                + 2 * (0.6 - self.humidity_map)
                + 2 * (0.6 - self.temperature_map)
                - np.maximum(self.elevation_map, 0) * 0.5
            ),
            -1.0,
        )
        self.habitability_map = habit

    def habitability_local_max(self) -> list[tuple]:
        hmap = self.habitability_map  # (height, width)
        local_max = maximum_filter(hmap, size=15)
        maxima_mask = (hmap == local_max) & self.is_land

        # argwhere returns (row, col) = (y, x)
        coords = np.argwhere(maxima_mask)
        scores = hmap[coords[:, 0], coords[:, 1]]

        locations = sorted(
            zip(scores, coords[:, 1], coords[:, 0]), reverse=True
        )  # (score, x, y)
        return locations

    def generate_rivers(self, n_sources: int = 20, max_length: int = 800) -> None:
        self.river_flow = np.zeros((self.height, self.width))

        mountain_tiles = [
            (tile.x, tile.y)
            for col in self.tiles
            for tile in col
            if self.elevation_map[tile.y, tile.x] > ELEVATION_MOUNTAIN
        ]
        sources = random.sample(mountain_tiles, min(n_sources, len(mountain_tiles)))

        for sx, sy in sources:
            x, y = sx, sy
            path = []
            for _ in range(max_length):
                path.append((x, y))
                if self.elevation_map[y, x] < ELEVATION_SEA:
                    break

                neigh_coords = self.tiles[x][y].neighbors(self)
                candidates = [
                    (self.elevation_map[ny, nx], nx, ny)
                    for nx, ny in neigh_coords
                    if self.elevation_map[ny, nx] <= self.elevation_map[y, x] + 0.02
                ]
                if not candidates:
                    break

                candidates.sort()
                _, x, y = random.choice(candidates[:3])

            for px, py in path:
                self.river_flow[py, px] += 1
                self.tiles[px][py].river = 1

    def compute_biomes(self) -> None:
        for x in range(self.width):
            for y in range(self.height):
                tile = self.tiles[x][y]
                tile.biome = tile.compute_biome(
                    self.elevation_map[y, x],
                    self.temperature_map[y, x],
                    self.humidity_map[y, x],
                )

    def generate(self) -> None:
        self.elevation()           # inclut les montagnes (ridge)
        self.compute_distance_to_water()
        self.humidity()
        self.temperature()
        self.generate_rivers()
        self.compute_habitability()
        self.compute_biomes()

    def step(self, year: int, tribes: list) -> None:
        """
        Everything that changes on the world map in one year.
        Called by Simulation.animate() before rendering.

        Future hooks:
          - climate drift
          - resource depletion / regeneration
          - recalculate habitability if resources change
        """
        for tribe in tribes:
            tribe.step(year, tribes)

    def biome_map(self) -> np.ndarray:
        """Returns RGB image of shape (height, width, 3) ready for imshow."""
        color_array = np.zeros((self.height, self.width, 3))
        for x in range(self.width):
            for y in range(self.height):
                biome = self.tiles[x][y].biome
                r, g, b = colors[biome]
                color_array[y, x] = [r / 255, g / 255, b / 255]
        return color_array
