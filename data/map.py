import numpy as np
from perlin_noise import PerlinNoise
import random
import math
from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
from heapq import heappush, heappop
from .resources import ResourceType


colors = {
    "ocean": (66, 110, 225),
    "plains": (26, 176, 25),
    "desert": (227, 193, 5),
    "mountains": (140, 140, 140),
    "snow": (250, 250, 250),
    "river": (66, 120, 241),
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
        self.is_river = 0
        self.biome = None

    def compute_biome(
        self, elevation: float, temperature: float, humidity: float
    ) -> str:
        if elevation < ELEVATION_SEA:
            return "ocean"
        if self.is_river == 1:
            return "river"
        if elevation > ELEVATION_MOUNTAIN:
            return "snow" if temperature < 0 else "mountains"
        if temperature > TEMP_DESERT and humidity < HUMIDITY_DESERT:
            return "desert"
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
        self.resource_maps: dict[str, np.ndarray] = {
            t.value: np.zeros((height, width)) for t in ResourceType
        }
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

        # River influence: binary mask blurred into a valley effect.
        # sigma controls how far the fertility bonus spreads from the river bank.
        # sigma=2 ≈ 2 tiles radius, peaks at 3.0 on the river itself.
        river_grid = np.zeros((self.height, self.width))
        for x in range(self.width):
            for y in range(self.height):
                river_grid[y, x] = self.tiles[x][y].is_river
        river_influence = gaussian_filter(river_grid, sigma=2) * 3.0

        # Temperature: bell curve centred on 0.5 (≈15°C, optimal for agriculture)
        temp_score = np.exp(-((self.temperature_map - 0.5) ** 2) / 0.08)

        # Humidity: bell curve centred on 0.4 (moderate — not too dry, not tropical)
        humidity_score = np.exp(-((self.humidity_map - 0.4) ** 2) / 0.125)

        # Water proximity: gentle decay from ocean/lake shore
        water_score = np.exp(-self.distance_map * 0.05)

        # Alluvial fertility bonus: flat low land adjacent to a river
        # (river deltas and floodplains — historically where civilisations begin)
        alluvial = (self.elevation_map < 0.15) & (river_grid == 1)
        fertility_bonus = alluvial.astype(float) * 2.0

        # Altitude penalty: non-linear above 0.15
        altitude_penalty = np.maximum(0, self.elevation_map - 0.15) ** 1.5 * 3.0

        habit = np.where(
            land,
            water_score
            + temp_score
            + humidity_score
            + river_influence
            + fertility_bonus
            - altitude_penalty,
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

    def _fill_sinks(self, elev: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """
        Fill depressions (sinks) in the elevation map so that every land tile
        drains to the ocean.  Uses an iterative flood-fill starting from the
        ocean border.

        Algorithm (simplified Priority-Flood):
          - Initialise a 'filled' surface equal to elev everywhere.
          - Ocean tiles are fixed at their real elevation (they are the outlets).
          - Propagate from low to high: if a land neighbour is lower than the
            current filled surface, raise it just above so water can escape.
        """

        filled = np.full_like(elev, np.inf)
        visited = np.zeros((self.height, self.width), dtype=bool)
        heap = []  # (filled_elevation, y, x)

        # Seed the heap with every ocean-border tile
        for y in range(self.height):
            for x in range(self.width):
                if elev[y, x] < ELEVATION_SEA:
                    filled[y, x] = elev[y, x]
                    visited[y, x] = True
                    heappush(heap, (elev[y, x], y, x))

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while heap:
            elev_cur, cy, cx = heappop(heap)
            for dy, dx in dirs:
                ny, nx = cy + dy, cx + dx
                if (
                    0 <= ny < self.height
                    and 0 <= nx < self.width
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    # Neighbour must be at least as high as current filled surface
                    filled[ny, nx] = max(elev[ny, nx], elev_cur + epsilon)
                    heappush(heap, (filled[ny, nx], ny, nx))

        return filled

    def generate_rivers(self, n_sources: int = 20, randomness: float = 0.3) -> None:
        """
        Organic river generation: stochastic descent on a sink-filled elevation map.

        Each river starts from a random mountain tile and walks downhill to the
        ocean. At each step, all strictly-downhill neighbours are candidates;
        they are chosen with probability proportional to (drop ^ (1 - randomness)).

          randomness=0.0 → always pick the steepest neighbour (D8-like, straight lines)
          randomness=1.0 → uniform random among all downhill neighbours (very wiggly)
          randomness=0.3 → mostly follows the slope, with organic meanders (default)

        Sink-filling guarantees every path reaches the ocean — no river gets stuck.
        """
        filled = self._fill_sinks(self.elevation_map)

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        self.river_flow = np.zeros((self.height, self.width))

        mountain_tiles = [
            (y, x)
            for y in range(self.height)
            for x in range(self.width)
            if self.elevation_map[y, x] > ELEVATION_MOUNTAIN
        ]
        sources = random.sample(mountain_tiles, min(n_sources, len(mountain_tiles)))

        for sy, sx in sources:
            y, x = sy, sx
            path = []

            while True:
                path.append((y, x))

                # Stop when we reach the ocean
                if filled[y, x] < ELEVATION_SEA:
                    break

                # Candidates: strictly downhill neighbours on the filled surface
                candidates = []
                for dy, dx in dirs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        drop = filled[y, x] - filled[ny, nx]
                        if drop > 0:
                            candidates.append((drop, ny, nx))

                if not candidates:
                    break  # flat peak with no outlet (shouldn't happen after fill)

                # Weighted random choice: weight = drop^(1 - randomness)
                # randomness=0 → weight=drop (steepest wins strongly)
                # randomness=1 → weight=1 (uniform)
                weights = [d ** (1.0 - randomness) for d, _, _ in candidates]
                total = sum(weights)
                r = random.random() * total
                cumul = 0.0
                chosen = candidates[-1]
                for w, (drop, ny, nx) in zip(weights, candidates):
                    cumul += w
                    if r <= cumul:
                        chosen = (drop, ny, nx)
                        break
                _, y, x = chosen

            for py, px in path:
                self.river_flow[py, px] += 1
                self.tiles[px][py].is_river = 1

    def compute_biomes(self) -> None:
        for x in range(self.width):
            for y in range(self.height):
                tile = self.tiles[x][y]
                tile.biome = tile.compute_biome(
                    self.elevation_map[y, x],
                    self.temperature_map[y, x],
                    self.humidity_map[y, x],
                )

    def generate_resources(self) -> None:
        for x in range(self.width):
            for y in range(self.height):
                tile = self.tiles[x][y]
                if tile.biome == "ocean" or tile.biome == "river":
                    self.resource_maps[ResourceType.WATER.value][y, x] = 1000
                    self.resource_maps[ResourceType.FOOD.value][y, x] = 5
                elif tile.biome == "mountains":
                    self.resource_maps[ResourceType.FOOD.value][y, x] = 1
                    self.resource_maps[ResourceType.STONE.value][y, x] = 10
                    self.resource_maps[ResourceType.IRON.value][y, x] = 5
                    self.resource_maps[ResourceType.GOLD.value][y, x] = 1
                elif tile.biome == "snow":
                    self.resource_maps[ResourceType.STONE.value][y, x] = 10
                    self.resource_maps[ResourceType.WATER.value][y, x] = 5
                elif tile.biome == "desert":
                    self.resource_maps[ResourceType.STONE.value][y, x] = 3
                    self.resource_maps[ResourceType.IRON.value][y, x] = 3
                    self.resource_maps[ResourceType.GOLD.value][y, x] = 0.1
                else:
                    self.resource_maps[ResourceType.FOOD.value][y, x] = 10
                    self.resource_maps[ResourceType.STONE.value][y, x] = 1

    def generate(self, n_rivers: int = 20, river_randomness: float = 0.3) -> None:
        self.elevation()  # inclut les montagnes (ridge)
        self.compute_distance_to_water()
        self.humidity()
        self.temperature()
        self.generate_rivers(n_sources=n_rivers, randomness=river_randomness)
        self.compute_habitability()
        self.compute_biomes()
        self.generate_resources()

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
