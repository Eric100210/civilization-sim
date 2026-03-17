import numpy as np
from perlin_noise import PerlinNoise
import random
import math
from scipy.ndimage import distance_transform_edt, maximum_filter

SCALE = 100.0


colors = {
    "ocean": (66, 110, 225),
    "plains": (26, 176, 25),
    "desert": (227, 193, 5),
    "mountains": (140, 140, 140),
    "snow": (250, 250, 250),
    "river": (66, 120, 241),
}


class Tile:

    def __init__(self, x: int, y: int, elevation: int, temperature: int | None = None, humidity: int | None = None):
        self.x = x
        self.y = y
        self.elevation = elevation
        self.temperature = temperature
        self.humidity = humidity
        self.river = 0
        self.biome = None
        self.habitability = None

    def compute_biome(self) -> str:

        if self.elevation < 0:
            return "ocean"

        if self.elevation > 0.4:
            if self.temperature < 0:
                return "snow"
            return "mountains"

        if self.temperature > 0.7 and self.humidity < 0.15:
            return "desert"
        
        if self.river == 1:
            return "river"

        return "plains"
    
    def neighbors(self, world: 'World') -> list[tuple[int,int]]:
        """
        Renvoie la liste des coordonnées des 8 voisins valides.
        """
        dirs = [(-1,-1), (-1,0), (-1,1),
                (0,-1),          (0,1),
                (1,-1),  (1,0),  (1,1)]
        result = []
        for dx, dy in dirs:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < world.width and 0 <= ny < world.height:
                result.append((nx, ny))
        return result
        

class World:

    def __init__(self, width: int, height: int, seed: int | None = None):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 100000)
        self.continent_noise = PerlinNoise(octaves=2, seed = self.seed + 1000)
        self.humidity_seed = random.randint(0,100000)
        self.tiles = [[None for _ in range(height)] for _ in range(width)]
        self.distance_map = None
        self.is_land = None
        self.is_water = None
        self.elevation_map = [[None for _ in range(height)] for _ in range(width)]
        self.humidity_map = [[None for _ in range(height)] for _ in range(width)]
        self.temperature_map = [[None for _ in range(height)] for _ in range(width)]
        self.habitability_map = [[None for _ in range(height)] for _ in range(width)]
        
    def humidity_noises(self, seed: int) -> tuple[np.ndarray]:
        noise1 = PerlinNoise(octaves=3, seed=seed)
        noise2 = PerlinNoise(octaves=6, seed=seed)
        noise3 = PerlinNoise(octaves=12, seed=seed)
        noise4 = PerlinNoise(octaves=24, seed=seed)
        return (noise1, noise2, noise3, noise4)

    def compute_distance_to_water(self) -> None:
        water_mask = np.zeros((self.width, self.height), dtype=bool)
        for x in range(self.width):
            for y in range(self.height):
                water_mask[x, y] = self.tiles[x][y].elevation < 0

        self.distance_map = distance_transform_edt(~water_mask)
        self.is_water = self.distance_map == 0
        self.is_land = self.distance_map > 0
    
    def elevation(self) -> None:
        noise1 = PerlinNoise(octaves=3, seed=self.seed)
        noise2 = PerlinNoise(octaves=3, seed=self.seed+1)
        noise3 = PerlinNoise(octaves=3, seed=self.seed+2)

        for y in range(self.height):
            for x in range(self.width):

                nx = x / self.width
                ny = y / self.height

                # continents (low frequency)
                continent = self.continent_noise([nx*0.8, ny*0.8])

                # principal relief
                elevation = noise1([nx*2, ny*2])

                # details
                elevation += 0.5 * noise2([nx*4, ny*4])
                elevation += 0.25 * noise3([nx*8, ny*8])

                elevation = elevation * 0.75 + continent + 0.05

                tile = Tile(x=x, y=y, elevation=elevation)
                self.tiles[x][y] = tile

    def add_mountains(self) -> None:
        ridge_noise = PerlinNoise(octaves=1, seed=self.seed + 100)

        for y in range(self.height):
            for x in range(self.width):
                nx = x / self.width
                ny = y / self.height
                tile = self.tiles[x][y]

                if tile.elevation > 0:
                    val = ridge_noise([nx*2, ny*2])
                    ridge = 1 - abs(val)

                    if ridge > 0.8:
                        tile.elevation += (ridge - 0.8) * 0.8

    def humidity(self) -> None:
        humidity_seeds = self.humidity_noises(self.humidity_seed)
        if not hasattr(self, "distance_map"):
            self.compute_distance_to_water()
        for y in range(self.height):
            for x in range(self.width):

                tile = self.tiles[x][y]

                distance = self.distance_map[x, y]

                water_effect = math.exp(-distance*0.2)

                humidity_noise = humidity_seeds[0]([x/self.width, y/self.height])
                humidity_noise *= 0.15

                humidity = humidity_noise + water_effect
                humidity = max(0, min(1, humidity))

                tile.humidity = humidity
                self.elevation_map[x][y] = tile.elevation
                self.humidity_map[x][y] = humidity

    def temperature(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[x][y]
                temperature = 1 - abs(tile.elevation * 1.8)
                tile.temperature = temperature
                self.temperature_map[x][y] = temperature

    def compute_habitability(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[x][y]
                if tile.elevation < 0:
                    tile.habitability = -1
                    self.habitability_map[x][y] = -1
                score = 0
                # Water proximity
                dist = self.distance_map[x, y]
                score += np.exp(-dist * 0.3)
                score += max(0, tile.river)
                # Humidity
                score += 2*(0.6 - tile.humidity)
                # Temperature (let's say the optimum is 0.6)
                score += 2*(0.6 - tile.temperature)
                # Elevation
                score -= max(tile.elevation, 0) * 0.5
                tile.habitability = score
                self.habitability_map[x][y] = score

    def habitability_local_max(self) -> list[tuple[int | float]]:
        hmap = self.habitability_map

        # local max in a 5x5 neighborhood
        local_max = maximum_filter(hmap, size=15)
        maxima_mask = (hmap == local_max)
        maxima_mask &= self.is_land

        coords = np.argwhere(maxima_mask)
        scores = [hmap[x][y] for x, y in coords]

        locations = list(zip(scores, coords[:,0], coords[:,1]))
        locations.sort(reverse=True)

        return locations     
    
    def generate_rivers(self, n_sources=20, max_length=800) -> None:
        self.river_flow = np.zeros((self.width, self.height))
        
        mountain_tiles = [(tile.x, tile.y) for row in self.tiles for tile in row if tile.elevation > 0.4]
        sources = random.sample(mountain_tiles, min(n_sources, len(mountain_tiles)))
        
        for sx, sy in sources:
            x, y = sx, sy
            path = []
            for _ in range(max_length):
                tile = self.tiles[x][y]
                path.append((x, y))
                if tile.elevation < 0:
                    break

                neigh_coords = tile.neighbors(self)
                candidates = []
                for nx, ny in neigh_coords:
                    h = self.tiles[nx][ny].elevation
                    if h <= tile.elevation + 0.02:  
                        candidates.append((h,nx,ny))
                
                if not candidates:
                    break
                
                candidates.sort()
                _, x, y = random.choice(candidates[:3]) 
            for px, py in path:
                self.river_flow[px, py] += 1
                self.tiles[px][py].river = 1

        
    def compute_biomes(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[x][y]
                tile.biome = tile.compute_biome()

    def generate(self) -> None:
        self.elevation()
        self.add_mountains()
        self.compute_distance_to_water()
        self.humidity()
        self.temperature()
        self.generate_rivers()
        self.compute_habitability()
        self.compute_biomes()

    def biome_map(self) -> np.ndarray:

        img = np.zeros((self.height, self.width, 3))

        for x in range(self.width):
            for y in range(self.height):

                biome = self.tiles[x][y].biome
                # conversion RGB (0-255) → matplotlib (0-1)
                r, g, b = colors[biome]
                img[y][x] = [r/255, g/255, b/255]

        return img
