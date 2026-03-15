import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import random
import math

WIDTH = 300
HEIGHT = 200
SCALE = 100.0


colors = {
    "ocean": (66, 110, 225),
    "plains": (26, 176, 25),
    "desert": (227, 193, 5),
    "mountains": (140, 140, 140),
    "snow": (250, 250, 250)
}


class Tile:

    def __init__(self, elevation, temperature = None, humidity = None):
        self.elevation = elevation
        self.temperature = temperature
        self.humidity = humidity
        self.biome = None

    def compute_biome(self):

        if self.elevation < 0:
            return "ocean"

        if self.elevation > 0.4:
            if self.temperature < 0:
                return "snow"
            return "mountains"

        if self.temperature > 0.7 and self.humidity < 0.15:
            return "desert"

        return "plains"


class World:

    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 100000)
        self.continent_noise = PerlinNoise(octaves=2, seed = self.seed + 1000)
        self.humidity_seed = random.randint(0,100000)
        self.tiles = [[None for _ in range(height)] for _ in range(width)]
        
    def noises(self, seed):
        noise1 = PerlinNoise(octaves=3, seed=seed)
        noise2 = PerlinNoise(octaves=6, seed=seed)
        noise3 = PerlinNoise(octaves=12, seed=seed)
        noise4 = PerlinNoise(octaves=24, seed=seed)
        return (noise1, noise2, noise3, noise4)
        
    def distance_to_water(self, x, y):
        min_dist = 999
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.tiles[nx][ny].elevation < 0:
                        dist = (dx**2 + dy**2) ** 0.5
                        min_dist = min(min_dist, dist)
        return min_dist
    
    def elevation(self):
        noise1 = PerlinNoise(octaves=3, seed=self.seed)
        noise2 = PerlinNoise(octaves=3, seed=self.seed+1)
        noise3 = PerlinNoise(octaves=3, seed=self.seed+2)

        for y in range(self.height):
            for x in range(self.width):

                nx = x / self.width
                ny = y / self.height

                # continents (très basse fréquence)
                continent = self.continent_noise([nx*0.8, ny*0.8])

                # relief principal
                elevation = noise1([nx*2, ny*2])

                # détails
                elevation += 0.5 * noise2([nx*4, ny*4])
                elevation += 0.25 * noise3([nx*8, ny*8])

                # combinaison
                elevation = elevation * 0.75 + continent + 0.05

                tile = Tile(elevation)
                self.tiles[x][y] = tile

    def add_mountains(self):
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

    def humidity(self):
        humidity_seeds = self.noises(self.humidity_seed)
        for y in range(self.height):
            for x in range(self.width):

                tile = self.tiles[x][y]

                distance = self.distance_to_water(x, y)

                water_effect = math.exp(-distance*0.2)

                humidity_noise = humidity_seeds[0]([x/self.width, y/self.height])
                humidity_noise *= 0.15

                humidity = humidity_noise + water_effect
                humidity = max(0, min(1, humidity))

                tile.humidity = humidity

    def temperature(self):
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[x][y]
                temperature = 1 - abs(tile.elevation * 1.8)
                tile.temperature = temperature
        
    def compute_biomes(self):
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[x][y]
                tile.biome = tile.compute_biome()

    def generate(self):
        self.elevation()
        self.add_mountains()
        self.humidity()
        self.temperature()
        self.compute_biomes()

    def biome_map(self):

        img = np.zeros((self.height, self.width, 3))

        for x in range(self.width):
            for y in range(self.height):

                biome = self.tiles[x][y].biome
                # conversion RGB (0-255) → matplotlib (0-1)
                r, g, b = colors[biome]
                img[y][x] = [r/255, g/255, b/255]

        return img


world = World(WIDTH, HEIGHT)
world.generate()



plt.imshow(world.biome_map())
plt.title(f"Generated World (seed={world.seed})")
plt.axis("off")
plt.show()