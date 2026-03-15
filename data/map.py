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

    def __init__(self, elevation, temperature, humidity):
        self.elevation = elevation
        self.temperature = temperature
        self.humidity = humidity
        self.biome = self.compute_biome()

    def compute_biome(self):

        if self.elevation < 0:
            return "ocean"

        if self.elevation > 0.45:
            if self.temperature < 0:
                return "snow"
            return "mountains"

        if self.temperature > 0.7 and self.humidity < 0.1:
            return "desert"

        return "plains"


class World:

    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 100000)
        self.humidity_seed = random.randint(0,100000)
        self.tiles = [[None for _ in range(height)] for _ in range(width)]

    def wind_direction(y, height):
        latitude = y / height
        if latitude < 0.33:
            return 1   # vent vers l'est
        elif latitude < 0.66:
            return -1  # vent vers l'ouest
        else:
            return 1

    def noises(self, seed):
        noise1 = PerlinNoise(octaves=3, seed=seed)
        noise2 = PerlinNoise(octaves=6, seed=seed)
        noise3 = PerlinNoise(octaves=12, seed=seed)
        noise4 = PerlinNoise(octaves=24, seed=seed)
        return (noise1, noise2, noise3, noise4)

    def generate(self):
        elevation_seeds = self.noises(self.seed)
        humidity_seeds = self.noises(self.humidity_seed)
        for y in range(self.height):
            for x in range(self.width):

                elevation = elevation_seeds[0]([x/self.width, y/self.height])
                elevation += 0.5 * elevation_seeds[1]([x/self.width, y/self.height])
                elevation += 0.25 * elevation_seeds[2]([x/self.width, y/self.height])
                elevation += 0.125 * elevation_seeds[3]([x/self.width, y/self.height])
                elevation += 0.1

                temperature = 1 - abs(elevation * 1.8)

                humidity = humidity_seeds[0]([x/self.width, y/self.height])
                humidity += 0.5 * humidity_seeds[1]([x/self.width, y/self.height])
                humidity += 0.25 * humidity_seeds[2]([x/self.width, y/self.height])
                humidity += 0.125 * humidity_seeds[3]([x/self.width, y/self.height])

                humidity = humidity * (1 - elevation)

                if elevation < 0.1:
                    humidity += 0.6

                tile = Tile(elevation, temperature, humidity)
                self.tiles[x][y] = tile

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