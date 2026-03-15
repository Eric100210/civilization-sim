import random
import numpy as np
import math
from .map import World


class Tribe():
    def __init__(self, world: World):
        self.world = world
        self.population = None
        self.resources = None
        self.territory = None
        self.technology = None
        self.aggressiveness = None

    def spawn(self):
        land = self.world.is_land
        habitability_map = self.world.habitability_map

