import matplotlib.pyplot as plt
from data.map import World
from data.tribe import Tribe

# -------Paramètres de simulation------
WIDTH = 300
HEIGHT = 200
NB_TRIBES = 3


# ------------Simulation---------------
world = World(WIDTH, HEIGHT)
world.generate()
tribes = [Tribe(world) for _ in range(NB_TRIBES)]

plt.imshow(world.biome_map())
plt.title(f"Generated World (seed={world.seed})")
plt.axis("off")
plt.show()