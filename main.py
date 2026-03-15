import matplotlib.pyplot as plt
from data import map, tribe

# -------Paramètres de simulation------
WIDTH = 300
HEIGHT = 200
NB_TRIBES = 3


# ------------Simulation---------------
world = map.World(WIDTH, HEIGHT)
world.generate()
tribes = [tribe.Tribe(world) for _ in range(NB_TRIBES)]
breakpoint()

plt.imshow(world.biome_map())
plt.title(f"Generated World (seed={world.seed})")
plt.axis("off")
plt.show()