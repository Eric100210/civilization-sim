import matplotlib.pyplot as plt
from data.map import World
from data.tribe import Tribe

# -------Paramètres de simulation------
WIDTH = 300
HEIGHT = 200
NB_TRIBES = 3


# ------------Simulation---------------

# --- Initialisation ---
world = World(WIDTH, HEIGHT)
world.generate()

tribes = []

for _ in range(NB_TRIBES):
    tribe = Tribe(world)
    x, y = tribe.spawn()
    tribes.append(tribe)

# --- Chaque année ---
    """recalculer la map d'habitabilité et les maxima locaux, 
    ça devient un but de s'étendre par là pour chaque civilisation (cf tribe.migrate()), 
    sauf si ça amène trop de dangers de guerre ?"""

initial_positions = [next(iter(tribe.territory)) for tribe in tribes]

for step in range(1000):
    for tribe in tribes:
        tribe.migrate()

positions = [next(iter(tribe.territory)) for tribe in tribes]

print("Début:", initial_positions)
print("Fin:", positions)


plt.imshow(world.biome_map())
plt.title(f"Generated World (seed={world.seed})")
plt.axis("off")
plt.show()