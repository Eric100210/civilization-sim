"""
Quick script to visualise the fixed test world (seed=1090).
Run with: python -m tests.preview_world
"""

import random
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from data.map import World

random.seed(42)
np.random.seed(42)

world = World(width=100, height=100, seed=1090)
world.generate(n_rivers=5, river_randomness=0.3)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Biome map
axes[0].imshow(world.biome_map(), origin="lower")
axes[0].set_title("Biomes — seed=1000")
axes[0].axis("off")

# Habitability map
im = axes[1].imshow(world.habitability_map, origin="lower", cmap="magma")
axes[1].set_title("Habitability — seed=1000")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], label="Habitability")

plt.tight_layout()
plt.show()
