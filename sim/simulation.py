from data.tribe import Tribe
from data.map import World
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


TRIBE_COLORS = ["#640C0C", "#3F250C", "#774F1A"]


class Simulation:
    def __init__(self, world: World, tribes: list[Tribe], params: dict[str, int]):
        self.world = world
        self.tribes = tribes
        self.params = params
        self.scatters = []
        self.ax = None

    def animate(self, frame: int) -> tuple:
        """
        Called once per frame by FuncAnimation.
        1. Advance the simulation by one year (all logic lives in world.step)
        2. Update the visual
        """
        # --- simulation tick ---
        self.world.step(frame, self.tribes)

        # --- rendering only below this line ---
        for i, tribe in enumerate(self.tribes):
            if tribe.alive:
                xs = [x for x, y in tribe.territory]
                ys = [y for x, y in tribe.territory]
                self.scatters[i].set_offsets(np.c_[xs, ys])
            else:
                self.scatters[i].set_offsets(np.empty((0, 2)))

        self.ax.set_title(f"Year {frame}")
        return tuple(self.scatters)

    def display_map(self) -> None:
        fig, self.ax = plt.subplots()
        img = self.world.biome_map()
        # origin="lower" so that y=0 is at the bottom, matching tiles[x][y] coords
        self.ax.imshow(img, origin="lower")
        self.ax.set_title("Year 0")
        self.ax.axis("off")

        for i, tribe in enumerate(self.tribes):
            xs = [x for x, y in tribe.territory]
            ys = [y for x, y in tribe.territory]
            sc = self.ax.scatter(xs, ys, c=TRIBE_COLORS[i], s=4, zorder=3, alpha=0.6)
            self.scatters.append(sc)

        ani = animation.FuncAnimation(
            fig,
            self.animate,
            frames=self.params["NB_YEARS"],
            interval=self.params.get("INTERVAL_MS", 50),
            blit=False,
            repeat=False,
        )
        plt.show()

    def start(self) -> None:
        self.display_map()
