from data.tribe import Tribe
from data.map import World
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Simulation:
    def __init__(self, world: World, tribes: list[Tribe], params: dict[str, int]):
        self.world = world
        self.tribes = tribes
        self.params = params
        self.scatter = None
        self.ax = None

    def tribe_positions(self):
        xs = []
        ys = []
        for tribe in self.tribes:
            x, y = next(iter(tribe.territory))
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def animate(self, frame):
        """For now, everything that happens during the simulation has to be there,
        To be able to display and animate what is happening"""
        for tribe in self.tribes:
            tribe.migrate()
        xs, ys = self.tribe_positions()
        self.scatter.set_offsets(np.c_[xs, ys])
        self.ax.set_title(f"Year {frame}")
        return (self.scatter,)

    def display_map(self):
        fig, self.ax = plt.subplots()
        img = self.world.biome_map()
        # origin="lower" so that y=0 is at the bottom, matching tiles[x][y] coords
        self.ax.imshow(img, origin="lower")
        self.ax.set_title("Tribe migration simulation")
        self.ax.axis("off")

        xs, ys = self.tribe_positions()
        self.scatter = self.ax.scatter(xs, ys, c="#59320D", s=10, zorder=3)
        ani = animation.FuncAnimation(
            fig, self.animate, frames=self.params["NB_YEARS"], interval=50, blit=False
        )
        plt.show()

    def start(self):
        self.display_map()
