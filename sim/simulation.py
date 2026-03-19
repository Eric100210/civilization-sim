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

    def tribe_positions(self) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for tribe in self.tribes:
            if tribe.alive:
                x, y = next(iter(tribe.territory))
                xs.append(x)
                ys.append(y)
        return np.array(xs), np.array(ys)

    def animate(self, frame: int) -> tuple:
        """
        Called once per frame by FuncAnimation.
        1. Advance the simulation by one year (all logic lives in world.step)
        2. Update the visual artists — nothing else
        """
        # --- simulation tick ---
        self.world.step(frame, self.tribes)

        # --- rendering only below this line ---
        xs, ys = self.tribe_positions()
        self.scatter.set_offsets(np.c_[xs, ys])
        self.ax.set_title(f"Year {frame}")
        return (self.scatter,)

    def display_map(self) -> None:
        fig, self.ax = plt.subplots()
        img = self.world.biome_map()
        # origin="lower" so that y=0 is at the bottom, matching tiles[x][y] coords
        self.ax.imshow(img, origin="lower")
        self.ax.set_title("Year 0")
        self.ax.axis("off")

        xs, ys = self.tribe_positions()
        self.scatter = self.ax.scatter(xs, ys, c="#59320D", s=10, zorder=3)

        ani = animation.FuncAnimation(
            fig,
            self.animate,
            frames=self.params["NB_YEARS"],
            interval=self.params.get("INTERVAL_MS", 50),
            blit=False,
        )
        plt.show()

    def start(self) -> None:
        self.display_map()
