from data.tribe import Tribe
from data.map import World
from data.resources import HISTORICAL_ERAS
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

        # Update each tribe's info block in the right panel
        for i, tribe in enumerate(self.tribes):
            if tribe.alive:
                pop = int(tribe.population)
                tiles = len(tribe.territory)
                era = HISTORICAL_ERAS[tribe.hist_eras]["name"]
                text = (
                    f"── Tribe {i + 1} ──\n"
                    f"Era      : {era}\n"
                    f"Pop      : {pop:,}\n"
                    f"Surface  : {tiles} km²\n"
                    f"Birth    : {tribe.birth_rate:.4f}\n"
                    f"Death    : {tribe.death_rate:.4f}\n"
                    f"Resources: {tribe.resources['iron']}\n"
                )
            else:
                text = f"── Tribe {i + 1} ──\n[extinct]"
            self.pop_texts[i].set_text(text)

        return tuple(self.scatters) + tuple(self.pop_texts)

    def display_map(self) -> None:
        fig = plt.figure(figsize=(14, 7))
        # Left column (map) takes 3/4 of width, right column (panel) takes 1/4
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        self.ax = fig.add_subplot(gs[0])
        self.panel = fig.add_subplot(gs[1])

        # Map
        img = self.world.biome_map()
        self.ax.imshow(img, origin="lower")
        self.ax.set_title("Year 0")
        self.ax.axis("off")

        # Right panel: text board
        self.panel.axis("off")

        for i, tribe in enumerate(self.tribes):
            xs = [x for x, y in tribe.territory]
            ys = [y for x, y in tribe.territory]
            sc = self.ax.scatter(xs, ys, c=TRIBE_COLORS[i], s=4, zorder=3, alpha=0.6)
            self.scatters.append(sc)

        # One text block per tribe in the right panel, stacked vertically
        self.pop_texts = []
        n = len(self.tribes)
        for i in range(n):
            y_pos = 0.95 - i * (0.9 / n)
            txt = self.panel.text(
                0.05,
                y_pos,
                "",
                transform=self.panel.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                family="monospace",
                color=TRIBE_COLORS[i],
            )
            self.pop_texts.append(txt)

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
