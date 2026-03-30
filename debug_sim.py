"""
Diagnostic script: run simulation headless and log key metrics each year
to understand why tribes go extinct around year 500.
"""
import sys
import random
import numpy as np

from data.map import World
from data.tribe import Tribe

params = {
    "WIDTH": 375,
    "HEIGHT": 250,
    "NB_TRIBES": 3,
    "NB_YEARS": 600,
    "N_RIVERS": 20,
    "RIVER_RANDOMNESS": 0.3,
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

world = World(params["WIDTH"], params["HEIGHT"], seed=SEED)
world.generate(n_rivers=params["N_RIVERS"], river_randomness=params["RIVER_RANDOMNESS"])

tribes = []
for _ in range(params["NB_TRIBES"]):
    tribe = Tribe(world)
    tribe.spawn()
    tribes.append(tribe)

print(f"{'Year':>5} | {'Tribe':>5} | {'Pop':>14} | {'Tiles':>6} | {'BirthR':>8} | {'DeathR':>8} | {'Food':>12} | {'Water':>10} | {'W_need':>8} | {'F_need':>8} | {'Era':>10}")
print("-" * 130)

for year in range(1, params["NB_YEARS"] + 1):
    # Step each tribe manually so we can inspect state mid-step
    for i, tribe in enumerate(tribes):
        if not tribe.alive:
            continue

        # --- replicate tribe.step() but with logging ---
        tribe.migrate()
        tribe.get_resources()

        food = tribe.resources["food"]
        water = tribe.resources["water"]
        tiles = len(tribe.territory)
        pop_before_growth = tribe.population

        tribe.population_growth()
        pop_after_growth = tribe.population

        tribe.expand()

        # compute what eat() will do BEFORE calling it
        water_needed = int(tribe.population / 100)
        food_needed = int(tribe.population / 100)
        water_lack = max(0, water_needed - water)
        food_lack = max(0, food_needed - food)

        tribe.eat()
        tribe.get_technology()
        tribe._check_extinction()

        # Log every 10 years or when something alarming happens
        alarming = (
            water_lack > 0
            or food_lack > 0
            or tribe.birth_rate > 0.05
            or not tribe.alive
        )
        if year % 10 == 0 or alarming:
            status = "[EXTINCT]" if not tribe.alive else ""
            print(
                f"{year:>5} | T{i+1:>4} | {pop_before_growth:>14,.1f} | {tiles:>6} | "
                f"{tribe.birth_rate:>8.4f} | {tribe.death_rate:>8.4f} | "
                f"{food:>12,.1f} | {water:>10,.1f} | "
                f"{water_needed:>8,} | {food_needed:>8,} | "
                f"{tribe.hist_eras:>10} {status}"
            )

    # Print separator every 50 years
    if year % 50 == 0:
        print("-" * 130)
