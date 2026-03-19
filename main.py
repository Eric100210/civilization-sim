import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data.map import World
from data.tribe import Tribe
from sim.simulation import Simulation

# -------Paramètres de simulation------
params = {
    "WIDTH": 300,
    "HEIGHT": 200,
    "NB_TRIBES": 3,
    "NB_YEARS": 1000,
    "INTERVAL_MS": 50,  # ms between frames — increase to slow down the animation
    "N_RIVERS": 20,  # number of river sources spawned in mountains
    "RIVER_RANDOMNESS": 0.3,  # 0.0 = steepest descent (straight), 1.0 = fully random (wiggly)
}


# ------------Simulation---------------

# --- Initialisation ---
world = World(params["WIDTH"], params["HEIGHT"])
world.generate(n_rivers=params["N_RIVERS"], river_randomness=params["RIVER_RANDOMNESS"])
world.display_habitability()
tribes = []

for _ in range(params["NB_TRIBES"]):
    tribe = Tribe(world)
    x, y = tribe.spawn()
    tribes.append(tribe)

    # --- Chaque année ---
    """recalculer la map d'habitabilité et les maxima locaux, 
    ça devient un but de s'étendre par là pour chaque civilisation (cf tribe.migrate()), 
    sauf si ça amène trop de dangers de guerre ?"""

simulation = Simulation(world, tribes, params)
simulation.start()
