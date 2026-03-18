import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data.map import World
from data.tribe import Tribe
from sim.simulation import Simulation

# -------Paramètres de simulation------
params = {
    "WIDTH" : 300,
    "HEIGHT" : 200,
    "NB_TRIBES" : 3,
    "NB_YEARS" : 1000,
}


# ------------Simulation---------------

# --- Initialisation ---
world = World(params["WIDTH"], params["HEIGHT"])
world.generate()
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
