from enum import Enum


class ResourceType(Enum):
    WATER = "water"
    FOOD = "food"
    WOOD = "wood"
    STONE = "stone"
    IRON = "iron"
    GOLD = "gold"


# Each era requires accumulated resources to unlock.
# Effects are applied in population_growth() and exploration().
HISTORICAL_ERAS = {
    0: {
        "name": "Nomadic",
        "unlock": {},
        "death_rate_bonus": 0.000,
        "exploration_bonus": 0,
    },
    1: {
        "name": "Neolithic",
        "unlock": {ResourceType.FOOD: 40, ResourceType.STONE: 12},
        "death_rate_bonus": 0.002,  # -0.2% mortality (was 0.0005)
        "exploration_bonus": 1,
    },
    2: {
        "name": "Bronze Age",
        "unlock": {ResourceType.IRON: 130, ResourceType.WOOD: 40},
        "death_rate_bonus": 0.004,  # -0.4% mortality (was 0.001)
        "exploration_bonus": 3,
    },
    3: {
        "name": "Iron Age",
        "unlock": {ResourceType.IRON: 180, ResourceType.GOLD: 50},
        "death_rate_bonus": 0.008,  # -0.8% mortality (was 0.002)
        "exploration_bonus": 6,
    },
}
