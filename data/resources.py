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
        "death_rate_bonus": 0.0,
        "exploration_bonus": 0,
    },
    1: {
        "name": "Neolithic",
        "unlock": {ResourceType.FOOD: 25, ResourceType.STONE: 10},
        "death_rate_bonus": 0.0005,  # -0.05% mortality
        "exploration_bonus": 1,  # +1 tile of exploration range
    },
    2: {
        "name": "Bronze Age",
        "unlock": {ResourceType.IRON: 50, ResourceType.WOOD: 50},
        "death_rate_bonus": 0.001,
        "exploration_bonus": 3,
    },
    3: {
        "name": "Iron Age",
        "unlock": {ResourceType.IRON: 150, ResourceType.GOLD: 30},
        "death_rate_bonus": 0.002,
        "exploration_bonus": 6,
    },
}
