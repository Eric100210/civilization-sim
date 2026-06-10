# civilization-sim
A procedural civilization simulator where tribes emerge, migrate, build cities, wage wars, and adapt to climate over centuries. Each world is uniquely generated, with evolving societies driven by resources, geography, and AI-based decision making.

# Running

```bash
pip install -r requirements.txt
python main.py          # animated simulation (matplotlib window)
python debug_sim.py     # headless run with yearly metrics logged
pytest tests/           # behavioural invariant tests
```

Simulation parameters (map size, number of tribes, years, rivers…) are set in the `params` dict at the top of `main.py`.

# World generation (`data/map.py`)

- **Elevation**: layered Perlin noise (continents + detail) with ridge noise for mountain chains.
- **Rivers**: stochastic descent on a sink-filled elevation map (Priority-Flood), so every river reaches the ocean. `RIVER_RANDOMNESS` controls meandering.
- **Climate**: humidity (noise + proximity to water) and temperature (elevation-based) drive biomes: ocean, plains, desert, mountains, snow, river.
- **Habitability**: combines water proximity, temperate climate, river valleys and alluvial plains, minus an altitude penalty. Tribes' decisions are based on this map.
- **Resources** per biome: food/water on plains and rivers, stone/iron/gold in mountains and deserts, wood on plains.

# Tribe mechanics (`data/tribe.py`)

Each tribe, once per year:

- **Migration** (nomadic phase, pop < 80): scouts a vision radius that grows with population and relocates toward better land, with an attachment penalty for leaving good land. Long-range memory of good spots found while exploring.
- **Resources**: harvests its territory; food and water are a yearly flux, stone/iron/gold/wood accumulate. An expedition (`exploration()`) walks beyond the borders, harvests, scouts migration targets and detects neighbouring tribes.
- **Population**: birth rate driven by habitat quality and food surplus; death rate driven by crowding, reduced by technology. Shortages of food or water kill proportionally.
- **Expansion** (sedentary phase) via three mechanisms:
  1. *Pressure* — budding off new tiles when population approaches carrying capacity (> 75% of K);
  2. *Opportunistic* — slow colonization of adjacent tiles more fertile than the current average;
  3. *Need-driven* — colonization of border tiles holding resources required for the next era (e.g. mountains for iron).
- **Technology**: four eras (Nomadic → Neolithic → Bronze Age → Iron Age), each unlocked by spending accumulated resources, lowering mortality and extending expeditions (`data/resources.py`).
- **War**: triggered by a temptation score (aggressiveness, enemy wealth, fertility gain, size ratio). Yearly attrition follows a Lanchester model weighted by era; the front advances proportionally to the loser's territory (~1%/year). Wars end by territorial capitulation (≥25% lost), one-sided capitulation (≥50% losses vs an intact opponent), mutual exhaustion, or a 40-year timeout — followed by pillage, partial assimilation and a truce. A tribe collapsing below 5 inhabitants goes extinct and is absorbed by the victor.
- **Trade**: not implemented yet (stub — a refused trade is meant to raise war temptation).

# Animation process

```
FuncAnimation (matplotlib)
    │
    └── Simulation.animate(frame)        ← displaying
            │
            ├── World.step(year, tribes) ← all the 'world' logic
            │       └── tribe.step(year, all_tribes)  ← all the 'tribe' logic
            │               ├── migrate()
            │               ├── get_resources()  (incl. exploration)
            │               ├── population_growth()
            │               ├── expand()
            │               ├── eat()
            │               ├── get_technology()
            │               ├── _check_extinction()
            │               └── war()            ← trade() à implémenter
            │
            └── scatter.set_offsets(...)  ← visual update of tribes
```

The right-hand panel shows each tribe's era, population, surface, vital rates and war/truce status.

# Tests

`tests/test_tribe.py` checks behavioural invariants rather than exact values (the simulation is stochastic): territorial integrity, resource flux vs stock, extinction rules, and regressions on tile ownership, pre-saturation expansion, need-driven colonization and war resolution.
