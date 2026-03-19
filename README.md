# civilization-sim
A procedural civilization simulator where tribes emerge, migrate, build cities, wage wars, and adapt to climate over centuries. Each world is uniquely generated, with evolving societies driven by resources, geography, and AI-based decision making.

# Animation process
FuncAnimation (matplotlib)
    │
    └── Simulation.animate(frame)        ← cosmétique uniquement
            │
            ├── World.step(year, tribes) ← toute la logique monde
            │       └── tribe.step(year, all_tribes)  ← toute la logique tribu
            │               ├── migrate()
            │               ├── reproduce()    ← à implémenter
            │               ├── death()        ← à implémenter
            │               ├── trade()        ← à implémenter
            │               └── war()          ← à implémenter
            │
            └── scatter.set_offsets(...)  ← mise à jour visuelle
