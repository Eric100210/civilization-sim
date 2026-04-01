import random
import numpy as np
from .map import World
from .resources import ResourceType, HISTORICAL_ERAS

HAB_THRESHOLD = 80


class Tribe:
    def __init__(self, world: World):
        self.world = world
        self.population: float | None = None
        self.resources: dict[str, int | float] = {t.value: 0 for t in ResourceType}
        self.territory: set[tuple[int, int]] | None = None
        self.hist_eras: int = 0  # era index into HISTORICAL_ERAS
        self.technology: float = 0
        self.aggressiveness: float = random.uniform(0.1, 0.9)
        self.alive: bool = True
        self.at_war: bool = False
        self.war_enemy: "Tribe" | None = None
        self.war_duration: int = 0  # années écoulées dans la guerre courante
        self.war_pop_start: float = (
            0.0  # population au début de la guerre (seuil exhaustion)
        )
        self.war_ter_start: int = 0  # taille territoire au début (seuil capitulation)
        self.truce_timer: int = 0  # années restantes de paix forcée
        self.trade_refused: bool = (
            False  # mis à True si la tribu ennemie refuse le commerce
        )
        self.nearby_enemies: set["Tribe"] = set()  # ennemis détectés par exploration()
        self.known_good_spots: dict[
            tuple[int, int], float
        ] = {}  # (x,y) -> habitability
        self._cached_border: list | None = None

    def spawn(self) -> tuple[int, int]:
        # argwhere on (height, width) returns (row, col) = (y, x)
        land_positions = np.argwhere(self.world.is_land)
        y, x = random.choice(land_positions)
        self.territory = {(x, y)}
        self.world.tiles[x][y].owner.add(self)  # enregistrer la tile de départ
        # initial population : lognormal distribution
        habit = self.world.habitability_map[y, x]
        self.population = np.random.lognormal(mean=3.0 + 0.1 * habit, sigma=0.4)
        return x, y

    def _territory_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (xs, ys) numpy arrays for the current territory."""
        if not self.territory:
            return np.array([], dtype=int), np.array([], dtype=int)
        arr = np.array(list(self.territory), dtype=int)
        return arr[:, 0], arr[:, 1]

    def step(self, year: int, all_tribes: list["Tribe"]) -> None:
        """Called once per year. Order matters: resources first, then decisions."""
        if not self.alive:
            return
        self._cached_border = None

        # Pendant la guerre : pas de migration ni d'expansion territoriale
        if not self.at_war:
            self.migrate()
        self.get_resources()  # collect this year's resources before any decisions
        self.population_growth()  # sees fresh food from this year
        if not self.at_war:
            self.expand()
        self.eat()
        self.get_technology()
        self._check_extinction()
        # self.trade(all_tribes)
        self.war(all_tribes)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def migrate(self) -> None:
        """
        Two-phase migration model inspired by Ammerman & Cavalli-Sforza (1984):

        1. EXPLORE — scout a radius of `vision` tiles and find the best spot.
           Vision grows with population (larger groups scout further).
        2. DECIDE — migrate only if the best spot is significantly better
           than current location AND the tribe isn't already well-settled.
           The threshold prevents endless restlessness in already-good zones.

        Only active during the nomadic phase (population < SEDENTARY_THRESHOLD).
        Once sedentary, expand() takes over territorial growth.
        """
        if self.population >= HAB_THRESHOLD:
            return

        x, y = next(iter(self.territory))
        habit_map = self.world.habitability_map
        current_habit = habit_map[y, x]

        # Vision radius: grows logarithmically with population (4–15 tiles)
        pop = max(1, self.population or 1)
        vision = int(np.clip(3 + np.log(pop / 30 + 1) * 2.5, 4, 15))

        # Scan all tiles within vision radius
        best_h, bx, by = current_habit, x, y
        for dy in range(-vision, vision + 1):
            for dx in range(-vision, vision + 1):
                if dx * dx + dy * dy > vision * vision:
                    continue  # circular radius
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                    h = habit_map[ny, nx]
                    if h > best_h:
                        best_h, bx, by = h, nx, ny

        # Migrate with a probability that depends on:
        #   - relative gain  : how much better is the destination?
        #   - attachment     : harder to leave already-good land
        #
        # p = sigmoid-like curve:
        #   gain < 0        → never migrate (destination is worse)
        #   gain = 0.2      → ~30% chance on average land
        #   gain = 0.5      → ~60% chance
        #   gain = 1.0+     → ~90% chance
        # Divided by (1 + current_habit * 0.3) so well-settled tribes
        # need a proportionally larger gain to bother moving.
        gain = best_h - current_habit
        if gain <= 0:
            # No immediate improvement in vision range — check long-range memory
            if self.known_good_spots:
                best_known_hab = max(self.known_good_spots.values())
                if best_known_hab > current_habit + 0.3:
                    best_known_pos = max(
                        self.known_good_spots, key=self.known_good_spots.get
                    )
                    bkx, bky = best_known_pos
                    step_x = int(np.sign(bkx - x))
                    step_y = int(np.sign(bky - y))
                    cx, cy = x + step_x, y + step_y
                    if (
                        0 <= cx < self.world.width
                        and 0 <= cy < self.world.height
                        and self.world.is_land[cy, cx]
                        and random.random() < 0.4
                    ):
                        self.territory = {(cx, cy)}
                        self._cached_border = None
            return
        attachment = 1.0 + current_habit * 0.3
        p = min(1.0, (gain / attachment) * 1.5)
        if random.random() < p:
            self.territory = {(bx, by)}
            self._cached_border = None

    def _carrying_capacity(self) -> float:
        """Sum of habitability over owned tiles, scaled by HAB_THRESHOLD."""
        xs, ys = self._territory_coords()
        if len(xs) == 0:
            return 1.0
        return max(
            1.0, float(self.world.habitability_map[ys, xs].sum()) * HAB_THRESHOLD
        )

    def _border_tiles(self) -> list[tuple[int, int]]:
        """All land tiles adjacent to the territory but not yet owned."""
        if self._cached_border is not None:
            return self._cached_border
        candidates = set()
        for x, y in self.territory:
            for nx, ny in self.world.tiles[x][y].neighbors(self.world):
                if self.world.is_land[ny, nx] and (nx, ny) not in self.territory:
                    candidates.add((nx, ny))
        self._cached_border = list(candidates)
        return self._cached_border

    def population_growth(self) -> None:
        n = len(self.territory)
        xs, ys = self._territory_coords()
        avg_hab = float(self.world.habitability_map[ys, xs].mean()) if n > 0 else 0.0
        density = self.population / max(1, n)

        # Natality: base pre-modern rate + habitat quality + food surplus bonus
        # Food bonus uses log to avoid runaway growth: large territories collect
        # enormous food totals, so a linear coefficient causes birth_rate > 1.
        food = self.resources[ResourceType.FOOD.value]
        self.birth_rate = 0.015 + avg_hab * 0.010 + np.log1p(food) * 0.001

        # Mortality: base pre-modern rate + density crowding - era technology bonus
        death_rate_bonus = HISTORICAL_ERAS[self.hist_eras]["death_rate_bonus"]
        self.death_rate = max(0.0, 0.008 + density * 0.000088 - death_rate_bonus)

        r = self.birth_rate - self.death_rate
        self.population = max(1.0, self.population * (1 + r))

    def expand(self) -> None:
        """
        Two mechanisms of territorial expansion:

        1. PRESSURE — overpopulation (P > K): the tribe must expand to survive.
           Number of new tiles ~ Poisson(pressure - 1).

        2. OPPORTUNISTIC — even without pressure, the tribe slowly colonizes
           adjacent tiles that are significantly more fertile than its current average.
        """
        if not self.territory:
            return

        if self.population < HAB_THRESHOLD:
            return

        K = self._carrying_capacity()
        pressure = self.population / K
        border = self._border_tiles()

        if not border:
            return

        # 1. Pressure-driven expansion
        if pressure > 1.0:
            n_tiles = np.random.poisson(lam=pressure - 1.0)
            for _ in range(n_tiles):
                if not border:
                    break
                idx = random.randrange(len(border))
                chosen = border[idx]
                border[idx] = border[-1]
                border.pop()
                if self.world.tiles[chosen[0]][chosen[1]].owner:
                    continue  # tile déjà prise → on essaie la suivante, pas stop
                self.territory.add(chosen)
                self.world.tiles[chosen[0]][chosen[1]].owner.add(self)

        # 2. Opportunistic expansion toward more fertile land
        avg_hab = K / (len(self.territory) * HAB_THRESHOLD)
        for nx, ny in border:
            tile_hab = self.world.habitability_map[ny, nx]
            if tile_hab > avg_hab and random.random() < 0.05:
                if not self.world.tiles[nx][ny].owner:
                    self.territory.add((nx, ny))
                    self.world.tiles[nx][ny].owner.add(self)

    def exploration(self) -> dict[str, int | float]:
        """
        Simulates an expedition leaving from a random border tile and walking
        length_exploration steps in a roughly consistent direction.

        At each step, the walker picks from the 3 neighbors closest to its
        current heading (forward-left, forward, forward-right), biased toward
        staying on course. This produces a realistic linear expedition that
        slightly meanders rather than covering the whole surrounding area.
        """
        harvest = {t.value: 0.0 for t in ResourceType}

        border = self._border_tiles()
        if not border:
            return harvest

        length_exploration = 7 + HISTORICAL_ERAS[self.hist_eras]["exploration_bonus"]
        exploration_efficiency = 0.2

        start = random.choice(border)
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        x, y = start
        outward_dirs = [
            (dx, dy) for dx, dy in dirs if (x + dx, y + dy) not in self.territory
        ]
        if not outward_dirs:
            return harvest  # surrounded — can't explore
        heading = random.choice(outward_dirs)

        # visited includes the full territory so the walk never crosses back into it
        visited = set(self.territory) | {(x, y)}

        for _ in range(length_exploration):
            hdx, hdy = heading

            # Candidate directions: forward-left, forward, forward-right
            # i.e. the 3 directions whose dot product (produit scalaire) with heading is highest
            scored = sorted(dirs, key=lambda d: d[0] * hdx + d[1] * hdy, reverse=True)
            candidates = scored[:3]  # top 3 most aligned with heading

            # Try candidates in order until we find a valid land tile
            moved = False
            random.shuffle(candidates)
            for dx, dy in candidates:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.world.width
                    and 0 <= ny < self.world.height
                    and self.world.is_land[ny, nx]
                    and (nx, ny) not in visited
                ):
                    x, y = nx, ny
                    heading = (dx, dy)
                    visited.add((x, y))
                    moved = True
                    break

            if not moved:
                break  # blocked (ocean, map edge) — expedition ends early

            # Détection d'une tribu ennemie : tile possédée OU voisine d'un territoire ennemi
            owners = self.world.tiles[x][y].owner
            for other in owners:
                if other is not self and other.alive:
                    self.nearby_enemies.add(other)
            # Vérifier aussi les 8 voisins : si l'un appartient à une autre tribu, on la détecte
            for dx2, dy2 in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx2, ny2 = x + dx2, y + dy2
                if 0 <= nx2 < self.world.width and 0 <= ny2 < self.world.height:
                    for other in self.world.tiles[nx2][ny2].owner:
                        if other is not self and other.alive:
                            self.nearby_enemies.add(other)

            for t in ResourceType:
                harvest[t.value] += (
                    self.world.resource_maps[t.value][y, x] * exploration_efficiency
                )

            # Record high-habitability tiles as potential migration targets
            tile_hab = self.world.habitability_map[y, x]
            if tile_hab > 1.8:
                if (x, y) not in self.known_good_spots or self.known_good_spots[
                    (x, y)
                ] < tile_hab:
                    self.known_good_spots[(x, y)] = tile_hab

        return harvest

    def get_resources(self) -> None:
        """Ressources sur les terres + exploration (def exploration) des zones alentours, surtout les montagnes pour les minerais"""
        # Consumables (water, food): reset each year — they are a flux, not a stock
        self.resources[ResourceType.WATER.value] = 0
        self.resources[ResourceType.FOOD.value] = 0
        # Accumulables (stone, iron, gold, wood): keep existing stock, add this year's harvest
        xs, ys = self._territory_coords()
        if len(xs) > 0:
            for t in ResourceType:
                self.resources[t.value] += float(
                    self.world.resource_maps[t.value][ys, xs].sum()
                )
        # Exploring to get more resources
        explored = self.exploration()
        for t in ResourceType:
            self.resources[t.value] += explored[t.value]

    def eat(self) -> None:
        water_needed = int(self.population / 100)
        food_needed = int(self.population / 100)

        water_lack = max(0, water_needed - self.resources[ResourceType.WATER.value])
        food_lack = max(0, food_needed - self.resources[ResourceType.FOOD.value])

        self.resources[ResourceType.WATER.value] -= water_needed
        self.resources[ResourceType.FOOD.value] -= food_needed

        self.population -= water_lack * 2
        self.population -= food_lack * 1
        self.population = max(0.0, self.population)

    def get_technology(self) -> None:
        """
        Unlock the next technological era if the tribe has accumulated
        enough resources. Resources are consumed on unlock.
        """
        next_era = self.hist_eras + 1
        if next_era not in HISTORICAL_ERAS:
            return
        requirements = HISTORICAL_ERAS[next_era]["unlock"]
        if all(self.resources[t.value] >= qty for t, qty in requirements.items()):
            for t, qty in requirements.items():
                self.resources[t.value] -= qty
            self.hist_eras = next_era

    def _check_extinction(self) -> None:
        """Mark tribe extinct if population falls below minimum viable threshold."""
        if self.population < 5:
            self.alive = False
            self.population = 0.0
            # Release ownership of all tiles
            for x, y in self.territory:
                self.world.tiles[x][y].owner.discard(self)
            self.territory = set()

    def _war_temptation(self, enemy: "Tribe") -> float:
        """
        Score de tentation de guerre contre `enemy`, entre 0 et 1.

        Facteurs :
          + aggressiveness        : trait permanent de la tribu (0.1–0.9)
          + enemy_resource_score  : richesse de l'ennemi (ce qu'on peut piller)
          - own_stability         : si on est riche et stable, moins envie de risquer
          + hab_gain              : leur territoire est-il plus fertile que le nôtre ?
          + trade_refused_bonus   : +0.25 si l'ennemi a refusé le commerce
          - size_ratio_penalty    : on évite les tribus 3x plus grandes
        """
        # Richesse de l'ennemi (ressources accumulables normalisées)
        enemy_wealth = (
            enemy.resources[ResourceType.IRON.value]
            + enemy.resources[ResourceType.GOLD.value]
            + enemy.resources[ResourceType.STONE.value]
        )
        enemy_resource_score = min(1.0, enemy_wealth / 500.0)

        # Stabilité propre : si on a beaucoup de ressources, moins de raison d'attaquer
        own_wealth = (
            self.resources[ResourceType.FOOD.value]
            + self.resources[ResourceType.WATER.value] * 10
        )
        own_stability = min(1.0, own_wealth / 1000.0)

        # Gain d'habitabilité : moyenne de leur territoire vs le nôtre
        if enemy.territory:
            ex, ey = zip(*enemy.territory)
            enemy_avg_hab = float(
                self.world.habitability_map[list(ey), list(ex)].mean()
            )
        else:
            enemy_avg_hab = 0.0
        xs, ys = self._territory_coords()
        own_avg_hab = (
            float(self.world.habitability_map[ys, xs].mean()) if len(xs) > 0 else 0.0
        )
        hab_gain = max(0.0, (enemy_avg_hab - own_avg_hab) / 2.0)  # normalisé 0–1

        # Bonus si commerce refusé
        trade_bonus = 0.25 if self.trade_refused else 0.0

        # Pénalité si l'ennemi est beaucoup plus grand
        size_ratio = enemy.population / max(1.0, self.population)
        size_penalty = min(0.6, max(0.0, (size_ratio - 1.5) * 0.3))

        score = (
            self.aggressiveness * 0.35
            + enemy_resource_score * 0.25
            - own_stability * 0.15
            + hab_gain * 0.20
            + trade_bonus
            - size_penalty
        )
        return float(np.clip(score, 0.0, 1.0))

    def trade(self, all_tribes: list["Tribe"]) -> None:
        pass

    def war(self, all_tribes: list["Tribe"]) -> None:
        """
        Deux modes :
          A) En cours de guerre : un tour de Lanchester + avance du front + check fin
          B) Déclenchement : évaluer _war_temptation() contre chaque ennemi proche
        """
        # --- Décompte de la trêve ---
        if self.truce_timer > 0:
            self.truce_timer -= 1

        # --- A) Tour de guerre en cours ---
        # On ne traite que si c'est self qui a déclaré (pour éviter double traitement)
        if self.at_war and self.war_enemy and self.war_enemy.alive:
            enemy = self.war_enemy
            # Évite le double traitement : seule la tribu avec le plus petit id() résout
            if id(self) > id(enemy):
                self.nearby_enemies.clear()
                return

            self.war_duration += 1
            enemy.war_duration = self.war_duration  # synchroniser les deux

            # --- Lanchester : pertes annuelles ---
            # alpha/beta = efficacité de combat, fonction de l'ère technologique
            alpha = 0.008 + self.hist_eras * 0.004  # efficacité de self
            beta = 0.008 + enemy.hist_eras * 0.004  # efficacité de enemy

            losses_self = beta * enemy.population * 0.01
            losses_enemy = alpha * self.population * 0.01

            self.population = max(0.0, self.population - losses_self)
            enemy.population = max(0.0, enemy.population - losses_enemy)

            # --- Avance du front : le vainqueur du tour prend 2-3 tiles ---
            # Le vainqueur du tour = celui qui a infligé le plus de pertes
            if losses_enemy > losses_self:
                attacker, defender = self, enemy
            else:
                attacker, defender = enemy, self

            n_tiles_front = random.randint(2, 3)
            self._transfer_border_tiles(attacker, defender, pct=None, n=n_tiles_front)

            # --- Conditions de fin ---
            territory_lost_self = 1.0 - len(self.territory) / max(1, self.war_ter_start)
            territory_lost_enemy = 1.0 - len(enemy.territory) / max(
                1, enemy.war_ter_start
            )
            pop_lost_self = 1.0 - self.population / max(1.0, self.war_pop_start)
            pop_lost_enemy = 1.0 - enemy.population / max(1.0, enemy.war_pop_start)

            TERRITORY_SURRENDER = 0.25  # capitulation à 25% de territoire perdu
            EXHAUSTION_THRESHOLD = 0.30  # armistice si 30% de pop perdue
            MAX_WAR_YEARS = 15

            capitulation = (
                territory_lost_self >= TERRITORY_SURRENDER
                or territory_lost_enemy >= TERRITORY_SURRENDER
            )
            exhaustion = (
                pop_lost_self >= EXHAUSTION_THRESHOLD
                and pop_lost_enemy >= EXHAUSTION_THRESHOLD
            )
            timeout = self.war_duration >= MAX_WAR_YEARS

            if capitulation or exhaustion or timeout:
                self._resolve_war(enemy, capitulation, exhaustion)

            self.nearby_enemies.clear()
            return

        # --- B) Déclenchement ---
        if self.at_war or self.truce_timer > 0 or not self.nearby_enemies:
            self.nearby_enemies.clear()
            return

        best_enemy = None
        best_score = 0.0
        for candidate in self.nearby_enemies:
            if not candidate.alive or candidate.at_war:
                continue
            score = self._war_temptation(candidate)
            if score > best_score:
                best_score = score
                best_enemy = candidate

        WAR_THRESHOLD = 0.55
        if best_enemy is not None and best_score >= WAR_THRESHOLD:
            # Initialisation de la guerre pour les deux camps
            for tribe, opp in ((self, best_enemy), (best_enemy, self)):
                tribe.at_war = True
                tribe.war_enemy = opp
                tribe.war_duration = 0
                tribe.war_pop_start = tribe.population
                tribe.war_ter_start = len(tribe.territory)
            print(
                f"[Guerre] Déclaration : pop {int(self.population)} "
                f"vs {int(best_enemy.population)} "
                f"(score={best_score:.2f}, aggr={self.aggressiveness:.2f})"
            )

        self.nearby_enemies.clear()

    def _resolve_war(
        self, enemy: "Tribe", capitulation: bool, exhaustion: bool
    ) -> None:
        """
        Fin de guerre selon la cause :
          - Capitulation (≥25% territoire perdu) : le perdant cède + pillage
          - Exhaustion mutuelle                  : armistice, pas de cession
          - Timeout                              : armistice léger
        """
        ter_lost_self = 1.0 - len(self.territory) / max(1, self.war_ter_start)
        ter_lost_enemy = 1.0 - len(enemy.territory) / max(1, enemy.war_ter_start)

        if capitulation:
            # Le plus grand perdant territorial est le vaincu
            if ter_lost_self >= ter_lost_enemy:
                loser, winner = self, enemy
            else:
                loser, winner = enemy, self
            # Pillage proportionnel à l'ampleur de la victoire
            pillage_pct = min(0.50, 0.20 + (ter_lost_self - ter_lost_enemy) * 0.5)
            self._pillage(winner, loser, pct=pillage_pct)
            truce_duration = random.randint(8, 20)
            print(
                f"[Guerre] Capitulation — perdant a cédé "
                f"{max(ter_lost_self, ter_lost_enemy) * 100:.0f}% de son territoire. "
                f"Trêve {truce_duration} ans."
            )
        else:
            # Exhaustion ou timeout : armistice symétrique
            truce_duration = random.randint(15, 30)
            print(
                f"[Guerre] Armistice ({'exhaustion' if exhaustion else 'timeout'}) "
                f"après {self.war_duration} ans. Trêve {truce_duration} ans."
            )

        # Reset état de guerre pour les deux
        for tribe in (self, enemy):
            tribe.at_war = False
            tribe.war_enemy = None
            tribe.war_duration = 0
            tribe.war_pop_start = 0.0
            tribe.war_ter_start = 0
            tribe.truce_timer = truce_duration
            tribe._cached_border = None

    def _transfer_border_tiles(
        self,
        winner: "Tribe",
        loser: "Tribe",
        pct: float | None = None,
        n: int | None = None,
    ) -> None:
        """
        Transfère des tiles de contact du perdant au vainqueur.
        Passer soit `pct` (fraction du total), soit `n` (nombre fixe).
        """
        if not loser.territory:
            return
        # Tiles du perdant adjacentes au territoire du vainqueur (zone de contact)
        contact = [
            (x, y)
            for (x, y) in loser.territory
            if any(
                (nx, ny) in winner.territory
                for nx, ny in self.world.tiles[x][y].neighbors(self.world)
            )
        ]
        if not contact:
            contact = list(loser.territory)  # fallback : tiles quelconques

        if n is not None:
            n_transfer = min(n, len(contact))
        else:
            n_transfer = max(1, int(len(contact) * (pct or 0.1)))
        transferred = random.sample(contact, n_transfer)

        for x, y in transferred:
            loser.territory.discard((x, y))
            self.world.tiles[x][y].owner.discard(loser)
            winner.territory.add((x, y))
            self.world.tiles[x][y].owner.add(winner)

        # Invalider les caches de bordure
        winner._cached_border = None
        loser._cached_border = None

    def _pillage(self, winner: "Tribe", loser: "Tribe", pct: float) -> None:
        """Le vainqueur vole pct% des ressources accumulables du perdant."""
        accumulables = [
            ResourceType.IRON,
            ResourceType.GOLD,
            ResourceType.STONE,
            ResourceType.WOOD,
        ]
        for t in accumulables:
            stolen = loser.resources[t.value] * pct
            loser.resources[t.value] -= stolen
            winner.resources[t.value] += stolen
