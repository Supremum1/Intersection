
import random
import numpy as np

from model.geometry import RoadGeometry
from model.car      import Car

from .config import (
    DT, LANES, MAX_CARS, SPAWN_RATE_PER_LANE,
    V_APPROACH,
    PLAN_DIST,
    TAU_REACT_MIN, TAU_REACT_MAX,
    RHO_MIN, RHO_MAX,
    SIGMA_MIN, SIGMA_MAX,
    LOG_LEVEL_DEFAULT,
    LANE_PRIORS,
)
from .priority import lane_role
from .decision  import compute_target_accel, derive_indicators, A_BRAKE_MAX, A_MAX_POS
from .config    import A_BRAKE_CMF, V_CREEP


def _classify(a_target, car):
    """Три состояния, перечисленные в постановке: едет / крадётся / стоит."""
    if car.v < 0.4:
        return 'стоит   '
    if car.v < V_CREEP and a_target < -0.3:
        return 'крадётся'
    return 'едет    '

def _binding_str(binding):
    kind, foe, tag = binding
    if kind == 'free':
        return 'путь свободен'
    if kind == 'lead':
        return f'держу дистанцию до #{foe}'
    if kind == 'em':
        return 'аварийный тормоз'
    # kind == 'cnf' — конфликтное ограничение
    if tag == 'yield-pdd':
        return f'уступаю #{foe} (по ПДД)'
    if tag == 'yield-committed':
        return f'жду #{foe} (он уже на перекрёстке)'
    if tag == 'yield-gap-time':
        return f'уступаю #{foe} (он у точки конфликта раньше)'
    if tag == 'yield-gap-uid':
        return f'уступаю #{foe} (моя очередь после)'
    return f'уступаю #{foe} ({tag})'


class IntersectionSimV2:

    def __init__(self):
        self.geom    = RoadGeometry()
        self.cars    = []
        self.next_uid = 1
        self.time    = 0.0

        self.spawn_lanes = ([('N', i) for i in range(LANES)]
                          + [('S', i) for i in range(LANES)])

        self.total_spawned   = 0
        self.total_finished  = 0
        self.collision_count = 0

        self.yield_events  = 0
        self.commit_events = 0
        self.creep_events  = 0
        self._prev_flags   = {}      # uid -> (yield, commit, creep)

        self.LOG_LEVEL = LOG_LEVEL_DEFAULT
        # uid -> (binding, classification) последний раз, когда логировали
        self._prev_decision = {}

        self.active_conflict_traces = {}
        self.total_selected         = 0
        self.stats_forbid           = 0
        self.stats_soft_select      = 0
        self.stats_inside_replan    = 0

    # ── маршруты, манёвры, спавн ──────────────────────────────

    def _allowed_maneuvers(self, origin, lane):
        return dict(LANE_PRIORS[lane_role(origin, lane)])

    def default_exit_lane(self, origin, lane, maneuver):
        if maneuver == 'straight':
            return lane
        return LANES - 1 if origin == 'N' else 0

    def _random_car(self, origin, lane):
        allowed = self._allowed_maneuvers(origin, lane)
        m       = random.choices(list(allowed.keys()),
                                 weights=list(allowed.values()), k=1)[0]
        exit_lane = self.default_exit_lane(origin, lane, m)
        path      = self.geom.build_path(origin, lane, m, exit_lane)

        desired = V_APPROACH * np.random.uniform(0.92, 1.12)
        if m == 'left':  desired *= 0.95
        elif m == 'right': desired *= 0.92

        color = tuple(np.clip([0.22 + 0.58 * random.random() for _ in range(3)], 0, 1))

        car = Car(
            uid=self.next_uid,
            origin=origin, lane=lane,
            maneuver=m, exit_lane=exit_lane,
            path=path, color=color,
            s=0.0,
            v=float(np.random.uniform(4.8, 7.2)),
            desired_v=float(desired),
            risk_aversion     = float(np.random.uniform(RHO_MIN, RHO_MAX)),
            reaction_delay    = float(np.random.uniform(TAU_REACT_MIN, TAU_REACT_MAX)),
            signal_reliability = float(np.random.uniform(SIGMA_MIN, SIGMA_MAX)),
        )
        self.next_uid += 1
        return car

    def _can_spawn(self, origin, lane):
        if len(self.cars) >= MAX_CARS:
            return False
        spawn_p = self.geom.incoming_spawn(origin, lane)
        for c in self.cars:
            if c.finished:
                continue
            if c.origin == origin and c.lane == lane and c.s < 12.0:
                return False
            if np.linalg.norm(c.pos() - spawn_p) < 9.5:
                return False
        return True

    def try_spawn(self):
        for origin, lane in self.spawn_lanes:
            if random.random() < SPAWN_RATE_PER_LANE * DT and self._can_spawn(origin, lane):
                car = self._random_car(origin, lane)
                self.cars.append(car)
                self.total_spawned += 1

    def same_lane_leader(self, car):
        """
        Ближайшая впереди машина по той же входной полосе или
        по тому же пути внутри Ω.
        """
        best, best_gap = None, 1e9
        for other in self.cars:
            if other.uid == car.uid or other.finished:
                continue
            same_stream = (other.origin == car.origin and other.lane == car.lane
                           and not other.cleared_conflict_zone())
            same_path   = (other.path.key == car.path.key)
            if not (same_stream or same_path):
                continue
            if other.s <= car.s:
                continue
            gap = other.s - car.s - 0.5 * (car.length + other.length)
            if gap < best_gap:
                best_gap = gap
                best     = other
        return best, best_gap

    def foes_of(self, car):
        """
        Список потенциально конфликтующих машин для car.
        """
        if (car.s < car.stopline_s - PLAN_DIST) and not car.inside_conflict_zone():
            return []
        out = []
        for other in self.cars:
            if other.uid == car.uid or other.finished or other.collision:
                continue
            if other.cleared_conflict_zone():
                continue
            out.append(other)
        return out

    # ── обновление наблюдаемого сигнала поворота 

    def _update_signals(self):
        for car in self.cars:
            if car.finished:
                continue
            car.signal_on = (car.maneuver in ('left', 'right')
                             and car.s >= car.stopline_s - 18.0)

    # ── главный шаг ───────────────────────────────────────────

    def step(self):
        self.time += DT
        self.try_spawn()
        self._update_signals()

        # ── 1. целевые ускорения ──
        a_targets = {}
        em_active = {}
        bindings  = {}
        for car in self.cars:
            if car.finished:
                continue
            a_t, em, binding = compute_target_accel(self, car)
            a_targets[car.uid] = a_t
            em_active[car.uid] = em
            bindings[car.uid]  = binding
        for car in self.cars:
            if car.finished:
                continue
            a_t = a_targets[car.uid]
            if em_active[car.uid] or a_t <= -A_BRAKE_CMF:
                car.a = float(np.clip(a_t, -A_BRAKE_MAX, A_MAX_POS))
                continue
            tau = max(0.10, car.reaction_delay)
            a_old = car.a
            a_new = a_old + (DT / tau) * (a_t - a_old)
            car.a = float(np.clip(a_new, -A_BRAKE_MAX, A_MAX_POS))
        for car in self.cars:
            if car.finished:
                continue
            v_new = car.v + car.a * DT
            car.v = float(np.clip(v_new, 0.0, max(car.desired_v + 0.5, 12.0)))
            car.s = float(min(car.path.length, car.s + car.v * DT))
        for car in self.cars:
            if car.finished:
                continue
            y, c, cr = derive_indicators(car.a, car)
            prev = self._prev_flags.get(car.uid, (False, False, False))
            if y and not prev[0]:  self.yield_events  += 1
            if c and not prev[1]:  self.commit_events += 1
            if cr and not prev[2]: self.creep_events  += 1
            self._prev_flags[car.uid] = (y, c, cr)

            # ── лог принятия решения 
            if self.LOG_LEVEL >= 1:
                cls     = _classify(a_targets[car.uid], car)
                binding = bindings[car.uid]
                key     = (cls, binding)
                prev_d  = self._prev_decision.get(car.uid)
                if prev_d != key:
                    print(f"[t={self.time:6.2f}] #{car.uid:<3}  "
                          f"{cls}  —  {_binding_str(binding)}")
                    self._prev_decision[car.uid] = key
            car.selected_to_go = c
            car.commit         = c
            car.plan_valid     = c
            car.debug_state    = ('select' if c
                                  else ('yield' if y else ''))
            car.debug_reason   = ('crawl' if cr else '')

            # финиш
            if car.s >= car.path.length - 0.05:
                car.finished = True
                self.total_finished += 1
        self._detect_collisions()

        for uid in list(self._prev_flags.keys()):
            if not any(c.uid == uid and not c.finished for c in self.cars):
                self._prev_flags.pop(uid, None)
                self._prev_decision.pop(uid, None)
        self.cars = [c for c in self.cars if not c.finished]

        self.total_selected      = self.commit_events
        self.stats_soft_select   = self.commit_events
        self.stats_forbid        = self.yield_events
        self.stats_inside_replan = self.creep_events

    def _detect_collisions(self):
        alive = [c for c in self.cars if not c.finished]
        for i in range(len(alive)):
            p1 = alive[i].pos()
            for j in range(i + 1, len(alive)):
                p2 = alive[j].pos()
                d  = float(np.linalg.norm(p2 - p1))
                if d > 0.55 * (alive[i].length + alive[j].length):
                    continue
                if self._obb_overlap(alive[i], alive[j]):
                    if not (alive[i].collision and alive[j].collision):
                        self.collision_count += 1
                    alive[i].collision = True
                    alive[j].collision = True

    @staticmethod
    def _car_corners(car):
        p   = car.pos()
        t   = car.tangent()
        n   = float(np.linalg.norm(t))
        t   = t / n if n > 1e-9 else np.array([0.0, 1.0])
        lat = np.array([-t[1], t[0]])
        hl, hw = 0.5 * car.length, 0.5 * car.width
        return [p - hl*t - hw*lat,
                p - hl*t + hw*lat,
                p + hl*t - hw*lat,
                p + hl*t + hw*lat]

    def _obb_overlap(self, a, b):
        ca, cb = self._car_corners(a), self._car_corners(b)
        axes = []
        for corners in (ca, cb):
            for edge in (corners[2] - corners[0], corners[1] - corners[0]):
                n = float(np.linalg.norm(edge))
                if n > 1e-9:
                    axes.append(edge / n)
        for axis in axes:
            pa = [float(np.dot(pt, axis)) for pt in ca]
            pb = [float(np.dot(pt, axis)) for pt in cb]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
        return True
