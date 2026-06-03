
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import (
    A_MAX_POS, A_BRAKE_CMF, A_BRAKE_MAX,
    V_MIN, V_MAX,
    IDM_T, IDM_G_MIN,
    T0_SAFE, BETA_SAFE, GAMMA_SAFE,
    T_GAP0, KAPPA1_GAP, KAPPA2_GAP,
    T_HORIZON,
    D_EM_FWD, D_EM_LAT, D_EM_FULL,
    EPS_DIST,
    CAR_WIDTH,
)
from .priority import (
    posterior_maneuver, pi_yield,
)

INF = math.inf


def v_ref_self(car):
    return V_MAX[car.maneuver]


def v_ref_ext(car):
    return max(V_MIN, min(V_MAX[car.maneuver], car.v))


def time_to_reach_self(car, s_target):
    if s_target <= car.s:
        return 0.0
    return (s_target - car.s) / v_ref_self(car)


def time_to_reach_ext(car, s_target):
    if s_target <= car.s:
        return 0.0
    return (s_target - car.s) / v_ref_ext(car)


# ── конфликтная область пары (i, j) ──────────────────────────

@dataclass
class ConflictRegion:
    s_i_in:  float
    s_i_out: float
    s_j_in:  float
    s_j_out: float


def conflict_region(geom, car_i, car_j) -> Optional[ConflictRegion]:
    if car_i.uid == car_j.uid:
        return None
    if car_i.path.key == car_j.path.key:
        return None
    shared = geom.shared_cells(car_i.path.key, car_j.path.key)
    if not shared:
        return None

    s_i = car_i.s - 0.25 * car_i.length
    s_j = car_j.s - 0.25 * car_j.length
    ahead = [(s1, s2) for (_, s1, s2) in shared if s1 >= s_i and s2 >= s_j]
    if not ahead:
        return None

    s_i_in  = min(a[0] for a in ahead)
    s_i_out = max(a[0] for a in ahead)
    s_j_in  = min(a[1] for a in ahead)
    s_j_out = max(a[1] for a in ahead)
    return ConflictRegion(s_i_in, s_i_out, s_j_in, s_j_out)


# желаемая скорость

def alpha_free(car):
    v_des = min(V_MAX[car.maneuver], car.desired_v)
    dv = v_des - car.v
    if dv >= 0.0:
        return min(A_MAX_POS, 0.95 * dv + 0.6)
    return max(-A_BRAKE_CMF, 0.85 * dv)


# (IDM)

def alpha_lead(car, leader, gap):
    if leader is None:
        return INF
    v       = max(0.0, car.v)
    v_des   = min(V_MAX[car.maneuver], car.desired_v)
    dv      = v - leader.v
    g_star  = IDM_G_MIN + max(
        0.0, v * IDM_T + v * dv / (2.0 * math.sqrt(A_MAX_POS * A_BRAKE_CMF))
    )
    g       = max(EPS_DIST, gap)
    a       = A_MAX_POS * (1.0 - (v / max(1e-3, v_des)) ** 4 - (g_star / g) ** 2)
    return float(np.clip(a, -A_BRAKE_MAX, A_MAX_POS))


def t_safe(car_i, q_j_hat):
    rho = float(np.clip(car_i.risk_aversion, 0.0, 1.0))
    return T0_SAFE + BETA_SAFE * (1.0 + rho) + GAMMA_SAFE * (1.0 - q_j_hat)


def t_gap(car_i):
    rho = float(np.clip(car_i.risk_aversion, 0.0, 1.0))
    boost = 1.0 + KAPPA2_GAP if car_i.maneuver == 'left' else 1.0
    return T_GAP0 * (1.0 + KAPPA1_GAP * rho) * boost

def alpha_brake_to(car, s_stop):
    if car.s >= s_stop:
        return -A_BRAKE_MAX
    dist = s_stop - car.s
    v    = max(0.0, car.v)
    return -(v * v) / (2.0 * max(dist, EPS_DIST))

def alpha_conflict(geom, car_i, car_j, region, default_exit_lane_fn):

    tau_i_in  = time_to_reach_self(car_i, region.s_i_in)
    tau_j_in  = time_to_reach_ext(car_j,  region.s_j_in)
    tau_i_out = tau_i_in + (region.s_i_out - region.s_i_in + car_i.length) / v_ref_self(car_i)
    tau_j_out = tau_j_in + (region.s_j_out - region.s_j_in + car_j.length) / v_ref_ext(car_j)

    if car_j.s < car_j.stopline_s and tau_j_in > T_HORIZON:
        return INF, 'beyond-horizon'

    q_j     = posterior_maneuver(car_i, car_j)
    q_j_hat = max(q_j.values()) if q_j else 1.0

    pi_ij   = pi_yield(geom, car_i, car_j, default_exit_lane_fn)
    pi_ji   = pi_yield(geom, car_j, car_i, default_exit_lane_fn)

    Tsafe   = t_safe(car_i, q_j_hat)
    if car_j.s >= car_j.stopline_s:
        role, tag = 'yield', 'yield-committed'
    elif pi_ij >= 0.5:
        role, tag = 'yield', 'yield-pdd'
    elif pi_ji >= 0.5:
        role, tag = 'first', 'first'
    else:
        Tgap = t_gap(car_i)
        delta = tau_j_in - tau_i_in
        if delta >= Tgap:
            role, tag = 'first', 'first'
        elif delta <= -Tgap:
            role, tag = 'yield', 'yield-gap-time'
        else:
            if car_i.uid < car_j.uid:
                role, tag = 'first', 'first'
            else:
                role, tag = 'yield', 'yield-gap-uid'

    if role == 'first':
        return INF, 'first'

    if tau_j_out + Tsafe <= tau_i_in:
        return INF, 'pass-clear'

    s_stop_geom = region.s_i_in - 0.5 * car_i.length - 0.5 * CAR_WIDTH - IDM_G_MIN
    s_stop_line = car_i.stopline_s - IDM_G_MIN
    s_stop = min(s_stop_geom, s_stop_line)
    return alpha_brake_to(car_i, s_stop), tag

def alpha_emergency(car, neighbours):
    p = car.pos()
    t = car.tangent()
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        return INF
    t   = t / n
    lat = np.array([-t[1], t[0]], dtype=float)

    for other in neighbours:
        if other.uid == car.uid or other.finished:
            continue
        dvec = other.pos() - p
        d    = float(np.linalg.norm(dvec))
        if d >= D_EM_FULL:
            continue
        fwd      = float(np.dot(dvec, t))
        lat_dist = abs(float(np.dot(dvec, lat)))
        if fwd > -0.5 * car.length and lat_dist < D_EM_LAT * car.width:
            return -A_BRAKE_MAX
        if d < D_EM_FWD:
            return -A_BRAKE_MAX
    return INF


def is_committed(car):
    """§6.1 .tex — машина пересекла стоп-линию."""
    return car.s >= car.stopline_s


def compute_target_accel(sim, car):
    if car.finished:
        return 0.0, False, ('free', None, None)

    a = alpha_free(car)
    binding = ('free', None, None)

    leader, gap = sim.same_lane_leader(car)
    a_l = alpha_lead(car, leader, gap)
    if a_l < a:
        a, binding = a_l, ('lead', leader.uid, None)
    if not is_committed(car):
        for other in sim.foes_of(car):
            region = conflict_region(sim.geom, car, other)
            if region is None:
                continue
            a_c, tag = alpha_conflict(sim.geom, car, other, region,
                                      sim.default_exit_lane)
            if a_c < a:
                a, binding = a_c, ('cnf', other.uid, tag)

    a_em = alpha_emergency(car, sim.cars)
    em_active = math.isfinite(a_em)
    if a_em < a:
        a, binding = a_em, ('em', None, None)

    return float(np.clip(a, -A_BRAKE_MAX, A_MAX_POS)), em_active, binding

def derive_indicators(a_star, car):
    yielding  = a_star <= -0.20 and car.s < car.stopline_s + 0.5
    committed = (car.s >= car.stopline_s) and (a_star > -0.5 * A_BRAKE_CMF)
    creeping  = (0.0 < car.v < 2.2) and yielding
    return yielding, committed, creeping
