

import numpy as np

from .config import LANES, LANE_PRIORS, D_VIS

def lane_role(origin, lane):
    if origin == 'N':
        if lane == 0:           return 'right'
        if lane == LANES - 1:   return 'left'
    else:  # 'S'
        if lane == 0:           return 'left'
        if lane == LANES - 1:   return 'right'
    return 'middle'


def lane_prior(origin, lane):
    """Априорное распределение манёвра по полосе."""
    return dict(LANE_PRIORS[lane_role(origin, lane)])

def hard_yield_rule(origin_i, maneuver_i, lane_i,
                    origin_j, maneuver_j, lane_j):
    # R1
    if {origin_i, origin_j} == {'N', 'S'} and origin_i != origin_j:
        if maneuver_i == 'left' and maneuver_j in ('straight', 'right'):
            return True
    # R2
    if origin_i == origin_j and lane_i != lane_j:
        if maneuver_i in ('left', 'right') and maneuver_j == 'straight':
            return True
    return False


def hard_yield(car_i, car_j):
    return hard_yield_rule(
        car_i.origin, car_i.maneuver, car_i.lane,
        car_j.origin, car_j.maneuver, car_j.lane,
    )

def is_visible(observer, other):
    d = float(np.linalg.norm(observer.pos() - other.pos()))
    return d <= D_VIS


def posterior_maneuver(observer, other):
    prior = lane_prior(other.origin, other.lane)

    if not is_visible(observer, other):
        return prior

    sigma = float(np.clip(other.signal_reliability, 0.05, 0.99))

    if other.signal_on:
        likelihood = {
            'left':     sigma,
            'right':    sigma,
            'straight': 1.0 - sigma,
        }
    else:
        likelihood = {
            'left':     1.0 - sigma,
            'right':    1.0 - sigma,
            'straight': sigma,
        }

    unnorm = {m: prior[m] * likelihood[m] for m in prior}
    z = sum(unnorm.values())
    if z <= 1e-12:
        return prior
    return {m: unnorm[m] / z for m in unnorm}

def pi_yield(geom, car_i, car_j, default_exit_lane_fn):
    q_j = posterior_maneuver(car_i, car_j)
    s = 0.0
    for m, q in q_j.items():
        if q <= 0.0:
            continue
        # проверяем физическое пересечение путей с гипотетическим манёвром j
        exit_lane = default_exit_lane_fn(car_j.origin, car_j.lane, m)
        path_j_hyp = geom.build_path(car_j.origin, car_j.lane, m, exit_lane)
        if not geom.shared_cells(car_i.path.key, path_j_hyp.key):
            continue
        if hard_yield_rule(car_i.origin, car_i.maneuver, car_i.lane,
                           car_j.origin, m, car_j.lane):
            s += q
    return float(np.clip(s, 0.0, 1.0))
