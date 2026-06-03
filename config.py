"""
model_v2/config.py — параметры матмодели v2.

Все символы соответствуют разделам .tex-документа.
Геометрия и габариты переиспользуются из model.config — они физические,
от модели решения не зависят.
"""

import random
import numpy as np

from model.config import (
    LANE_W, LANES, ROAD_HALF, INTERSECTION_HALF,
    SPAWN_DIST, EXIT_DIST, VIEW, RENDER_VIEW, CELL_SIZE,
    CAR_LENGTH, CAR_WIDTH,
    DT, FPS_MS, SIM_STEPS,
    MAX_CARS, SPAWN_RATE_PER_LANE,
)

# ── воспроизводимость ──
SEED = 2
random.seed(SEED)
np.random.seed(SEED)

# ── ускорения / торможения, §4 .tex ──
A_MAX_POS  = 1.8     # a_max^+    — комфортное ускорение
A_BRAKE_CMF = 2.8    # a_cmf^-    — комфортное торможение
A_BRAKE_MAX = 6.4    # a_max^-    — предельное торможение

# ── желаемые/предельные скорости по манёвру ──
V_MIN       = 0.5
V_STRAIGHT  = 9.0
V_LEFT      = 7.0
V_RIGHT     = 6.8
V_APPROACH  = 9.2
V_CREEP     = 2.1

V_MAX = {
    'straight': V_STRAIGHT,
    'left':     V_LEFT,
    'right':    V_RIGHT,
}

IDM_T      = 1.10
IDM_G_MIN  = 2.2

T0_SAFE    = 0.7    # T_0
BETA_SAFE  = 0.4    # β
GAMMA_SAFE = 0.5    # γ

T_GAP0     = T0_SAFE + BETA_SAFE + GAMMA_SAFE + 0.1   
KAPPA1_GAP = 0.30
KAPPA2_GAP = 0.45

T_HORIZON  = 5.0

D_VIS      = 18.0   # дистанция, на которой видим сигнал поворота
D_EM_FWD   = 3.0    # продольный радиус аварийного торможения
D_EM_LAT   = 1.10   
D_EM_FULL  = 5.8    

STOP_BUFFER  = 1.0   # запас перед стоп-линией
PLAN_DIST    = 10.0  
EPS_DIST     = 0.05  

TAU_REACT_MIN = 0.45
TAU_REACT_MAX = 1.00

RHO_MIN = 0.20
RHO_MAX = 1.00

SIGMA_MIN = 0.55
SIGMA_MAX = 0.95

LANE_PRIORS = {
    'left':   {'left': 0.68, 'straight': 0.28, 'right': 0.04},
    'right':  {'right': 0.54, 'straight': 0.40, 'left':  0.06},
    'middle': {'straight': 0.70, 'left': 0.17, 'right': 0.13},
}

LOG_LEVEL_DEFAULT = 1
