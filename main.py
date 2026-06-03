
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Slider

from model_v2.config import (
    CAR_LENGTH, CAR_WIDTH,
    FPS_MS, INTERSECTION_HALF,
    LANE_W, LANES,
    RENDER_VIEW, ROAD_HALF,
    SIM_STEPS, VIEW, DT,
)
from model_v2.simulation import IntersectionSimV2


def run_headless(seconds=120.0, seed=None):
    import random as _rnd
    import numpy as _np
    if seed is not None:
        _rnd.seed(seed); _np.random.seed(seed)
    sim = IntersectionSimV2()
    sim.LOG_LEVEL = 0
    steps = int(seconds / DT)
    for _ in range(steps):
        sim.step()
    print(f"t={sim.time:6.1f}s  spawned={sim.total_spawned:4d}  "
          f"finished={sim.total_finished:4d}  alive={len(sim.cars):3d}  "
          f"coll={sim.collision_count:3d}  "
          f"yield_ev={sim.yield_events:4d}  commit_ev={sim.commit_events:4d}  "
          f"creep_ev={sim.creep_events:4d}")
    return sim

# ── визуализация ──────────────────────────────────────────────

class Renderer:
    def __init__(self, sim):
        self.sim  = sim
        self.view = RENDER_VIEW

        self.fig, self.ax = plt.subplots(figsize=(10.8, 10.8))
        plt.subplots_adjust(bottom=0.08)

        self.ax.set_aspect("equal")
        self.ax.set_xlim(-self.view, self.view)
        self.ax.set_ylim(-self.view, self.view)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("#dbe5e8")

        self.paused = False
        self.speed  = 1.0
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.car_patches  = {}
        self.labels       = {}
        self.debug_labels = {}

        self.draw_static_scene()
        self.create_hud()
        self.create_slider()

    def create_hud(self):
        self.stats = self.ax.text(
            -self.view + 2.0, self.view - 2.0, "",
            ha="left", va="top", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.9),
            zorder=20,
        )
        self.side = self.ax.text(
            self.view - 2.0, self.view - 2.0, "",
            ha="right", va="top", fontsize=8.8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="black", alpha=0.88),
            zorder=20,
        )

    def create_slider(self):
        ax_slider = self.fig.add_axes([0.20, 0.01, 0.60, 0.04])
        self.slider = Slider(ax=ax_slider, label="Speed",
                             valmin=0.1, valmax=5.0,
                             valinit=1.0, valstep=0.1)
        self.slider.on_changed(self.on_speed_change)

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key in ("0", "1", "2"):
            self.sim.LOG_LEVEL = int(event.key)

    def on_speed_change(self, val):
        self.speed = val

    def draw_static_scene(self):
        asphalt   = "#53565c"
        lane_mark = "#f6efbf"

        self.ax.add_patch(Rectangle((-ROAD_HALF, -VIEW), 2*ROAD_HALF, 2*VIEW,
                                    fc=asphalt, ec="none", zorder=0))
        self.ax.add_patch(Rectangle((-VIEW, -ROAD_HALF), 2*VIEW, 2*ROAD_HALF,
                                    fc=asphalt, ec="none", zorder=0))

        xs = [-ROAD_HALF + i * LANE_W for i in range(2 * LANES + 1)]
        for x in xs:
            major = abs(x) < 1e-9 or abs(abs(x) - ROAD_HALF) < 1e-9
            self.ax.plot([x, x], [-VIEW, VIEW],
                         color=lane_mark,
                         lw=2.3 if major else 1.0,
                         linestyle="-" if major else (0, (7, 7)),
                         zorder=2)
        ys = [-ROAD_HALF + i * LANE_W for i in range(2 * LANES + 1)]
        for y in ys:
            major = abs(y) < 1e-9 or abs(abs(y) - ROAD_HALF) < 1e-9
            self.ax.plot([-VIEW, VIEW], [y, y],
                         color=lane_mark,
                         lw=2.3 if major else 1.0,
                         linestyle="-" if major else (0, (7, 7)),
                         zorder=2)
        for sign in (-1, 1):
            self.ax.plot([-ROAD_HALF, ROAD_HALF],
                         [sign*INTERSECTION_HALF, sign*INTERSECTION_HALF],
                         color="white", lw=2.8, zorder=3)
        self.ax.add_patch(Rectangle(
            (-INTERSECTION_HALF, -INTERSECTION_HALF),
            2*INTERSECTION_HALF, 2*INTERSECTION_HALF,
            fill=False, ec="white", lw=1.1, linestyle="--", zorder=3,
        ))

    def ensure_artist(self, car):
        if car.uid in self.car_patches:
            return self.car_patches[car.uid]
        body = Rectangle((-CAR_LENGTH/2.0, -CAR_WIDTH/2.0),
                         CAR_LENGTH, CAR_WIDTH, fc=car.color,
                         ec="black", lw=0.9, zorder=8)
        nose = Rectangle((CAR_LENGTH/2.0 - 0.55, -CAR_WIDTH/2.0),
                         0.55, CAR_WIDTH, fc="white", ec="none",
                         alpha=0.75, zorder=9)
        self.ax.add_patch(body)
        self.ax.add_patch(nose)

        label = self.ax.text(0.0, 0.0, str(car.uid),
                             ha="center", va="center", fontsize=8.5,
                             color="black",
                             bbox=dict(boxstyle="circle,pad=0.18",
                                       fc="white", ec="black", alpha=0.95),
                             zorder=12)
        dbg = self.ax.text(0.0, 0.0, "",
                           ha="center", va="bottom", fontsize=7.0,
                           color="black",
                           bbox=dict(boxstyle="round,pad=0.12",
                                     fc="#ffffe0", ec="black", alpha=0.45),
                           zorder=13)
        dbg.set_visible(False)
        self.car_patches[car.uid]  = (body, nose)
        self.labels[car.uid]       = label
        self.debug_labels[car.uid] = dbg
        return body, nose

    def remove_gone(self):
        alive = {c.uid for c in self.sim.cars if not c.finished}
        for uid in list(self.car_patches):
            if uid in alive:
                continue
            body, nose = self.car_patches.pop(uid)
            body.remove(); nose.remove()
            self.labels.pop(uid).remove()
            self.debug_labels.pop(uid).remove()

    def update(self, _frame):
        if self.paused:
            return []
        for _ in range(int(max(1, self.speed))):
            self.sim.step()
        self.remove_gone()

        committed = 0
        inside    = 0
        waiting   = 0
        for car in self.sim.cars:
            body, nose = self.ensure_artist(car)
            p = car.pos()
            angle = math.radians(car.angle_deg())
            tr = Affine2D().rotate(angle).translate(p[0], p[1]) + self.ax.transData
            body.set_transform(tr); nose.set_transform(tr)

            if car.collision:
                body.set_edgecolor("red"); body.set_linewidth(2.2)
            elif car.commit:
                body.set_edgecolor("lime"); body.set_linewidth(2.1)
            elif car.v < 0.30 and car.s < car.stopline_s + 1.1:
                body.set_edgecolor("orange"); body.set_linewidth(1.8)
            else:
                body.set_edgecolor("black"); body.set_linewidth(0.9)

            self.labels[car.uid].set_position((p[0], p[1]))

            dbg = self.debug_labels[car.uid]
            if car.debug_state:
                state_map = {"select": "edu", "yield": "wait"}
                dbg.set_visible(True)
                dbg.set_position((p[0], p[1] + 3.1))
                dbg.set_text(f"{state_map.get(car.debug_state, car.debug_state)}")
            else:
                dbg.set_visible(False)

            committed += int(car.commit)
            inside    += int(car.inside_conflict_zone())
            waiting   += int(car.v < 0.35 and car.s < car.stopline_s + 2.0)

        self.stats.set_text(
            f"t = {self.sim.time:7.2f} s\n"
            f"active       = {len(self.sim.cars):3d}\n"
            f"spawned      = {self.sim.total_spawned:3d}\n"
            f"finished     = {self.sim.total_finished:3d}\n"
            f"committed    = {committed:3d}\n"
            f"inside       = {inside:3d}\n"
            f"waiting      = {waiting:3d}\n"
            f"collisions   = {self.sim.collision_count:3d}\n"
            f"space: pause"
        )
        self.side.set_text(
            "события (на переходах, не на тик)\n"
            f"yield events  = {self.sim.yield_events}\n"
            f"commit events = {self.sim.commit_events}\n"
            f"creep events  = {self.sim.creep_events}\n"
            f"speed x       = {self.speed:.1f}"
        )
        self.ax.set_title("Перекрёсток без светофора — модель v2", fontsize=11)
        return []


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
        run_headless(seconds)
        return
    sim      = IntersectionSimV2()
    renderer = Renderer(sim)
    anim = animation.FuncAnimation(
        renderer.fig, renderer.update,
        frames=SIM_STEPS, interval=FPS_MS,
        blit=False, cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
