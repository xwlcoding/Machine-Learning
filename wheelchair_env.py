import io
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ══════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════
ROOM_W = 10.0
ROOM_H = 10.0
DT = 0.1
MAX_STEPS = 500
MIN_SPAWN_DIST = 5.0

GOAL_RADIUS     = 0.4
COLLISION_DIST  = 0.3
DA_MIN          = 0.8
WALL_MARGIN     = 1.0
WALL_PENALTY    = -1.5

K_ANGULAR        = 1.0
K_PENALTY        = 2.0
SUCCESS_REWARD   = 10.0
COLLISION_PENALTY = -10.0

DISCRETE_ACTIONS = {
    0: -1.50,   # Counterclockwise (fast)
    1: -0.75,   # Counterclockwise (slow)
    2:  0.00,   # Straight
    3:  0.75,   # Clockwise (slow)
    4:  1.50,   # Clockwise (fast)
}
LINEAR_VELOCITY = 0.5

# ══════════════════════════════════════════════════════════
#  Dark theme palette (shared by all render helpers)
# ══════════════════════════════════════════════════════════
_BG       = "#0d1117"
_PANEL    = "#161b22"
_BORDER   = "#30363d"
_MUTED    = "#8b949e"
_REWARD_CMAP = "RdYlGn"
_REWARD_NORM = Normalize(vmin=-6, vmax=12)


# ══════════════════════════════════════════════════════════
#  Helper classes
# ══════════════════════════════════════════════════════════
class Person:
    """A moving pedestrian."""
    def __init__(self, x, y, speed=0.3):
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = random.uniform(0, 2 * math.pi)

    def step(self, dt, room_w, room_h):
        self.x += self.speed * math.cos(self.angle) * dt
        self.y += self.speed * math.sin(self.angle) * dt
        if self.x < 0.5 or self.x > room_w - 0.5:
            self.angle = math.pi - self.angle
        if self.y < 0.5 or self.y > room_h - 0.5:
            self.angle = -self.angle
        self.x = np.clip(self.x, 0.5, room_w - 0.5)
        self.y = np.clip(self.y, 0.5, room_h - 0.5)


class StaticObstacle:
    """A fixed obstacle."""
    def __init__(self, x, y, radius=0.4):
        self.x = x
        self.y = y
        self.radius = radius


# ══════════════════════════════════════════════════════════
#  Main Environment
# ══════════════════════════════════════════════════════════
class WheelchairNavEnv(gym.Env):
    """
    Wheelchair navigation environment with built-in matplotlib rendering.

    render_mode options
    -------------------
    None        — no rendering (fastest, use for training)
    "rgb_array" — render() returns an H×W×3 uint8 numpy array
    "png_bytes" — render() returns raw PNG bytes (for Colab widgets)
    "save_png"  — render() saves a .png file to disk and returns the path
    """
    metadata = {"render_modes": ["human", "rgb_array", "png_bytes", "save_png"]}

    def __init__(
        self,
        action_type  = "discrete",
        n_people     = 3,
        n_obstacles  = 4,
        render_mode  = None,
        max_steps    = MAX_STEPS,
        seed         = None,
    ):
        super().__init__()

        self.action_type  = action_type
        self.n_people     = n_people
        self.n_obstacles  = n_obstacles
        self.render_mode  = render_mode
        self.max_steps    = max_steps
        self._rng         = np.random.default_rng(seed)

        self.agent_x   = 0.0
        self.agent_y   = 0.0
        self.agent_yaw = 0.0
        self.goal_x    = 0.0
        self.goal_y    = 0.0
        self.people:    list[Person]         = []
        self.obstacles: list[StaticObstacle] = []
        self._step_count  = 0
        self._initial_dist = 0.0
        self._diag        = math.hypot(ROOM_W, ROOM_H)

        # Trajectory buffer — filled each episode for the map trail
        self._traj_x: list[float] = []
        self._traj_y: list[float] = []
        self._traj_r: list[float] = []   # reward at each step (for colour)

        # Observation space: 7 normalised floats in [-1, 1]
        self.observation_space = spaces.Box(
            low  = np.array([-1, 0, 0, -1, 0, 0, -1], dtype=np.float32),
            high = np.array([ 1, 1, 1,  1, 1, 1,  1], dtype=np.float32),
        )

        # Action space
        if action_type == "discrete":
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Box(
                low  = np.array([-1.5], dtype=np.float32),
                high = np.array([ 1.5], dtype=np.float32),
            )

    # ──────────────────────────────────────────────────────
    #  Core API
    # ──────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Agent spawn
        self.agent_x   = float(self._rng.uniform(1.0, ROOM_W - 1.0))
        self.agent_y   = float(self._rng.uniform(1.0, ROOM_H - 1.0))
        self.agent_yaw = float(self._rng.uniform(-math.pi, math.pi))

        # Goal spawn (must be >= MIN_SPAWN_DIST away)
        while True:
            gx = float(self._rng.uniform(1.0, ROOM_W - 1.0))
            gy = float(self._rng.uniform(1.0, ROOM_H - 1.0))
            if math.hypot(gx - self.agent_x, gy - self.agent_y) >= MIN_SPAWN_DIST:
                self.goal_x, self.goal_y = gx, gy
                break

        # Static obstacles
        self.obstacles = []
        for _ in range(self.n_obstacles):
            for _ in range(50):
                ox = float(self._rng.uniform(0.8, ROOM_W - 0.8))
                oy = float(self._rng.uniform(0.8, ROOM_H - 0.8))
                if (math.hypot(ox - self.agent_x, oy - self.agent_y) > 1.2 and
                        math.hypot(ox - self.goal_x, oy - self.goal_y) > 1.2):
                    self.obstacles.append(StaticObstacle(ox, oy))
                    break

        # Moving people
        self.people = []
        for _ in range(self.n_people):
            for _ in range(50):
                px = float(self._rng.uniform(1.0, ROOM_W - 1.0))
                py = float(self._rng.uniform(1.0, ROOM_H - 1.0))
                if math.hypot(px - self.agent_x, py - self.agent_y) > 1.5:
                    self.people.append(Person(px, py))
                    break

        self._step_count   = 0
        self._initial_dist = math.hypot(
            self.goal_x - self.agent_x, self.goal_y - self.agent_y
        )

        # Clear trajectory
        self._traj_x = [self.agent_x]
        self._traj_y = [self.agent_y]
        self._traj_r = []

        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.action_type == "discrete":
            angular_vel = DISCRETE_ACTIONS[int(action)]
        else:
            angular_vel = float(np.clip(np.asarray(action).flat[0], -1.5, 1.5))

        self.agent_yaw += angular_vel * DT
        self.agent_yaw  = (self.agent_yaw + math.pi) % (2 * math.pi) - math.pi
        self.agent_x   += LINEAR_VELOCITY * math.cos(self.agent_yaw) * DT
        self.agent_y   += LINEAR_VELOCITY * math.sin(self.agent_yaw) * DT
        self.agent_x    = float(np.clip(self.agent_x, 0.3, ROOM_W - 0.3))
        self.agent_y    = float(np.clip(self.agent_y, 0.3, ROOM_H - 0.3))

        for p in self.people:
            p.step(DT, ROOM_W, ROOM_H)

        self._step_count += 1
        reward, terminated = self._compute_reward()
        truncated = self._step_count >= self.max_steps

        # Append to trajectory
        self._traj_x.append(self.agent_x)
        self._traj_y.append(self.agent_y)
        self._traj_r.append(reward)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    ##
    def render(self):
        """
        Gymnasium render() — behaviour depends on render_mode set at init.

        render_mode="rgb_array"  → returns H×W×3 uint8 numpy array
        render_mode="png_bytes"  → returns raw PNG bytes (Colab widget ready)
        render_mode="save_png"   → saves wheelchair_render.png, returns path
        render_mode=None         → returns None
        """
        if self.render_mode is None:
            return None
        png = self._render_map_png()
        if self.render_mode == "png_bytes":
            return png
        if self.render_mode == "save_png":
            path = "wheelchair_render.png"
            with open(path, "wb") as f:
                f.write(png)
            return path
        if self.render_mode == "rgb_array":
            import PIL.Image
            img = PIL.Image.open(io.BytesIO(png))
            return np.array(img.convert("RGB"))
        return None

    def render_frame(self, obs=None, q_vals=None, action_idx=None,
                     episode=None, step=None):
        """
        Four-panel frame: map + reward breakdown + observation vector + Q-values.
        Returns PNG bytes. Pass q_vals and action_idx from your agent for the
        Q-value panel; if omitted those panels show placeholder text.

        Designed to be called from sarsa_HP.py like:
            png = env.render_frame(obs=obs, q_vals=q_vals, action_idx=action)
        """
        return self._render_four_panel(obs, q_vals, action_idx, episode, step)

    def close(self):
        plt.close("all")
    ##

    # ──────────────────────────────────────────────────────
    #  Reward
    # ──────────────────────────────────────────────────────
    def _compute_reward(self):
        terminated = False
        dc = self._dist_to_goal()
        dg = self._initial_dist

        ratio = dc / dg if dg > 0 else 1.0
        rd    = 2 ** ratio
        R_d   = rd if dc < dg else max(1.0, rd)

        theta   = self._angle_to_goal()
        r_theta = K_ANGULAR * (1 - math.cos(theta))
        R_theta = r_theta if abs(theta) < math.pi / 2 else -r_theta

        hd  = self._count_nearby_people()
        da  = self._min_dist_to_any()
        if hd > 0 and da < DA_MIN:
            P = hd * (-K_PENALTY)
        elif hd == 0 and da < DA_MIN:
            P = -K_PENALTY
        else:
            P = 0.0
        R_a = P

        dw  = self._dist_to_wall()
        R_c = 0.0
        if da < COLLISION_DIST or dw < COLLISION_DIST:
            R_c = COLLISION_PENALTY
            terminated = True

        R_s = 0.0
        if dc < GOAL_RADIUS:
            R_s = SUCCESS_REWARD
            terminated = True

        R_w   = WALL_PENALTY if dw < WALL_MARGIN else 0.0
        R_eff = -0.01
        total = (R_d * R_theta) + R_a + R_c + R_s + R_eff + R_w
        return float(total), terminated

    ##
    def reward_breakdown(self) -> dict:
        """
        Return a dict of every reward component at the current state.
        Useful for the reward-bars panel and for debugging.
        """
        dc    = self._dist_to_goal()
        dg    = self._initial_dist
        ratio = dc / dg if dg > 0 else 1.0
        rd    = 2 ** ratio
        R_d   = rd if dc < dg else max(1.0, rd)
        theta = self._angle_to_goal()
        r_th  = K_ANGULAR * (1 - math.cos(theta))
        R_th  = r_th if abs(theta) < math.pi / 2 else -r_th
        R_p   = R_d * R_th
        hd    = self._count_nearby_people()
        da    = self._min_dist_to_any()
        dw    = self._dist_to_wall()
        if hd > 0 and da < DA_MIN:
            R_a = hd * (-K_PENALTY)
        elif hd == 0 and da < DA_MIN:
            R_a = -K_PENALTY
        else:
            R_a = 0.0
        R_w   = WALL_PENALTY if dw < WALL_MARGIN else 0.0
        R_c   = COLLISION_PENALTY if (da < COLLISION_DIST or dw < COLLISION_DIST) else 0.0
        R_s   = SUCCESS_REWARD if dc < GOAL_RADIUS else 0.0
        R_eff = -0.01
        return {
            "R_d  (dist rate)":  round(R_d,  4),
            "R_th (angular)":    round(R_th, 4),
            "R_prog (Rd×Rth)":   round(R_p,  4),
            "R_a  (proximity)":  round(R_a,  4),
            "R_w  (wall pen.)":  round(R_w,  4),
            "R_c  (collision)":  round(R_c,  4),
            "R_s  (success)":    round(R_s,  4),
            "R_eff (step pen.)": round(R_eff, 4),
            "TOTAL":             round(R_p + R_a + R_w + R_c + R_s + R_eff, 4),
        }
    ##

    # ──────────────────────────────────────────────────────
    #  Observation / info
    # ──────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        theta_goal = self._angle_to_goal()       / math.pi
        dist_goal  = self._dist_to_goal()        / self._diag
        r_min      = self._min_dist_to_obstacle() / self._diag
        theta_obs  = self._angle_to_nearest_obstacle() / math.pi
        hd         = self._count_nearby_people() / max(1, self.n_people)
        collision  = 1.0 if self._min_dist_to_any() < COLLISION_DIST else 0.0
        yaw_norm   = self.agent_yaw / math.pi
        obs = np.array(
            [theta_goal, dist_goal, r_min, theta_obs, hd, collision, yaw_norm],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> dict:
        return {
            "agent_pos":    (round(self.agent_x, 3), round(self.agent_y, 3)),
            "agent_yaw":    round(math.degrees(self.agent_yaw), 2),
            "goal_pos":     (round(self.goal_x, 3), round(self.goal_y, 3)),
            "dist_to_goal": round(self._dist_to_goal(), 3),
            "step":         self._step_count,
        }

    # ──────────────────────────────────────────────────────
    #  Distance / angle helpers
    # ──────────────────────────────────────────────────────
    def _dist_to_goal(self) -> float:
        return math.hypot(self.goal_x - self.agent_x, self.goal_y - self.agent_y)

    def _angle_to_goal(self) -> float:
        raw  = math.atan2(self.goal_y - self.agent_y, self.goal_x - self.agent_x)
        diff = raw - self.agent_yaw
        return (diff + math.pi) % (2 * math.pi) - math.pi

    def _min_dist_to_obstacle(self) -> float:
        if not self.obstacles:
            return self._diag
        return min(
            math.hypot(o.x - self.agent_x, o.y - self.agent_y) - o.radius
            for o in self.obstacles
        )

    def _angle_to_nearest_obstacle(self) -> float:
        if not self.obstacles:
            return 0.0
        nearest = min(self.obstacles,
                      key=lambda o: math.hypot(o.x - self.agent_x, o.y - self.agent_y))
        raw  = math.atan2(nearest.y - self.agent_y, nearest.x - self.agent_x)
        diff = raw - self.agent_yaw
        return (diff + math.pi) % (2 * math.pi) - math.pi

    def _min_dist_to_any(self) -> float:
        dists = []
        for o in self.obstacles:
            dists.append(math.hypot(o.x - self.agent_x, o.y - self.agent_y) - o.radius)
        for p in self.people:
            dists.append(math.hypot(p.x - self.agent_x, p.y - self.agent_y))
        return min(dists) if dists else self._diag

    def _count_nearby_people(self) -> int:
        return sum(
            1 for p in self.people
            if math.hypot(p.x - self.agent_x, p.y - self.agent_y) < DA_MIN
        )

    def _dist_to_wall(self) -> float:
        return min(
            self.agent_x,
            ROOM_W - self.agent_x,
            self.agent_y,
            ROOM_H - self.agent_y,
        )

    ###
    # ──────────────────────────────────────────────────────
    #  Internal rendering helpers
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _fig_to_png(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        buf.seek(0)
        data = buf.read()
        buf.close()
        plt.close(fig)
        return data

    @staticmethod
    def _style_axes(axes):
        """Apply dark theme to a list (or single) of axes."""
        if not hasattr(axes, "__iter__"):
            axes = [axes]
        for ax in axes:
            ax.set_facecolor(_PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(_BORDER)
            ax.tick_params(colors=_MUTED, labelsize=8)

    @staticmethod
    def _reward_colour(r: float):
        return plt.get_cmap(_REWARD_CMAP)(_REWARD_NORM(float(np.clip(r, -6, 12))))

    # ── Panel: top-down map ────────────────────────────────
    def _draw_map(self, ax):
        ax.clear()
        ax.set_facecolor(_BG)
        ax.set_xlim(-0.4, ROOM_W + 0.4)
        ax.set_ylim(-0.4, ROOM_H + 0.4)
        ax.set_aspect("equal")
        ax.tick_params(colors=_MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        # Grid
        for v in range(0, int(ROOM_W) + 1, 2):
            ax.axvline(v, color="#1c1c2e", lw=0.5)
        for h in range(0, int(ROOM_H) + 1, 2):
            ax.axhline(h, color="#1c1c2e", lw=0.5)

        # Room boundary
        ax.add_patch(mpatches.Rectangle(
            (0, 0), ROOM_W, ROOM_H,
            lw=2, edgecolor="#555577", facecolor="none"))

        # Trajectory coloured by reward
        tx, ty, tr = self._traj_x, self._traj_y, self._traj_r
        if len(tx) > 1:
            for i in range(len(tx) - 1):
                r_val = tr[i] if i < len(tr) else 0.0
                ax.plot([tx[i], tx[i+1]], [ty[i], ty[i+1]],
                        color=self._reward_colour(r_val),
                        lw=1.5, alpha=0.75)

        # Static obstacles
        for ob in self.obstacles:
            ax.add_patch(plt.Circle((ob.x, ob.y), ob.radius,
                                    color="#7B4F2E", alpha=0.9, zorder=3))
            ax.text(ob.x, ob.y, "O", ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold", zorder=4)

        # Moving people
        for i, p in enumerate(self.people):
            ax.add_patch(plt.Circle((p.x, p.y), DA_MIN,
                                    color="#FF8C00", alpha=0.07))
            ax.add_patch(plt.Circle((p.x, p.y), DA_MIN, fill=False,
                                    edgecolor="#FFA500", lw=0.8, ls="--", alpha=0.4))
            ax.add_patch(plt.Circle((p.x, p.y), 0.22,
                                    color="#FF6600", zorder=5))
            ax.text(p.x, p.y + 0.42, f"P{i+1}", ha="center",
                    fontsize=7, color="#FFA500", fontweight="bold", zorder=6)

        # Goal
        ax.add_patch(plt.Circle((self.goal_x, self.goal_y), GOAL_RADIUS,
                                color="#00ff99", alpha=0.15))
        ax.add_patch(plt.Circle((self.goal_x, self.goal_y), GOAL_RADIUS,
                                fill=False, edgecolor="#00ff99", lw=1.8))
        ax.plot(self.goal_x, self.goal_y, "*",
                color="#00ff99", ms=15, mec="white", mew=0.5, zorder=6)
        ax.text(self.goal_x, self.goal_y + 0.62, "GOAL", ha="center",
                fontsize=8, color="#00ff99", fontweight="bold", zorder=7)

        # Dashed line agent → goal
        ax.plot([self.agent_x, self.goal_x], [self.agent_y, self.goal_y],
                "--", color="#00ff99", alpha=0.2, lw=0.8)

        # Agent
        da    = self._min_dist_to_any()
        a_col = ("#ff2222" if da < COLLISION_DIST else
                 "#ffaa00" if da < DA_MIN else "#3399ff")
        ax.add_patch(plt.Circle((self.agent_x, self.agent_y), 0.28,
                                color=a_col, zorder=7))
        ax.annotate(
            "",
            xy=(self.agent_x + 0.60 * math.cos(self.agent_yaw),
                self.agent_y + 0.60 * math.sin(self.agent_yaw)),
            xytext=(self.agent_x, self.agent_y),
            arrowprops=dict(arrowstyle="->", color="white", lw=2.0),
            zorder=8,
        )

        # Reward colourbar
        sm = ScalarMappable(cmap=_REWARD_CMAP, norm=_REWARD_NORM)
        sm.set_array([])
        cbar = ax.get_figure().colorbar(
            sm, ax=ax, fraction=0.022, pad=0.01, shrink=0.5)
        cbar.set_label("Step reward", color=_MUTED, fontsize=7)
        cbar.ax.yaxis.set_tick_params(labelsize=6)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_MUTED)

        # Legend
        ax.legend(handles=[
            mpatches.Patch(color="#3399ff", label="Agent"),
            mpatches.Patch(color="#ffaa00", label="Near obj"),
            mpatches.Patch(color="#ff2222", label="Collision"),
            mpatches.Patch(color="#00ff99", label="Goal"),
            mpatches.Patch(color="#FF6600", label="Person"),
            mpatches.Patch(color="#7B4F2E", label="Obstacle"),
        ], loc="upper right", fontsize=6.5, framealpha=0.5,
           facecolor=_BG, labelcolor="white", edgecolor="#444")

        dc  = self._dist_to_goal()
        ang = math.degrees(self._angle_to_goal())
        st  = ("COLLISION" if da < COLLISION_DIST else
               "NEAR" if da < DA_MIN else "CLEAR")
        ax.set_title(f"Dist {dc:.2f}m  Angle {ang:+.0f}°  [{st}]",
                     color="white", fontsize=9, pad=4)
        ax.set_xlabel("x (m)", color=_MUTED, fontsize=8)
        ax.set_ylabel("y (m)", color=_MUTED, fontsize=8)

    # ── Panel: reward breakdown bars ──────────────────────
    def _draw_reward_bars(self, ax):
        ax.clear()
        ax.set_facecolor(_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        bd     = self.reward_breakdown()
        labels = list(bd.keys())
        vals   = list(bd.values())
        cols   = [
            "#3a9de0" if l == "TOTAL" else
            "#2ecc71" if v > 0.05 else
            "#e74c3c" if v < -0.05 else "#555"
            for l, v in zip(labels, vals)
        ]
        yp   = list(range(len(labels)))
        bars = ax.barh(yp, vals, color=cols, edgecolor="#111", lw=0.4)
        ax.set_yticks(yp)
        ax.set_yticklabels(labels, color="white", fontsize=8)
        ax.axvline(0, color="#555", lw=0.8)
        ax.tick_params(axis="x", colors=_MUTED, labelsize=7)
        ax.set_title("Reward Components", color="white", fontsize=9, pad=4)
        ax.invert_yaxis()
        for bar, v in zip(bars, vals):
            ax.text(
                v + (0.05 if v >= 0 else -0.05),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=7.5, color="white",
            )

    # ── Panel: observation vector bars ────────────────────
    def _draw_obs_bars(self, ax, obs):
        ax.clear()
        ax.set_facecolor(_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        labels = [
            "obs[0] angle/goal",
            "obs[1] dist/goal",
            "obs[2] min dist",
            "obs[3] angle/obs",
            "obs[4] people",
            "obs[5] collision",
            "obs[6] yaw",
        ]
        cols = [
            "#e74c3c" if i == 5 and v > 0.5 else
            "#2ecc71" if i == 5 else
            "#e67e22" if abs(v) > 0.7 else "#7f8c8d"
            for i, v in enumerate(obs)
        ]
        yp = list(range(len(obs)))
        ax.barh(yp, obs, color=cols, edgecolor="#111", lw=0.4)
        ax.set_yticks(yp)
        ax.set_yticklabels(labels, color="white", fontsize=7.5)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_xlim(-1.15, 1.15)
        ax.tick_params(axis="x", colors=_MUTED, labelsize=7)
        ax.set_title("Observation Vector", color="white", fontsize=9, pad=4)
        ax.invert_yaxis()
        for i, v in enumerate(obs):
            ax.text(
                v + (0.04 if v >= 0 else -0.04), i,
                f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=7, color="white",
            )

    # ── Panel: Q-value bars ───────────────────────────────
    @staticmethod
    def _draw_q_bars(ax, q_vals, action_idx):
        ax.clear()
        ax.set_facecolor(_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        if q_vals is None:
            ax.text(0.5, 0.5, "Q-values not provided",
                    ha="center", va="center", color=_MUTED,
                    transform=ax.transAxes, fontsize=9)
            ax.set_title("Q-values per Action", color="white", fontsize=9, pad=4)
            return

        best  = int(np.argmax(q_vals)) if action_idx is None else action_idx
        cols  = ["#f0883e" if i == best else "#58a6ff"
                 for i in range(len(q_vals))]
        xlabs = [f"A{i}: {DISCRETE_ACTIONS[i]:+.2f}" for i in range(len(q_vals))]
        bars  = ax.bar(range(len(q_vals)), q_vals,
                       color=cols, edgecolor="#111", lw=0.4)
        ax.set_xticks(range(len(q_vals)))
        ax.set_xticklabels(xlabs, color="white", fontsize=7.5, rotation=10)
        ax.axhline(0, color="#555", lw=0.8)
        ax.tick_params(axis="y", colors=_MUTED, labelsize=7)
        ax.set_title("Q-values (orange = chosen)", color="white", fontsize=9, pad=4)
        for bar, v in zip(bars, q_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + (0.03 if v >= 0 else -0.03),
                f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7, color="white",
            )

    # ── Four-panel episode frame ───────────────────────────
    def _render_four_panel(self, obs=None, q_vals=None,
                           action_idx=None, episode=None, step=None) -> bytes:
        fig = plt.figure(figsize=(16, 6), facecolor=_BG)
        gs  = gridspec.GridSpec(
            1, 4, figure=fig,
            left=0.05, right=0.97, top=0.88, bottom=0.10, wspace=0.42)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

        ep_str  = f"Ep {episode}" if episode is not None else ""
        st_str  = f"  Step {step}" if step is not None else ""
        act_str = ""
        if action_idx is not None:
            act_str = f"  |  Action {action_idx} ({DISCRETE_ACTIONS[action_idx]:+.2f} rad/s)"
        q_str = ""
        if q_vals is not None:
            q_str = f"  |  Best Q {max(q_vals):+.2f}"

        fig.suptitle(
            f"Agent  {ep_str}{st_str}{act_str}{q_str}",
            color="white", fontsize=10, fontweight="bold")

        self._draw_map(axes[0])
        self._draw_reward_bars(axes[1])

        if obs is not None:
            self._draw_obs_bars(axes[2], obs)
        else:
            axes[2].set_facecolor(_BG)
            axes[2].text(0.5, 0.5, "obs not provided",
                         ha="center", va="center", color=_MUTED,
                         transform=axes[2].transAxes, fontsize=9)

        self._draw_q_bars(axes[3], q_vals, action_idx)

        return self._fig_to_png(fig)

    # ── Simple map-only PNG ────────────────────────────────
    def _render_map_png(self) -> bytes:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=_BG)
        self._draw_map(ax)
        return self._fig_to_png(fig)
