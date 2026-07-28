import os
import io
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

REWARD_THRESHOLD  = 8.0
SUCCESS_THRESHOLD = 0.8

from wheelchair_env import (
    WheelchairNavEnv, GOAL_RADIUS, COLLISION_DIST, DA_MIN,
    ROOM_W, ROOM_H, K_ANGULAR, K_PENALTY,
    SUCCESS_REWARD, COLLISION_PENALTY,
    WALL_MARGIN, WALL_PENALTY, DISCRETE_ACTIONS,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import ipywidgets as widgets
    from IPython.display import display, Image as IPImage
    warnings.filterwarnings("ignore", category=UserWarning)
    COLAB = True
except ImportError:
    COLAB = False

OUT_DIR = "sarsa_results_εgreedy_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  1. Hyperparameters
# ══════════════════════════════════════════════════════════
class HP:
    STATE_DIM   = 7
    N_ACTIONS   = 5
    HIDDEN_SIZE = 256

    BUFFER_SIZE = 50000
    BATCH_SIZE  = 96

    # Paper 2 (Xi & Shino, 2020): α=0.05, γ=0.8
    LR          = 3e-4
    GAMMA       = 0.99

    TRAIN_EPISODES = 2000
    EVAL_EPISODES  = 50
    CONV_WINDOW    = 50

    # Add ε-greedy hyperparameters
    EPS_START = 0.8
    EPS_END = 0.05
    EPS_DECAY = 0.995


# ══════════════════════════════════════════════════════════
#  2. Replay Buffer
# ══════════════════════════════════════════════════════════
class ReplayBuffer:

    def __init__(self, state_dim, max_size=HP.BUFFER_SIZE):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0
        self.states       = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions      = np.zeros(max_size, dtype=np.int64)
        self.rewards      = np.zeros(max_size, dtype=np.float32)
        self.next_states  = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_actions = np.zeros(max_size, dtype=np.int64)
        self.dones        = np.zeros(max_size, dtype=np.float32)

    def add(self, state, action, reward, next_state, next_action, done):
        self.states[self.ptr]       = state
        self.actions[self.ptr]      = action
        self.rewards[self.ptr]      = reward
        self.next_states[self.ptr]  = next_state
        self.next_actions[self.ptr] = next_action
        self.dones[self.ptr]        = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(device),
            torch.LongTensor (self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_states[idx]).to(device),
            torch.LongTensor (self.next_actions[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════
#  3. SARSA Agent
# ══════════════════════════════════════════════════════════
class SARSAAgent:

    def __init__(self, state_dim=7, n_actions=5, hidden=256,
                 learning_rate=HP.LR, gamma=HP.GAMMA):
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_actions  = n_actions
        self.action_dim = n_actions
        self.gamma      = gamma

        # Add ε variables
        self.epsilon = HP.EPS_START
        self.eps_min = HP.EPS_END
        self.eps_decay = HP.EPS_DECAY

        # Single Q-network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_actions),
        ).to(self.device)
        self.q_opt = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.total_steps  = 0
        self.update_count = 0
        self.q_losses     = []

    # Add ε-greedy version
    def select_action_epsilon(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)  # explore
        else:
            q_vals = self.q_values(state)
            return int(np.argmax(q_vals))             # exploit

    def q_values(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(s).cpu().numpy().flatten()

    # Replace τ decay
    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def update(self, replay_buffer):
        """
        SARSA TD update (Paper 2, eq.8):
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]
        """
        if len(replay_buffer) < HP.BATCH_SIZE:
            return None

        self.update_count += 1
        s, a, r, s2, a2, d = replay_buffer.sample(HP.BATCH_SIZE, self.device)

        q_current = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next   = self.q_net(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
            q_target = r + (1.0 - d) * self.gamma * q_next

        loss = F.mse_loss(q_current, q_target)
        self.q_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 3.0)
        self.q_opt.step()

        q_loss = loss.item()
        self.q_losses.append(q_loss)
        return q_loss

    def save(self, path="sarsa_wheelchair_v1.pt"):
        torch.save({
            "q_net":        self.q_net.state_dict(),
            "epsilon":      self.epsilon,
            "total_steps":  self.total_steps,
            "update_count": self.update_count,
        }, path)
        print(f"  Saved → {path}")

    def load(self, path="sarsa_wheelchair_v1.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.epsilon = ckpt.get("epsilon",          self.eps_min)
        self.total_steps  = ckpt.get("total_steps",  0)
        self.update_count = ckpt.get("update_count", 0)
        print(f"  Loaded ← {path}")


# ══════════════════════════════════════════════════════════
#  4. Metrics Logger
# ══════════════════════════════════════════════════════════
class MetricsLogger:

    def __init__(self, window=HP.CONV_WINDOW):
        self.window = window
        self.ep_rewards      = []
        self.ep_lengths      = []
        self.ep_successes    = []
        self.ep_losses       = []
        self.ep_taus         = []
        self.rolling_reward  = []
        self.rolling_success = []
        self.convergence_ep  = None
        self.total_steps_log = []

    def log(self, reward, length, success, mean_loss, tau, total_steps):
        self.ep_rewards.append(reward)
        self.ep_lengths.append(length)
        self.ep_successes.append(int(success))
        self.ep_losses.append(mean_loss if mean_loss is not None else float("nan"))
        self.ep_taus.append(tau)
        self.total_steps_log.append(total_steps)

        w = self.window
        n = len(self.ep_rewards)
        r_mean = float(np.mean(self.ep_rewards[max(0, n - w):]))
        s_mean = float(np.mean(self.ep_successes[max(0, n - w):]))
        self.rolling_reward.append(r_mean)
        self.rolling_success.append(s_mean)

        if self.convergence_ep is None:
            if r_mean >= REWARD_THRESHOLD and s_mean >= SUCCESS_THRESHOLD:
                self.convergence_ep = n

    def convergence_rate(self):
        w = self.window
        recent = self.ep_rewards[-w:]
        return float(np.mean([1 if r >= REWARD_THRESHOLD else 0 for r in recent]))

    def sample_efficiency(self):
        if self.convergence_ep is None:
            return None
        return self.total_steps_log[self.convergence_ep - 1]

    def success_rate(self):
        return self.rolling_success[-1] if self.rolling_success else 0.0

    def average_reward(self):
        return self.rolling_reward[-1] if self.rolling_reward else 0.0

    def total_reward(self):
        return float(np.sum(self.ep_rewards))


# ══════════════════════════════════════════════════════════
#  5. Plot Helpers
# ══════════════════════════════════════════════════════════
_BG      = "#0d1117"
_PANEL   = "#161b22"
_GRID    = "#21262d"
_ACCENT1 = "#58a6ff"
_ACCENT2 = "#3fb950"
_ACCENT3 = "#f0883e"
_ACCENT4 = "#ff7b72"
_ACCENT5 = "#a371f7"
_TEXT    = "#c9d1d9"
_MUTED   = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor":   _PANEL,
    "axes.edgecolor":   _GRID,
    "axes.labelcolor":  _TEXT,
    "xtick.color":      _MUTED,
    "ytick.color":      _MUTED,
    "text.color":       _TEXT,
    "grid.color":       _GRID,
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": _PANEL,
    "legend.edgecolor": _GRID,
    "legend.labelcolor": _TEXT,
    "font.family":      "monospace",
})


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved graph → {path}")
    return path


def smooth(arr, w=20):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_total_reward(logger):
    fig, ax = plt.subplots(figsize=(10, 4))
    eps = np.arange(1, len(logger.ep_rewards) + 1)
    ax.plot(eps, logger.ep_rewards, color=_ACCENT1, alpha=0.25, lw=0.8, label="Episode reward")
    sm = smooth(logger.ep_rewards, 30)
    ax.plot(np.arange(30, len(logger.ep_rewards) + 1), sm,
            color=_ACCENT1, lw=2, label="Smoothed (w=30)")
    ax.axhline(REWARD_THRESHOLD, color=_ACCENT2, ls="--", lw=1.2, label=f"Target ({REWARD_THRESHOLD})")
    if logger.convergence_ep:
        ax.axvline(logger.convergence_ep, color=_ACCENT3, ls=":", lw=1.5,
                   label=f"Converged @ ep {logger.convergence_ep}")
    ax.set_title("Total Reward per Episode", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.legend(fontsize=8); ax.grid(True)
    _save(fig, "01_total_reward.png")


def plot_average_reward(logger):
    fig, ax = plt.subplots(figsize=(10, 4))
    eps = np.arange(1, len(logger.rolling_reward) + 1)
    ax.plot(eps, logger.rolling_reward, color=_ACCENT2, lw=2,
            label=f"Rolling mean (w={logger.window})")
    ax.axhline(REWARD_THRESHOLD, color=_ACCENT3, ls="--", lw=1.2, label=f"Target ({REWARD_THRESHOLD})")
    ax.fill_between(eps, logger.rolling_reward, alpha=0.15, color=_ACCENT2)
    ax.set_title(f"Average Reward (Rolling Window = {logger.window})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Avg Reward")
    ax.legend(fontsize=8); ax.grid(True)
    _save(fig, "02_average_reward.png")


def plot_convergence_rate(logger):
    w = logger.window
    rates = []
    for i in range(1, len(logger.ep_rewards) + 1):
        chunk = logger.ep_rewards[max(0, i - w): i]
        rates.append(np.mean([1 if r >= REWARD_THRESHOLD else 0 for r in chunk]))
    fig, ax = plt.subplots(figsize=(10, 4))
    eps = np.arange(1, len(rates) + 1)
    ax.plot(eps, rates, color=_ACCENT3, lw=2, label=f"Convergence rate (w={w})")
    ax.axhline(SUCCESS_THRESHOLD, color=_ACCENT4, ls="--", lw=1.2,
               label=f"Target ({SUCCESS_THRESHOLD})")
    ax.fill_between(eps, rates, alpha=0.15, color=_ACCENT3)
    if logger.convergence_ep:
        ax.axvline(logger.convergence_ep, color=_ACCENT5, ls=":", lw=1.5,
                   label=f"Converged @ ep {logger.convergence_ep}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Convergence Rate Over Training", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Fraction of episodes ≥ reward threshold")
    ax.legend(fontsize=8); ax.grid(True)
    _save(fig, "03_convergence_rate.png")


def plot_sample_efficiency(logger):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(logger.total_steps_log, logger.rolling_reward, color=_ACCENT4, lw=2,
            label=f"Rolling avg reward (w={logger.window})")
    ax.axhline(REWARD_THRESHOLD, color=_ACCENT2, ls="--", lw=1.2,
               label=f"Target ({REWARD_THRESHOLD})")
    eff = logger.sample_efficiency()
    if eff:
        ax.axvline(eff, color=_ACCENT3, ls=":", lw=1.5,
                   label=f"Steps to converge: {eff:,}")
    ax.set_title("Sample Efficiency  (Cumulative Steps vs Rolling Reward)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Environment Steps"); ax.set_ylabel("Avg Reward")
    ax.legend(fontsize=8); ax.grid(True)
    _save(fig, "04_sample_efficiency.png")


def plot_success_rate(logger):
    fig, ax = plt.subplots(figsize=(10, 4))
    eps = np.arange(1, len(logger.rolling_success) + 1)
    ax.plot(eps, logger.rolling_success, color=_ACCENT5, lw=2,
            label=f"Success rate (w={logger.window})")
    ax.fill_between(eps, logger.rolling_success, alpha=0.15, color=_ACCENT5)
    ax.axhline(SUCCESS_THRESHOLD, color=_ACCENT3, ls="--", lw=1.2,
               label=f"Target ({SUCCESS_THRESHOLD})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Success Rate Over Training", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate")
    ax.legend(fontsize=8); ax.grid(True)
    _save(fig, "05_success_rate.png")


def plot_q_loss(logger):
    losses = [l for l in logger.ep_losses if not math.isnan(l)]
    if not losses:
        print("  No losses recorded – skipping Q-loss graph.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    eps = np.arange(1, len(losses) + 1)
    ax.plot(eps, losses, color=_ACCENT1, alpha=0.3, lw=0.8, label="Per-episode mean loss")
    sm = smooth(losses, 30)
    ax.plot(np.arange(30, len(losses) + 1), sm, color=_ACCENT1, lw=2, label="Smoothed (w=30)")
    ax.set_title("Q-Loss During Training", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=8); ax.grid(True)
    _save(fig, "06_q_loss.png")


def plot_tau(logger):
    fig, ax = plt.subplots(figsize=(10, 3))
    eps = np.arange(1, len(logger.ep_taus) + 1)
    ax.plot(eps, logger.ep_taus, color=_ACCENT3, lw=2)
    ax.fill_between(eps, logger.ep_taus, alpha=0.15, color=_ACCENT3)
    ax.set_title("Boltzmann Temperature (τ) Decay", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Temperature τ")
    ax.set_ylim(-0.02, 1.05); ax.grid(True)
    _save(fig, "07_tau_decay.png")


def plot_episode_length(logger):
    fig, ax = plt.subplots(figsize=(10, 4))
    eps = np.arange(1, len(logger.ep_lengths) + 1)
    ax.plot(eps, logger.ep_lengths, color=_MUTED, alpha=0.3, lw=0.8, label="Episode length")
    sm = smooth(logger.ep_lengths, 30)
    ax.plot(np.arange(30, len(logger.ep_lengths) + 1), sm,
            color=_ACCENT2, lw=2, label="Smoothed (w=30)")
    ax.set_title("Episode Length Over Training", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
    ax.legend(fontsize=8); ax.grid(True)
    _save(fig, "08_episode_length.png")


def plot_eval_bar(eval_metrics):
    labels = ["Total Reward", "Avg Reward", "Conv. Rate", "Success Rate"]
    values = [
        eval_metrics["total_reward"],
        eval_metrics["average_reward"],
        eval_metrics["convergence_rate"],
        eval_metrics["success_rate"],
    ]
    colors = [_ACCENT1, _ACCENT2, _ACCENT3, _ACCENT5]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor=_GRID, linewidth=0.6, width=0.55)
    ax.axhline(0, color=_GRID, lw=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (abs(bar.get_height()) * 0.02 + 0.3),
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=_TEXT)
    ax.set_title(f"Post-Training Evaluation  ({HP.EVAL_EPISODES} episodes)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Value"); ax.grid(True, axis="y")
    eff = eval_metrics.get("sample_efficiency")
    note = f"Sample efficiency: {eff:,} steps to converge" if eff else "Sample efficiency: not converged"
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=_MUTED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_PANEL, edgecolor=_GRID))
    _save(fig, "09_eval_bar.png")


def plot_dashboard(logger, eval_metrics):
    fig = plt.figure(figsize=(18, 10), facecolor=_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.06, right=0.97, top=0.90, bottom=0.08,
                            wspace=0.38, hspace=0.48)

    def styled(ax):
        ax.set_facecolor(_PANEL)
        ax.grid(True)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)

    eps = np.arange(1, len(logger.ep_rewards) + 1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(eps, logger.ep_rewards, color=_ACCENT1, alpha=0.2, lw=0.7)
    sm0 = smooth(logger.ep_rewards, 30)
    ax0.plot(np.arange(30, len(logger.ep_rewards) + 1), sm0, color=_ACCENT1, lw=2)
    ax0.axhline(REWARD_THRESHOLD, color=_ACCENT2, ls="--", lw=1)
    ax0.set_title("Total Reward", color=_TEXT); ax0.set_xlabel("Episode", color=_MUTED)
    styled(ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(eps, logger.rolling_reward, color=_ACCENT2, lw=2)
    ax1.fill_between(eps, logger.rolling_reward, alpha=0.15, color=_ACCENT2)
    ax1.axhline(REWARD_THRESHOLD, color=_ACCENT3, ls="--", lw=1)
    ax1.set_title("Rolling Avg Reward", color=_TEXT); ax1.set_xlabel("Episode", color=_MUTED)
    styled(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(eps, logger.rolling_success, color=_ACCENT5, lw=2)
    ax2.fill_between(eps, logger.rolling_success, alpha=0.15, color=_ACCENT5)
    ax2.axhline(SUCCESS_THRESHOLD, color=_ACCENT3, ls="--", lw=1)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Success Rate", color=_TEXT); ax2.set_xlabel("Episode", color=_MUTED)
    styled(ax2)

    w = logger.window
    rates = []
    for i in range(1, len(logger.ep_rewards) + 1):
        chunk = logger.ep_rewards[max(0, i - w): i]
        rates.append(np.mean([1 if r >= REWARD_THRESHOLD else 0 for r in chunk]))
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(eps, rates, color=_ACCENT3, lw=2)
    ax3.fill_between(eps, rates, alpha=0.15, color=_ACCENT3)
    ax3.axhline(SUCCESS_THRESHOLD, color=_ACCENT4, ls="--", lw=1)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Convergence Rate", color=_TEXT); ax3.set_xlabel("Episode", color=_MUTED)
    styled(ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(logger.total_steps_log, logger.rolling_reward, color=_ACCENT4, lw=2)
    ax4.axhline(REWARD_THRESHOLD, color=_ACCENT2, ls="--", lw=1)
    eff = logger.sample_efficiency()
    if eff:
        ax4.axvline(eff, color=_ACCENT3, ls=":", lw=1.5)
    ax4.set_title("Sample Efficiency", color=_TEXT); ax4.set_xlabel("Total Steps", color=_MUTED)
    styled(ax4)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    eff_str = f"{eff:,}" if eff else "N/A"
    summary = (
        f"{'═'*30}\n"
        f"  POST-TRAINING EVALUATION\n"
        f"  ({HP.EVAL_EPISODES} episodes)\n"
        f"{'═'*30}\n"
        f"  Total Reward      : {eval_metrics['total_reward']:.2f}\n"
        f"  Avg Reward/Ep     : {eval_metrics['average_reward']:.3f}\n"
        f"  Convergence Rate  : {eval_metrics['convergence_rate']:.3f}\n"
        f"  Success Rate      : {eval_metrics['success_rate']:.3f}\n"
        f"  Sample Efficiency : {eff_str} steps\n"
        f"{'─'*30}\n"
        f"  Conv. Episode     : {logger.convergence_ep or 'N/A'}\n"
        f"  Training Episodes : {HP.TRAIN_EPISODES}\n"
        f"{'═'*30}"
    )
    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             va="top", ha="left", fontsize=9.5, color=_TEXT,
             fontfamily="monospace",
             bbox=dict(facecolor=_PANEL, edgecolor=_ACCENT1, boxstyle="round,pad=0.5", lw=1.5))

    fig.suptitle("SARSA Agent — Full Training Dashboard", fontsize=15,
                 fontweight="bold", color=_TEXT)
    _save(fig, "10_dashboard.png")


# ══════════════════════════════════════════════════════════
#  6. Training Loop
# ══════════════════════════════════════════════════════════
def train(agent, env, replay_buffer, logger):

    print(f"\n{'='*55}")
    print(f"  SARSA TRAINING  —  {HP.TRAIN_EPISODES} episodes")
    print(f"  Device: {agent.device}")
    print(f"{'='*55}\n")

    for ep in range(1, HP.TRAIN_EPISODES + 1):
        obs, _ = env.reset()
        action    = agent.select_action_epsilon(obs)
        ep_reward = 0.0
        ep_steps  = 0
        ep_losses = []
        success   = False
        done      = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done    = terminated or truncated
            success = bool(terminated and reward >= SUCCESS_REWARD * 0.9)

            next_action = agent.select_action_epsilon(next_obs)

            replay_buffer.add(obs, action, reward, next_obs, next_action, done)

            loss = agent.update(replay_buffer)
            if loss is not None:
                ep_losses.append(loss)

            obs    = next_obs
            action = next_action
            ep_reward += reward
            ep_steps  += 1
            agent.total_steps += 1

        agent.decay_epsilon()
        mean_loss = float(np.mean(ep_losses)) if ep_losses else None
        # CHANGE logging (τ → ε)
        logger.log(ep_reward, ep_steps, success, mean_loss,
                   agent.epsilon, agent.total_steps)

        if ep % 100 == 0 or ep == 1:
            rr = logger.rolling_reward[-1]
            sr = logger.rolling_success[-1]
            print(f"  Ep {ep:4d}/{HP.TRAIN_EPISODES} | "
                  f"reward {ep_reward:+7.2f} | "
                  f"rolling {rr:+6.2f} | "
                  f"success {sr:.2f} | "
                  f"ε {agent.epsilon:.3f} | "
                  f"steps {agent.total_steps:,}")

    print(f"\n  Training complete.  Total steps: {agent.total_steps:,}")
    if logger.convergence_ep:
        print(f"  Converged at episode {logger.convergence_ep}  "
              f"({logger.total_steps_log[logger.convergence_ep-1]:,} steps)")
    else:
        print("  ⚠  Did not meet both convergence thresholds during training.")


# ══════════════════════════════════════════════════════════
#  7. Post-Training Evaluation
# ══════════════════════════════════════════════════════════
def evaluate(agent, env, logger, n_episodes=HP.EVAL_EPISODES):

    print(f"\n{'='*55}")
    print(f"  POST-TRAINING EVALUATION  —  {n_episodes} episodes (greedy τ→0)")
    print(f"{'='*55}\n")

    rewards, successes, lengths = [], [], []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        success   = False
        done      = False

        while not done:
            # Modify evaluation
            q_vals = agent.q_values(obs)
            action = int(np.argmax(q_vals))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward >= SUCCESS_REWARD * 0.9:
                success = True
            ep_reward += reward
            ep_steps  += 1

        rewards.append(ep_reward)
        successes.append(int(success))
        lengths.append(ep_steps)

        if ep % 10 == 0:
            print(f"  Eval ep {ep:3d}/{n_episodes} | "
                  f"reward {ep_reward:+7.2f} | "
                  f"{'SUCCESS' if success else 'fail   '}")

    metrics = {
        "total_reward":      round(float(np.sum(rewards)),   3),
        "average_reward":    round(float(np.mean(rewards)),  3),
        "convergence_rate":  round(float(np.mean([1 if r >= REWARD_THRESHOLD else 0 for r in rewards])), 3),
        "success_rate":      round(float(np.mean(successes)), 3),
        "sample_efficiency": logger.sample_efficiency(),
    }

    print(f"\n  ── Final Evaluation Metrics ──────────────────────")
    for k, v in metrics.items():
        vstr = f"{v:,}" if isinstance(v, int) else (f"{v:.3f}" if v is not None else "N/A")
        print(f"    {k:<22}: {vstr}")
    print(f"  {'─'*48}")

    return metrics


# ══════════════════════════════════════════════════════════
#  8. Save All 10 Graphs
# ══════════════════════════════════════════════════════════
def save_all_graphs(logger, eval_metrics):
    print(f"\n  Saving graphs to '{OUT_DIR}/' ...")
    plot_total_reward(logger)
    plot_average_reward(logger)
    plot_convergence_rate(logger)
    plot_sample_efficiency(logger)
    plot_success_rate(logger)
    plot_q_loss(logger)
    plot_tau(logger)
    plot_episode_length(logger)
    plot_eval_bar(eval_metrics)
    plot_dashboard(logger, eval_metrics)
    print(f"  All 10 graphs saved.\n")


# ══════════════════════════════════════════════════════════
#  9. Colab Display
# ══════════════════════════════════════════════════════════
def display_graphs_in_colab():
    if not COLAB:
        print("  Not in Colab — open the PNG files in the sarsa_results_εgreedy_v3/ folder.")
        return
    files = sorted([f for f in os.listdir(OUT_DIR) if f.endswith(".png")])
    for fname in files:
        path = os.path.join(OUT_DIR, fname)
        print(f"\n  ── {fname} ──")
        display(IPImage(filename=path))


# ══════════════════════════════════════════════════════════
#  10. Main
# ══════════════════════════════════════════════════════════
def main():
    env           = WheelchairNavEnv(action_type="discrete", n_people=3,
                                     n_obstacles=4, render_mode=None)
    agent         = SARSAAgent(
        state_dim     = HP.STATE_DIM,
        n_actions     = HP.N_ACTIONS,
        hidden        = HP.HIDDEN_SIZE,
        learning_rate = HP.LR,
        gamma         = HP.GAMMA,
    )
    replay_buffer = ReplayBuffer(HP.STATE_DIM, HP.BUFFER_SIZE)
    logger        = MetricsLogger(window=HP.CONV_WINDOW)

    train(agent, env, replay_buffer, logger)

    model_path = os.path.join(OUT_DIR, "sarsa_wheelchair_converged_εgreedy_v3.pt")
    agent.save(model_path)

    eval_metrics = evaluate(agent, env, logger, n_episodes=HP.EVAL_EPISODES)

    save_all_graphs(logger, eval_metrics)
    display_graphs_in_colab()

    env.close()
    print("\n  Done!  All outputs in:", OUT_DIR)
    return agent, logger, eval_metrics


if __name__ == "__main__":
    main()
