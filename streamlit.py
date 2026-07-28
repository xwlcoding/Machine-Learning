"""
Streamlit Live Dashboard — Wheelchair Navigation
"""

import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wheelchair Nav — Live Dashboard",
    page_icon="🦽",
    layout="wide",
)

st.title("🦽 Wheelchair Navigation")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    n_episodes  = st.slider("Episodes to run", 1, 20, 5)
    n_people    = st.slider("Moving people",   0,  8, 3)
    n_obstacles = st.slider("Static obstacles",0,  8, 4)
    seed        = st.number_input("Env seed", value=42, step=1)
    run_btn  = st.button("▶ Run", use_container_width=True, type="primary")
    stop_btn = st.button("⏹ Stop", use_container_width=True)

# ── Layout: map (large) | metrics+charts (right) ─────────────────────────────
col_map, col_right = st.columns([2, 1], gap="large")

with col_map:
    st.subheader("Agent View")
    map_placeholder = st.empty()

with col_right:
    st.subheader("Live Metrics")
    m1, m2, m3 = st.columns(3)
    ep_metric     = m1.empty()
    step_metric   = m2.empty()
    reward_metric = m3.empty()

    st.markdown("**Cumulative Reward**")
    cum_reward_chart = st.empty()

    st.markdown("**Distance to Goal**")
    dist_chart = st.empty()

    st.markdown("**Step Reward**")
    step_reward_chart = st.empty()

# ── Bottom row: reward breakdown | obs vector ─────────────────────────────────
st.divider()
col_rew, col_obs = st.columns(2, gap="large")

with col_rew:
    st.subheader("Reward Breakdown")
    reward_bars_placeholder = st.empty()

with col_obs:
    st.subheader("Observation Vector")
    obs_bars_placeholder = st.empty()

st.divider()
episode_log = st.empty()

# ── Session state ─────────────────────────────────────────────────────────────
if "stop" not in st.session_state:
    st.session_state.stop = False
if "log" not in st.session_state:
    st.session_state.log = []

if stop_btn:
    st.session_state.stop = True

# ── Helper: render individual panels as PIL images ────────────────────────────
_BG = "#0d1117"

def render_reward_bars(env) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 3), facecolor=_BG)
    env._draw_reward_bars(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def render_obs_bars(env, obs) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 3), facecolor=_BG)
    env._draw_obs_bars(ax, obs)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# ── Run logic ─────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.stop = False
    st.session_state.log  = []

    try:
        from stable_baselines3 import DDPG
        from wheelchair_env import WheelchairNavEnv
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.stop()

    MODEL_PATH = "ddpg_wheelchair_OUv2"
    with st.spinner("Loading model ..."):
        try:
            model = DDPG.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}\n\nMake sure ddpg_wheelchair_final.zip is in the same folder.")
            st.stop()

    env = WheelchairNavEnv(
        action_type  = "continuous",
        n_people     = n_people,
        n_obstacles  = n_obstacles,
        render_mode  = "png_bytes",
        seed         = int(seed),
    )

    # ── Accumulate per-episode stats for the summary ──────────────────────────
    all_episodes_data = []  # list of dicts, one per episode

    for ep in range(1, n_episodes + 1):
        if st.session_state.stop:
            st.warning("Stopped by user.")
            break

        obs, _ = env.reset()
        total_reward = 0.0
        done         = False
        step         = 0
        cum_rewards  = []
        dist_history = []
        step_rewards = []

        while not done:
            if st.session_state.stop:
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done          = terminated or truncated
            step         += 1

            cum_rewards.append(total_reward)
            dist_history.append(info["dist_to_goal"])
            step_rewards.append(reward)

            # ── Large map (top left) ──────────────────────────────────────────
            map_png = env.render()
            map_img = Image.open(io.BytesIO(map_png))
            map_placeholder.image(map_img, use_container_width=True)

            # ── Metric cards ──────────────────────────────────────────────────
            ep_metric.metric("Episode", f"{ep}/{n_episodes}")
            step_metric.metric("Step", step)
            reward_metric.metric("Cumul. reward", f"{total_reward:.2f}")

            # ── Line charts (top right) ───────────────────────────────────────
            cum_reward_chart.line_chart(
                pd.DataFrame({"Cumulative Reward": cum_rewards}), height=130)
            dist_chart.line_chart(
                pd.DataFrame({"Dist to Goal (m)": dist_history}), height=130)
            step_reward_chart.line_chart(
                pd.DataFrame({"Step Reward": step_rewards}), height=130)

            # ── Reward breakdown + obs vector (bottom row) ────────────────────
            reward_bars_placeholder.image(
                render_reward_bars(env), use_container_width=True)
            obs_bars_placeholder.image(
                render_obs_bars(env, obs), use_container_width=True)

        # ── Episode summary ───────────────────────────────────────────────────
        outcome = (
            "✅ GOAL"          if terminated and info["dist_to_goal"] < 0.4
            else "💥 COLLISION" if terminated
            else "⏱ TIMEOUT"
        )
        log_line = (
            f"**Ep {ep}/{n_episodes}** — {outcome} | "
            f"steps={step} | total={total_reward:.2f} | "
            f"final dist={info['dist_to_goal']:.2f} m"
        )
        st.session_state.log.append(log_line)
        episode_log.markdown("### Episode Log\n" + "\n\n".join(st.session_state.log))

        # ── Store episode stats ───────────────────────────────────────────────
        all_episodes_data.append({
            "Episode":         ep,
            "Total Reward":    total_reward,
            "Steps":           step,
            "Final Dist (m)":  info["dist_to_goal"],
            "Outcome":         outcome,
        })

    env.close()
    st.success("✅ All episodes complete!")

    # ── Overall summary charts ────────────────────────────────────────────────
    if all_episodes_data:
        st.divider()
        st.header("📊 Overall Summary")

        summary_df = pd.DataFrame(all_episodes_data).set_index("Episode")

        # ── Aggregate metrics ─────────────────────────────────────────────────
        avg_reward      = summary_df["Total Reward"].mean()
        avg_steps       = summary_df["Steps"].mean()
        n_success       = summary_df["Outcome"].str.startswith("✅").sum()
        avg_success_pct = n_success / len(summary_df) * 100

        am1, am2, am3 = st.columns(3, gap="large")
        am1.metric("Avg Reward / Episode",  f"{avg_reward:.2f}")
        am2.metric("Avg Steps / Episode",   f"{avg_steps:.1f}")
        am3.metric("Avg Success Rate",      f"{avg_success_pct:.1f}%")

        st.divider()

        sc1, sc2 = st.columns(2, gap="large")
        sc3, sc4 = st.columns(2, gap="large")

        with sc1:
            st.markdown("**Total Reward per Episode**")
            st.bar_chart(summary_df[["Total Reward"]], height=250)

        with sc2:
            st.markdown("**Steps per Episode**")
            st.bar_chart(summary_df[["Steps"]], height=250)

        with sc3:
            st.markdown("**Final Distance to Goal per Episode**")
            st.bar_chart(summary_df[["Final Dist (m)"]], height=250)

        with sc4:
            st.markdown("**Outcome Breakdown**")
            outcome_counts = (
                summary_df["Outcome"]
                .value_counts()
                .rename_axis("Outcome")
                .reset_index(name="Count")
                .set_index("Outcome")
            )
            st.bar_chart(outcome_counts, height=250)

        # ── Summary stats table ───────────────────────────────────────────────
        st.markdown("**Episode Stats Table**")
        st.dataframe(
            summary_df.style.format({
                "Total Reward":   "{:.2f}",
                "Final Dist (m)": "{:.2f}",
            }),
            use_container_width=True,
        )