"""
This eval.py is used to evaluate 1000 episode after RL training is done.
The fixed seed produce results without variance changing.
By evaluating 1000 episode, we can evaluate the trained model's performance without bias.
"""
import numpy as np
from stable_baselines3 import TD3
from wheelchair_env import WheelchairNavEnv
import random
import numpy as np
import torch


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model = TD3.load('td3_wheelchair_final.zip')

env = WheelchairNavEnv(
    action_type='continuous',
    n_people=3,
    n_obstacles=4,
    render_mode='rgb_array',
    seed=SEED
)

N_EVAL = 1000
eval_rewards = []
eval_success = []
eval_collisions = []
eval_timeouts = []


for ep in range(N_EVAL):
    obs, _ = env.reset(seed=SEED + ep)
    total_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # check outcome only after episode ends
    dist = info.get("dist_to_goal", float("inf"))

    success   = int(terminated and dist <= 0.4)
    collision = int(terminated and dist > 0.4)
    timeout   = int(truncated)

    eval_rewards.append(total_reward)
    eval_success.append(success)
    eval_collisions.append(collision)
    eval_timeouts.append(timeout)

    print(f"Ep {ep+1:>3} | reward: {total_reward:>8.2f} | {'GOAL' if success else 'COLLISION' if collision else 'TIMEOUT'}")
print()
print(f"  Total reward   : {np.sum(eval_rewards):.4f}")
print(f"  Average reward : {np.mean(eval_rewards):.4f}")
print(f"  Success rate : {np.mean(eval_success)*100:.2f} %")
print(f"  Collision rate: {np.mean(eval_collisions)*100:.2f} %")
print(f"  Timeout rate  : {np.mean(eval_timeouts)*100:.2f} %")