Reinforcement Learning (RL) enables the wheelchair to learn optimal decisions directly from continuous user interaction through the method of trial-and-error. This learning process allows the system to adapt dynamically to individual users based on their current situation and preferences. For example, a RL-based wheelchair may learn to reduce assistance on disabled persons as they gain confidence or skill, or increase support automatically during periods of fatigue or difficulty. Additionally, RL enables long-term decision-making optimization, allowing the system to balance immediate task success with the goal to achieve desired outcomes through continuous exploration and exploitation (Manager & Manager, 2025).


DDPG (Deep Deterministic Policy Gradient) is an off-policy, model-free reinforcement learning algorithm that uses an actor-critic method to solve continuous control problems with a continuous action space. It expands on the Q-learning and actor-critic algorithm and uses the Bellman equation and off-policy data points to remember the Q-function, which it subsequently uses to learn the policy. It is highly connected to Q-learning, and it is also one of the popular methods used. (Sumiea et al., 2024)

DDPG with OU Noise is the best-performing model when prioritising real-world outcomes, success rate and average reward.
https://github.com/user-attachments/assets/cf649535-daf4-458d-b939-c5748a2c25e3

