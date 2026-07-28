# Intelligent Wheelchair Control using Reinforcement Learning

An advanced framework for adaptive wheelchair control leveraging **Reinforcement Learning (RL)** to dynamically adjust assistance based on user confidence, fatigue, and real-time interaction. 

---

## 📌 Overview

Traditional assistive wheelchair systems often rely on rigid, pre-programmed rules. This project implements RL-based control strategies that allow the wheelchair to learn optimal decisions through continuous trial-and-error. 

### Key Capabilities:
* **Adaptive Support:** Automatically reduces assistance as users gain skill/confidence and increases support during fatigue or difficult maneuvers.
* **Long-Term Optimization:** Balances immediate task success with overarching goals through continuous exploration and exploitation.
* **Continuous Control:** Utilizes advanced actor-critic frameworks designed for continuous action spaces.

---

## 🚀 Featured Algorithm: DDPG

The core architecture utilizes **Deep Deterministic Policy Gradient (DDPG)**, an off-policy, model-free algorithm. 
* Combines Q-learning principles with policy gradients using the Bellman equation.
* Tailored for continuous control problems.
* **Result:** **DDPG with Ornstein-Uhlenbeck (OU) Noise** demonstrated the best performance when prioritizing real-world outcomes, achieving the highest average rewards and top success rates.

> 📊 **Model Performance Visualization**  
> ![DDPG Performance](https://github.com/user-attachments/assets/cf649535-daf4-458d-b939-c5748a2c25e3)

---

## 📈 Algorithm Benchmarks & Results

Extensive evaluations were conducted across various algorithms, exploration strategies, and metrics (Average Rewards, Convergence Rate, Sample Efficiency, and Success Rate):

| Algorithms | Approach | Average Rewards | Convergence Rate | Sample Efficiency | Success Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TD3** | OU Noise | 90.22 | 79.60% | 355,053 | 70.00% |
| | Gaussian Noise Fading | 106.62 | 79.60% | 355,053 | 68.40% |
| **DQN** | Epsilon-greedy | 109.59 | 51.20% | 229,611 | 61.00% |
| | Thompson Sampling | 99.95 | 55.95% | 272,336 | 58.00% |
| **SARSA** | Epsilon-greedy | 99.79 | 82.00% | 152,291 | 69.70% |
| | Thompson Sampling | 77.05 | 86.00% | 217,606 | 68.50% |
| **DDPG** | **OU Noise** | **113.04** | **66.55%** | **280,591** | **70.80%** |
| | Thompson Sampling | 104.51 | 66.45% | 261,215 | 68.80% |

---

## References
* Manager, & Manager. (2025). Reinforcement Learning for Adaptive Assistive Robotics.
* Sumiea, et al. (2024). Deep Deterministic Policy Gradient Applications in Continuous Control Systems.

## 🛠️ Installation & Quick Start

```bash
# Clone the repository
git clone [https://github.com/your-username/intelligent-wheelchair-rl.git](https://github.com/your-username/intelligent-wheelchair-rl.git)
cd intelligent-wheelchair-rl

# Install dependencies
pip install -r requirements.txt

# Run the optimal DDPG model
python train.py --algorithm ddpg --noise ou
