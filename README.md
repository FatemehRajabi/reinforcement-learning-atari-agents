# Reinforcement Learning: Atari Agents with Stable-Baselines3

This repository contains two reinforcement learning projects where I trained agents to play classic Atari games using policy-gradient algorithms (A2C and PPO).

---

## 🎮 Projects

### 1. Pong (A2C)
- Algorithm: Advantage Actor-Critic (A2C)
- Environment: PongNoFrameskip-v4
- Status: Agent reaches stable gameplay after ~1M timesteps

📂 [See Project Folder](./pong_a2c)

---

### 2. MsPacman (PPO)
- Algorithm: Proximal Policy Optimization (PPO)
- Environment: MsPacmanNoFrameskip-v4
- Status: Agent achieves strong reward after training over ~1M timesteps

📂 [See Project Folder](./mspacman_ppo)

---

## 🧰 Tech Stack
- Python
- Stable-Baselines3
- OpenAI Gym + Atari environments
- PyTorch
- OpenCV (for video capture)
- TensorBoard (optional for visualization)

---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/YOUR_USERNAME/reinforcement-learning-atari-agents.git
cd reinforcement-learning-atari-agents
python -m venv rl-env
source rl-env/bin/activate  # or .\rl-env\Scripts\activate on Windows
pip install -r requirements.txt
