# SoC-2025-Super-Mario-Quest-using-RL
By Khushi Kucheria Roll Number : 24B1219
# 🕹Super Mario Quest using Reinforcement Learning

A fun and educational reinforcement learning (RL) journey — starting with grid-based Q-learning in **Taxi-v3**, and progressing to training an intelligent **Super Mario Bros** agent using **Proximal Policy Optimization (PPO)**.

This project was developed as part of the **SoC 2025** program and showcases how RL techniques scale from simple environments to visually complex games.

---

## Week-by-Week Breakdown

### Week 2: Q-Learning with Taxi-v3

The `Assignment-1_Taxi-v3.pdf` notebook focuses on implementing and evaluating a Q-learning agent for the `Taxi-v3` environment from OpenAI Gym.

* **Environment**: `Taxi-v3` is a classic reinforcement learning environment where an agent learns to navigate a taxi to pick up and drop off passengers.
* **Algorithm**: Q-learning, a value-based reinforcement learning algorithm, is used to train the agent.
* **Key Parameters**:
    * `alpha` (learning rate): 0.1
    * `gamma` (discount factor): 0.6
    * `epsilon` (exploration rate): 0.1
* **Training**: The agent is trained for 100,000 episodes.
* **Evaluation**: After training, the agent's performance is evaluated over 10,000 episodes, showing metrics like average timesteps, penalties, and rewards per episode.
    * Average timesteps per episode: 14.80
    * Average penalties per episode: 0.44
    * Average rewards per episode: 2.20

### Week 3: Super Mario Bros Environment Setup

We set up the **Super Mario Bros** environment using `gym-super-mario-bros`.

#### 🧰 Setup Included:
- Environment: `SuperMarioBrosRandomStages-v0`
- Action Set: `SIMPLE_MOVEMENT` via `JoypadSpace`
- Tested random action rollout and printed frame shapes
- Explored rendering, reward signals, and `info` dictionary

🔗 **Code:** [`env_setup.py`](./env_setup.py)

### Week 4: Exploring Super Mario Bros Environment

The `week_4.py` provides a basic interaction with the Super Mario Bros environment to understand its observation space and information dictionary.

* **Environment**: `SuperMarioBros-1-1-v0` is set up with `COMPLEX_MOVEMENT`.
* **Observation Space**: The code demonstrates how to reset the environment and print the `info` dictionary and the shape of the observation (state). This helps in understanding the kind of data the agent will receive (e.g., `'coins'`, `'life'`, `'score'`, `'x_pos'`, `'y_pos'` in the `info` dictionary, and the `(240, 256, 3)` shape for the RGB image observations).

### Week 5: Training PPO Models for Super Mario Bros

The `week_5.ipynb` notebook showcases successful training of PPO agents for Super Mario Bros using different movement sets and on a different level, building upon the environment setup.

* **Environment Preparation**: Similar to Week 3, the environment is wrapped with `JoypadSpace`, `ResizeObservation` (to 84x84), `GrayScaleObservation` (keeping dimensions), `Monitor` (for logging stats), `DummyVecEnv`, and `VecFrameStack` (n\_stack=4).

* **Simple Movement Training**:
    * **Movement Set**: `SIMPLE_MOVEMENT` is used.
    * **Training Time**: The model is trained for 100,000 total timesteps.
    * **Performance**: During training, the agent shows increasing `ep_rew_mean` (mean episode reward) and `ep_len_mean` (mean episode length), indicating learning progress. The final reported mean episode reward is 552, and the mean episode length is approximately 16,200 timesteps.
    * **Saving**: The trained model is saved to `train/ppo_mario_simple/mario_model`.

---
