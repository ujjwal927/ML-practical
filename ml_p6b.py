
import numpy as np
import random
import gymnasium as gym
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


# Step 1: Create Taxi Environment
env = gym.make('Taxi-v3')

n_states = env.observation_space.n
n_actions = env.action_space.n

print(f'States: {n_states}, Actions: {n_actions}')


# Step 2: Initialize Q-Table
Q = np.zeros((n_states, n_actions))

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995

n_episodes = 5000
max_steps = 200

rewards_per_episode = []

random.seed(42)
np.random.seed(42)


# Step 3: Train Q-Learning Agent
for episode in range(n_episodes):

    state, _ = env.reset(seed=episode)

    total_reward = 0
    done = False

    for step in range(max_steps):

        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()

        else:
            action = int(np.argmax(Q[state]))

        # Execute Action
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Update Q-Table using Bellman Equation
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        state = next_state

        total_reward += reward

        if done:
            break

    # Decay Exploration Rate
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    rewards_per_episode.append(total_reward)


# Step 4: Calculate Moving Average
window = 100

moving_avg = np.convolve(
    rewards_per_episode,
    np.ones(window) / window,
    mode='valid'
)


# Step 5: Plot Learning Curve
plt.figure(figsize=(8, 5))

plt.plot(moving_avg)

plt.xlabel('Episode')

plt.ylabel(
    f'Reward (moving average, window={window})'
)

plt.title('Q-Learning: Reward per Episode (Taxi Problem)')

plt.grid(True)

plt.tight_layout()

plt.savefig('taxi_learning_curve.png', dpi=120)

plt.close()


# Step 6: Evaluate Trained Agent
test_episodes = 20

total_rewards = []
total_steps = []

for ep in range(test_episodes):

    state, _ = env.reset(seed=1000 + ep)

    done = False
    steps = 0
    ep_reward = 0

    while not done and steps < max_steps:

        # Choose Best Action
        action = int(np.argmax(Q[state]))

        # Execute Greedy Action
        state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        ep_reward += reward

        steps += 1

    total_rewards.append(ep_reward)
    total_steps.append(steps)


# Step 7: Display Evaluation Results
print(
    f'\nAverage reward over {test_episodes} '
    f'greedy test episodes: {np.mean(total_rewards):.2f}'
)

print(
    f'Average steps taken: {np.mean(total_steps):.2f}'
)


# Step 8: Project Completion
print('\nTaxi Q-Learning Project Completed!')

print('Generated file: taxi_learning_curve.png')

env.close()
