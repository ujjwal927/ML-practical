
import numpy as np
import random
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


# Step 1: Define the Maze
# 0 = Free Cell, 1 = Wall

maze = np.array([
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])

ROWS, COLS = maze.shape

START = (0, 0)
GOAL = (4, 4)


# Step 2: Convert Row and Column into State
def rc_to_state(r, c):
    return r * COLS + c


# Step 3: Check Valid Movement
def is_valid(r, c):
    return (
        0 <= r < ROWS
        and 0 <= c < COLS
        and maze[r, c] == 0
    )


# Step 4: Define Environment Step Function
def step(state, action):

    r, c = divmod(state, COLS)

    nr, nc = r, c

    if action == 0:
        nr = r - 1       # Move Up

    elif action == 1:
        nr = r + 1       # Move Down

    elif action == 2:
        nc = c - 1       # Move Left

    elif action == 3:
        nc = c + 1       # Move Right

    if not is_valid(nr, nc):
        return state, -5.0, False

    next_state = rc_to_state(nr, nc)

    if (nr, nc) == GOAL:
        return next_state, 20.0, True

    return next_state, -1.0, False


# Step 5: Initialize Q-Learning Parameters
N_STATES = ROWS * COLS
N_ACTIONS = 4

Q = np.zeros((N_STATES, N_ACTIONS))

alpha = 0.1
gamma = 0.9

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

n_episodes = 800
max_steps = 200

random.seed(42)
np.random.seed(42)


# Step 6: Train Q-Learning Agent
for episode in range(n_episodes):

    state = rc_to_state(*START)

    for _ in range(max_steps):

        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, N_ACTIONS - 1)

        else:
            action = int(np.argmax(Q[state]))

        # Execute Action
        next_state, reward, done = step(state, action)

        # Q-Learning Update
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        state = next_state

        if done:
            break

    # Decay Exploration Rate
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# Step 7: Evaluate Learned Policy
state = rc_to_state(*START)

path = [START]

for _ in range(max_steps):

    action = int(np.argmax(Q[state]))

    state, reward, done = step(state, action)

    r, c = divmod(state, COLS)

    path.append((r, c))

    if done:
        break


# Step 8: Display Learned Path
print("\nLearned path from start to goal:")
print(path)

if path[-1] == GOAL:
    print("\nGoal reached successfully!")
else:
    print("\nGoal not reached. Increase training episodes.")


# Step 9: Visualize Maze and Learned Path
plt.figure(figsize=(5, 5))

plt.imshow(maze, cmap='Greys', interpolation='nearest')

path_r = [p[0] for p in path]
path_c = [p[1] for p in path]

plt.plot(
    path_c,
    path_r,
    marker='o',
    color='red',
    linewidth=2,
    label='Learned Path'
)

plt.scatter(
    [START[1]],
    [START[0]],
    color='green',
    s=150,
    label='Start'
)

plt.scatter(
    [GOAL[1]],
    [GOAL[0]],
    color='blue',
    s=150,
    label='Goal'
)

plt.xticks(range(COLS))
plt.yticks(range(ROWS))

plt.xlabel('Column')
plt.ylabel('Row')

plt.legend()

plt.title('Q-Learning: Learned Maze Path')

plt.tight_layout()

plt.savefig('maze_path.png', dpi=120)

plt.close()


# Step 10: Project Completion
print("\nQ-Learning Maze Solver Completed!")

print("Generated file: maze_path.png")
