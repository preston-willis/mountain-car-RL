import gym
import random
import numpy as np
from tqdm import tqdm

env = gym.make('MountainCar-v0')  # Make environment
decay = 0.9993
alpha = 0.1
gamma = 1
q = np.zeros((19, 15, 3))  # State-action-value matrix
ep = 0  # Episode
max = 21000  # Max episode
epsilon = 1  # Mutation rate


def state_formatter(s):
    """
    formats state space to positive wole numbers
    """

    # Round state space to tenths place to form finite values

    s[0] = s[0].round(1)
    s[1] = s[1].round(2)

    # Make numbers whole

    s[0] *= 10
    s[1] *= 100

    # Make numbers whole

    s[0] += 12
    s[1] += 7

    # Change data type

    integer = [0, 0]
    integer[0] = int(s[0])
    integer[1] = int(s[1])

    return integer


pbar = tqdm(range(max), ascii=" .oO0", bar_format="{l_bar}{bar}|{postfix}")
policy = [[env.action_space.sample() for _ in range(0, 15)] for _ in range(0, 19)]
policy = np.array(policy)

for ep in range(max):
    pbar.update(1)
    epsilon *= decay
    if epsilon < 0.1:
        epsilon = 0.1  # Update epsilon for convergence
    if ep == 20000:
        epsilon = 0
        alpha = 0
    pbar.set_postfix(Epsilon=str(round(epsilon, 2)))

    state = state_formatter(env.reset())  # Reset environment
    while True:

        if ep > 20000 and ep is not 0:
            env.render()
        else:
            env.close()

        if random.uniform(0, 1) > epsilon:
            action = np.argmax(q[state[0], state[1], :])
            policy[state[0], state[1]] = action
        else:
            action = env.action_space.sample()

        new_state, reward, end, info = env.step(action)

        if end:
            break

        new_state = state_formatter(new_state)

        q[state[0], state[1], action] += alpha * (reward + gamma*np.max(q[new_state[0], new_state[1], :]) -
                                                  q[state[0], state[1], action])

        state = new_state
