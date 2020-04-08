import gym
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optimal
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
import math
import pandas as pd

env = gym.make('MountainCar-v0')  # Make environment
env.seed(1); torch.manual_seed(1); np.random.seed(1)

# HYPERPARAMETERS
steps = 200
decay = 0.95 # Epsilon Decay
alpha = 0.001 # Learning rate
gamma = 0.99 # Scheduler Parameter
max_episodes = 100  # Max episode
epsilon = 0.3 # Mutation rate

# DATA LOGGING
reward_history = np.zeros(max_episodes)
loss_history = np.zeros(max_episodes)
epsilon_history = np.zeros(max_episodes)
max_position_history = np.zeros(max_episodes)
final_position_history = np.zeros(max_episodes)
state_history = np.zeros(max_episodes)
max_position = -math.inf
successes = 0

# MATPLOTLIB
plt.ion()
figure = plt.figure()
fig, ax = plt.subplots(2, 2)
fig.tight_layout()


# DQN NETWORK
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 100
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)

# DQN NETWORK
class Predict(nn.Module):
    def __init__(self):
        super(Predict, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.hidden = 100
        self.l1 = nn.Linear(self.state_space+1, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.state_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)


# INITIALIZE DQN
policy = Policy()
predict = Predict()
loss_fn = nn.MSELoss()
predict_loss_fn = nn.MSELoss()
optimizer = optimal.SGD(policy.parameters(), lr=alpha)
predict_optimizer = optimal.SGD(predict.parameters(), lr=alpha)
scheduler = optimal.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
predict_scheduler = optimal.lr_scheduler.StepLR(predict_optimizer, step_size=1, gamma=0.9)

progress_bar = tqdm(range(max_episodes), bar_format="{l_bar}{bar}|{postfix}")

for ep in range(max_episodes):
    progress_bar.update(1)
    progress_bar.set_postfix(Epsilon=str(round(epsilon, 2)))

    # RESET
    total_reward = 0
    episode_loss = 0
    state = env.reset()

    for s in range(steps):
        #  sample Q values from policy
        Q = policy.forward(torch.from_numpy(state).type(torch.FloatTensor))  # " Q(s, a) "

        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            #  find optimal action according to policy (over axis -1)
            _, max = torch.max(Q, -1)
            action = max.item()

        state_est = predict.forward(torch.from_numpy(np.concatenate((state, [action]))).type(torch.FloatTensor))

        new_state, reward, terminal, _ = env.step(action)
        # Adjust reward based on car position
        reward = new_state[0] + 0.5

        if new_state[0] > max_position:
            max_position = new_state[0]

        if new_state[0] >= 5:
            reward += 1

        total_reward += reward

        Q_new = policy.forward(torch.from_numpy(new_state).type(torch.FloatTensor))  # " find Q (s', a') "


        #  find optimal action Q value for next step
        new_max, _ = torch.max(Q_new, -1)  # " max(Q(s', a')) "

        Q_target = Q.clone()
        Q_target = Variable(Q_target.data)

        #  update target value function according to TD
        Q_target[action] = reward + torch.mul(new_max.detach(), gamma)  # " reward + gamma*(max(Q(s', a')) "

        # Calculate loss
        predict_loss = predict_loss_fn(torch.Tensor(state_est), torch.Tensor(new_state))
        loss = loss_fn(Q, Q_target)  # " reward + gamma*(max(Q(s', a')) - Q(s, a)) "
        episode_loss += predict_loss.item()

        # Update original policy according to Q_target ( supervised learning )
        predict.zero_grad()
        policy.zero_grad()
        loss.backward()
        predict_loss.backward()
        optimizer.step()
        predict_optimizer.step()

        #if state_est[0] < new_state[0]+0.01 and state_est[0] > new_state[0]-0.01 and state_est[1] < new_state[1]+0.01 and state_est[1] > new_state[1]-0.01:
            #predict_scheduler.step()

        #  Q and Q_target should converge

        if terminal:
            # Adjust hyperparameters on success
            if new_state[0] >= 0.5:
                successes += 1
                epsilon *= decay
                scheduler.step()
                print("Success! "+str(successes)+" so far.")

            break
        else:
            state = new_state # Reset state

    # LOGGING
    loss_history[ep] = episode_loss
    reward_history[ep] = total_reward
    max_position_history[ep] = max_position
    epsilon_history[ep] = epsilon
    final_position_history[ep] = new_state[0]
    state_history[ep] = state_est[0]

    # GRAPHING
    p = pd.Series(final_position_history[:ep])
    ma = p.rolling(10).mean()

    if ep % 10 == 0:
        ax[0][0].plot(reward_history[:ep], color="red")
        ax[1][0].plot(loss_history[:ep], color="red")
        ax[0][0].set_title('Reward History')
        ax[1][0].set_title('Loss History')
        ax[0][1].set_title('Max Position')
        ax[0][1].plot(ma, color="red")
        ax[0][1].plot(max_position_history[:ep], color="blue")
        ax[1][1].plot(state_history[:ep], color="red")
        ax[1][1].plot(final_position_history[:ep], color="blue")
        ax[1][1].set_title('Final Position History')
        fig.canvas.draw()
    plt.pause(0.0001)

# SAVE MODEL
torch.save(policy.state_dict(), 'trained-10000.mdl')
torch.save(predict.state_dict(), 'predict-1000.mdl')
