import gym
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optimal
from torch.autograd import Variable

env = gym.make('MountainCar-v0')  # Make environment
decay = 0.993
alpha = 0.01
gamma = 0.9
max_episodes = 1000  # Max episode
epsilon = 1  # Mutation rate
reward_history = []


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 200
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)


policy = Policy()
loss_fn = nn.MSELoss()
optimizer = optimal.SGD(policy.parameters(), lr=alpha)
scheduler = optimal.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

progress_bar = tqdm(range(max_episodes), ascii=" .oO0", bar_format="{l_bar}{bar}|{postfix}")

for ep in range(max_episodes):
    progress_bar.update(1)
    progress_bar.set_postfix(Epsilon=str(round(epsilon, 2)))

    epsilon *= decay
    total_reward = 0

    if epsilon < 0.1:
        epsilon = 0.1

    if ep > 900:
        epsilon = 0
        alpha = 0
        gamma = 0

    state = env.reset()

    while True:

        if (ep % 100 == 0 and ep is not 0) or ep > 900:
            env.render()
        else:
            env.close()

        #  sample Q values from policy
        Q = policy.forward(torch.from_numpy(state).type(torch.FloatTensor))  # " Q(s, a) "

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            #  find optimal action according to policy (over axis -1)
            _, max = torch.max(Q, -1)
            action = max.item()

        new_state, reward, terminal, _ = env.step(action)

        total_reward += reward

        Q_new = policy.forward(torch.from_numpy(new_state).type(torch.FloatTensor))  # " find Q (s', a') "

        #  find optimal action Q value for next step
        new_max, _ = torch.max(Q_new, -1)  # " max(Q(s', a')) "

        Q_target = Q.clone()
        Q_target = Variable(Q_target.data)

        #  update target value function according to TD
        Q_target[action] = reward + torch.mul(new_max.detach(), gamma)  # " reward + gamma*(max(Q(s', a')) "

        # Calculate loss
        loss = loss_fn(Q, Q_target)  # " reward + gamma*(max(Q(s', a')) - Q(s, a)) "

        # Update original policy according to Q_target ( supervised learning )
        policy.zero_grad()
        loss.backward()
        optimizer.step()

        #  Q and Q_target should converge

        state = new_state

        if terminal:
            break

    reward_history.append(reward)
