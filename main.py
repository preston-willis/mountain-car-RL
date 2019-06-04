import gym
import random
import numpy as np
from GameTickPacket import GameData
from Plotter import Plotter


class Trainer:

    def __init__(self):
        self.plotter = Plotter()
        self.state_indices = [[]] * 855
        self.epsilon = 2
        self.ep = 0
        self.env = gym.make("MountainCar-v0")
        self.max_states = 855
        self.game_data = GameData()
        self.tick = self.game_data.game_tick_packet
        self.episodes = self.game_data.episode_packet.episodes
        self.policy = [[None, None, None] for _ in range(self.max_states)]
        self.min_error = 0
        self.alpha = 1
        self.epsilon_decay = 0.999
        self.max = 10000
        self.avg_reward = []


    def main(self):
        while self.ep < self.max:
            state = self.env.reset()
            state[0] = state[0].round(1) * 10
            state[1] = state[1].round(2) * 100

            while True not in self.tick.end:
                if self.ep % 100 == 0:
                    self.env.render()
                else:
                    self.env.close()

                action = None
                if self.policy is not None:
                    for i in self.policy:
                        if i[0] == state[0] and i[1] == state[1]:
                            action = i[2]

                if action is None:
                    action = self.env.action_space.sample()
                    print("yay "+str(self.ep))
                self.tick.update(self.env.step(action), action)
                state = self.tick.state[self.tick.count].state_action[:2]
                self.tick.tick()

            self.game_data.reset()

            q = self.calc_q(self.tick.unique_state_indices, self.episodes[len(self.episodes) - 1].reward)
            temp = self.greedy(self.tick.unique_state_indices, q)

            for i, p in enumerate(temp):
                self.policy[i] = p
                self.state_indices[i] = p[:2]

            if self.ep % 100 == 0:
                print("----------")
                print("[*] Policy at episode "+str(self.ep))
                for i, p in enumerate(temp):
                    print(self.policy[i])

            if self.ep % 100 == 0:
                self.plotter.show(self.avg_reward)

            #self.epsilon *= self.epsilon_decay
            self.ep += 1

    def calc_q(self, unique_states, episode_reward):
        values = []
        # print()
        # print("----------------- ENCOUNTERED STATES -----------------")
        # print()

        for count, (i, p) in enumerate(zip(unique_states, unique_states.values())):

            frame = None
            episode = None

            for c in range(len(self.tick.gametime)):
                try:
                    episode = c
                    frame = self.tick.gametime[c].index(i)
                    break
                except Exception:
                    pass

            if self.episodes[episode].state_history[frame].state_action in [i.state_action for i in self.episodes[len(self.episodes)-1].state_history]:
                self.episodes[episode].state_history[frame].value += self.episodes[len(self.episodes)-1].reward

            values.append(self.episodes[episode].state_history[frame].value)
            #print(str(count) + " -> " + str(p) + " -> " + str(self.episodes[episode].state_history[frame].value))

        self.avg_reward.append(self.episodes[len(self.episodes)-1].reward)

        return values

    def greedy(self, unique_states, q):
        raw_state = [i for i in unique_states.values()]
        state_groups = [[]]
        for u, p in enumerate(raw_state):
            group = []
            for i, x in enumerate(raw_state):
                if x[0] == p[0] and x[1] == p[1] and i not in (item for sublist in state_groups for item in sublist):
                    group.append(i)
            state_groups.append(group)

        state_groups = [x for x in state_groups if x]

        maximums = []

        for p in range(len(state_groups)):
            arr = []
            for i in range(len(state_groups[p])):
                arr.append(q[state_groups[p][i]])
            maximums.append(arr.index(np.max(arr)))

        # print()
        # print("----------------- OPTIMAL STATE ACTIONS -----------------")
        # print()

        updated_policy = []

        for p in range(len(state_groups)):
            # print(raw_state[state_groups[p][maximums[p]]])
            updated_policy.append(raw_state[state_groups[p][maximums[p]]])
        if random.uniform(0, 1) > self.epsilon:
            print('epsilon')
            updated_policy[random.randint(0, len(updated_policy)-1)][2] = self.env.action_space.sample()

        return updated_policy


t = Trainer()
t.main()
