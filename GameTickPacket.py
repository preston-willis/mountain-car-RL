import numpy as np

class GameData:
    def __init__(self):
        self.game_tick_packet = GameTickPacket()
        self.episode_packet = EpisodePacket()

    def reset(self):
        self.episode_packet.update(self.game_tick_packet)
        self.calc_ep_reward(self.episode_packet.episodes[len(self.episode_packet.episodes) - 1])
        self.game_tick_packet.count = 0
        self.game_tick_packet.end = []
        self.game_tick_packet.state = []
        self.game_tick_packet.episode += 1

    def calc_ep_reward(self, data):
        max = -np.inf
        for i in data.state_history:
            if i.state_action[0]+12 > max:
                max = i.state_action[0]
        data.reward = max+17


class EpisodePacket:
    def __init__(self):
        self.episodes = []

    def update(self, data):
        self.episodes.append(Episode(data))


class Episode:
    def __init__(self, data):
        self.state_history = data.state
        self.reward = 0


class GameTickPacket:
    def __init__(self):
        self.state = []
        self.end = []
        self.count = 0
        self.reward = 2
        self.unique_state_indices = {}
        self.frame_continuous = 0
        self.episode = 0
        self.gametime = [[None for _ in range(200)] for _ in range(9999)]

    def update(self, tick, action):
        state, reward, end, info = tick

        state[0] = state[0].round(1) * 10
        state[1] = state[1].round(2) * 100

        self.state.append(State(state, action))
        self.end.append(end)

        if self.is_used(state, action) is None:
            self.unique_state_indices.update({self.frame_continuous: self.state[self.count].state_action})
            self.gametime[self.episode][self.count] = self.frame_continuous

    def is_used(self, state, action):
        for i in self.unique_state_indices.values():
            if i == [state[0], state[1], action]:
                return True

    def tick(self):
        self.count += 1
        self.frame_continuous += 1


class State:
    def __init__(self, s, a):
        self.state_action = [s[0], s[1], a]
        self.value = 0
