import copy
import random
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from tqdm.notebook import trange

from tic_env import OptimalPlayer, TictactoeEnv


def play_games(player_opt, agent, max_games=20_000, env=TictactoeEnv()):
    turns = np.array(["X", "O"])
    winner_list = np.zeros(max_games)

    p_bar = trange(max_games)
    for nbGames in range(max_games):
        env.reset()
        grid, _, __ = env.observe()

        player_opt.player = turns[nbGames % 2]
        agent.player = turns[(nbGames + 1) % 2]

        for roundGame in range(9):
            if env.current_player == player_opt.player:
                if roundGame > 1 and isinstance(player_opt, Agent):
                    player_opt.learn(grid, 0)
                move = player_opt.act(grid)
            else:
                if roundGame > 1 and isinstance(agent, Agent):
                    agent.learn(grid, 0)
                move = agent.act(grid)

            bad_move = env.grid[move] != 0
            if bad_move:
                env.end = True
                end = True
                env.num_step += 1
                env.current_player = "X" if env.num_step % 2 == 0 else "O"
                winner = env.current_player
            else:
                grid, end, winner = env.step(move, print_grid=False)

            if end:
                if winner == agent.player:
                    winner_list[nbGames] = 1
                    if isinstance(player_opt, Agent):
                        player_opt.learn(grid, -1, end=True)
                    if isinstance(agent, Agent):
                        agent.learn(grid, 1, end=True)
                elif winner == player_opt.player:
                    winner_list[nbGames] = -1
                    if isinstance(player_opt, Agent):
                        player_opt.learn(grid, 1, end=True)
                    if isinstance(agent, Agent):
                        agent.learn(grid, -1, end=True)
                else:
                    if isinstance(player_opt, Agent):
                        player_opt.learn(grid, 0, end=True)
                    if isinstance(agent, Agent):
                        agent.learn(grid, 0, end=True)
                break
        p_bar.update(1)
    p_bar.close()
    env.reset()
    return winner_list


def play_games_with_m(player_opt, agent, max_games_total=20_000, delta_m=250, env=TictactoeEnv()):
    m_opt = []
    m_random = []
    player_real_opt = OptimalPlayer(epsilon=0.0)
    player_random = OptimalPlayer(epsilon=1.0)
    winner_list = []

    p_bar = trange(max_games_total)
    for _ in range(max_games_total // delta_m):
        winner_list.append(play_games(player_opt, agent, delta_m, env=env))

        current_epsilon = agent.epsilon
        agent.epsilon = 0
        agent.isLearning = False

        m_opt.append(play_games(player_real_opt, agent, 500, env=env).mean())
        m_random.append(play_games(player_random, agent, 500, env=env).mean())

        agent.isLearning = True
        agent.epsilon = current_epsilon

        p_bar.update(delta_m)
    winner_list = np.concatenate(winner_list)
    return winner_list, m_opt, m_random


class Agent(ABC):
    def __init__(
        self, epsilon=0.2, player="X", learning_rate=0.05, discount_factor=0.99, n_max=100
    ):
        if isinstance(epsilon, tuple):
            self.epsilon_min, self.epsilon_max = epsilon
            self.epsilon = self.epsilon_max
        else:
            self.epsilon = epsilon
            self.epsilon_min = epsilon
            self.epsilon_max = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.state = None
        self.action = None

        self.n = 0
        self.n_max = n_max

        self.isLearning = True

        self.player = player  # 'X' or 'O'

    def decrease_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - self.n / self.n_max))

    def set_player(self, player="X", j=-1):
        self.player = player
        if j != -1:
            self.player = "X" if j % 2 == 0 else "O"

    @staticmethod
    def empty(state):
        """Return all empty positions."""
        available_actions = []
        for x in range(3):
            for y in range(3):
                position = (x, y)
                if state[position] == 0:
                    available_actions.append(position)
        return available_actions

    def random_action(self, state):
        """Choose a random action from the available options."""
        available_actions = self.empty(state)

        return random.choice(available_actions)

    @abstractmethod
    def best_action(self, state):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self, s_prime, reward, end=False):
        pass


class Qtable:
    def __init__(self, q_tab=None, default_value=0.0):
        if q_tab is None:
            self.q_tab = {}
        self.q_tab = q_tab
        self.default_value = default_value

    @staticmethod
    def hash(state_action):
        """
        hash a state-action tuple to a unique integer. Each element of the grid (state) has 3
        states, same for the action position.
        Then, we consider the state-action pair as a base 3 number, and we hash it to an integer.
        Can be easily recover from hash to state-action pair.
        """
        state, action = state_action

        hash_state = (state.flatten() + 1) @ np.array(
            [3**0, 3**1, 3**2, 3**3, 3**4, 3**5, 3**6, 3**7, 3**8]
        )
        hash_action = action[0] * 3**9 + action[1] * 3**10
        return hash_state + hash_action

    @staticmethod
    def reverse_hash(h):
        state_reverse = np.zeros((3, 3))
        state_reverse[0, 0] = h // 3**0 % 3
        state_reverse[0, 1] = h // 3**1 % 3
        state_reverse[0, 2] = h // 3**2 % 3
        state_reverse[1, 0] = h // 3**3 % 3
        state_reverse[1, 1] = h // 3**4 % 3
        state_reverse[1, 2] = h // 3**5 % 3
        state_reverse[2, 0] = h // 3**6 % 3
        state_reverse[2, 1] = h // 3**7 % 3
        state_reverse[2, 2] = h // 3**8 % 3
        state_reverse -= 1
        action_reverse = [None, None]
        action_reverse[0] = h // 3**9 % 3
        action_reverse[1] = h // 3**10 % 3
        action_reverse = tuple(action_reverse)
        return state_reverse, action_reverse

    def __getitem__(self, key: tuple):
        return self.q_tab.get(self.hash(key), self.default_value)

    def __setitem__(self, key: tuple, value):
        self.q_tab[self.hash(key)] = value


class QlearningAgent(Agent):
    """
    Description:
        A class to implement an epsilon-greedy Qlearning Agent in Tic-tac-toe.

    Parameters:
        epsilon: float, in [0, 1]. This is a value between 0-1 that indicates the
            probability of making a random action instead of the greedy action
            at any given time.

        learning_rate: float, in [0, 1]. Setting it to 0 means that the Q-values are
            never updated, hence nothing is learned. Setting a high value such as 0.9
            means that learning can occur quickly.

        discount_factor: float, in [0, 1]. This models the fact that future rewards are
            worth less than immediate rewards. Setting it to 0 means that the agent
            will only learn from immediate rewards. Setting it to 1 means that the
            agent will learn from all rewards equally.
    """

    def __init__(
        self,
        epsilon=0.2,
        player="X",
        learning_rate=0.05,
        discount_factor=0.99,
        n_max=100,
        q=Qtable(),
    ):
        Agent.__init__(self, epsilon, player, learning_rate, discount_factor, n_max)

        self.q = q

    def best_action(self, state):
        """
        Choose the available actions which have a maximum expected future reward.
        If there are multiple actions with the same maximum expected future reward,
        choose one of them at random.
        """
        # Get the available moves
        available_actions = self.empty(state)

        # Get the best move
        best_actions = []
        best_value = -999.0
        for action in available_actions:
            q_sa = self.q[state, action]
            if q_sa == best_value:
                best_actions.append(action)
            if q_sa > best_value:
                best_actions = [action]
                best_value = q_sa

        return random.choice(best_actions)

    def act(self, state):
        """
        epsilon-greedy action selection, according to the Q-table.
        """
        self.state = state

        # whether move in random or not
        if random.random() < self.epsilon:
            self.action = self.random_action(state)
        else:
            # Get the best move
            self.action = self.best_action(state)

        return self.action

    def learn(self, s_prime, reward, end=False):
        """
        Q-learning update. If it's the end of a game, we set Q(s',a') = 0.
        """
        if self.isLearning:
            if not end:
                # Get the best move
                a_prime = self.best_action(s_prime)

                # Update the Q-value
                self.q[self.state, self.action] += self.learning_rate * (
                    reward
                    + self.discount_factor * self.q[s_prime, a_prime]
                    - self.q[self.state, self.action]
                )
            else:
                self.q[self.state, self.action] += self.learning_rate * (
                    reward - self.q[self.state, self.action]
                )

                self.state = None
                self.action = None

                self.n += 1
                self.decrease_epsilon()
        elif end:
            self.state = None
            self.action = None


class QNetwork(nn.Module):
    def __init__(self, input_size=18, hidden_size1=128, hidden_size2=128, output_size=9):
        super(QNetwork, self).__init__()
        self.flattener = nn.Flatten()
        self.inputLayer = nn.Linear(input_size, hidden_size1)
        self.fullyConnected = nn.Linear(hidden_size1, hidden_size2)
        self.outputLayer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.flattener(x)
        x = nn_functional.relu(self.inputLayer(x))
        x = nn_functional.relu(self.fullyConnected(x))
        x = self.outputLayer(x)
        return x


class DQNAgent(Agent):
    """
    Our Q-network will be a simple linear neural network with two hidden layers.
    """

    def __init__(
        self,
        epsilon=0.2,
        player="X",
        learning_rate=0.0005,
        discount_factor=1.0,
        n_max=100,
        q_model=QNetwork(),
        batch_size=64,
        c=500,
        r=deque(maxlen=10_000),
        criterion=nn.HuberLoss(),
        second_player=False,
    ):
        super(DQNAgent, self).__init__(epsilon, player, learning_rate, discount_factor, n_max)

        # If a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_model = q_model.to(self.device)
        self.q_target = copy.deepcopy(q_model).to(self.device)

        self.r = r
        self.batch_size = min(batch_size, r.maxlen)

        self.t = 0
        self.c = c

        self.criterion = criterion.to(self.device)

        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=learning_rate)

        self.lossCurve = []

        self.second_player = second_player

    def best_action(self, state):
        """
        Choose the available actions which have a maximum expected future reward
        using the Q-network.
        """
        # convert state to tensor, adding batch dimension
        with torch.no_grad():
            q_values = self.q_model.forward(state)
        return q_values.argmax(dim=1).item()

    def grid_to_state(self, grid):
        state = torch.tensor(grid, dtype=torch.int64)
        state = nn_functional.one_hot(state + 1, 3)
        if not self.second_player:
            state = state[:, :, (2, 0)]
        else:
            state = state[:, :, (0, 2)]
        state = state.unsqueeze(0)
        state = state.type(torch.float).to(self.device)
        return state

    def act(self, grid):
        """
        epsilon-greedy action selection, according to the Q-table.
        """
        self.state = self.grid_to_state(grid)

        # whether move in random or not
        if random.random() < self.epsilon:
            action = self.random_action(grid)
            self.action = action[0] * 3 + action[1]
        else:
            # Get the best move
            self.action = self.best_action(self.state)
            # action is a tuple of (x, y) from self.action
            action = (self.action // 3, self.action % 3)

        return action

    def learn(self, grid, reward, end=False):
        if self.isLearning:
            if not end:
                s_prime = self.grid_to_state(grid)

                self.r.append((self.state, self.action, reward, s_prime))
            else:
                self.r.append((self.state, self.action, reward, None))

                self.state = None
                self.action = None

                self.n += 1
                self.decrease_epsilon()
            # self.r is a deque with max length equal to buffer_size so it auto pop

            if len(self.r) < self.batch_size:
                batch = self.r
                max_q_target = torch.zeros(len(self.r)).to(self.device)
            else:
                # sample random minibatch from self.r
                batch = random.sample(self.r, self.batch_size)
                max_q_target = torch.zeros(self.batch_size).to(self.device)

            # convert to tensor
            states = torch.cat([x[0] for x in batch]).to(self.device)
            actions = [x[1] for x in batch]
            rewards = torch.tensor([x[2] for x in batch], dtype=torch.float).to(self.device)

            self.optimizer.zero_grad()

            q_theta_sj_aj = self.q_model.forward(states)[torch.arange(len(actions)), actions]

            s_prime_mask = torch.tensor([x[3] is not None for x in batch], dtype=torch.bool).to(
                self.device
            )
            if s_prime_mask.any():
                s_primes = torch.cat([x[3] for x in batch if x[3] is not None]).to(self.device)

                max_q_target[s_prime_mask] = (
                    self.q_target.forward(s_primes).max(dim=1).values.detach()
                )
            loss = self.criterion(q_theta_sj_aj, rewards + self.discount_factor * max_q_target)

            loss.backward()
            self.optimizer.step()

            self.lossCurve.append(loss.item())

            self.t += 1
            if self.t == self.c:
                self.t = 0
                self.q_target.load_state_dict(self.q_model.state_dict())

        elif end:
            self.state = None
            self.action = None
