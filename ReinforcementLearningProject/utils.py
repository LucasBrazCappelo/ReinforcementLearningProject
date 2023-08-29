import copy
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from tic_env import OptimalPlayer, TictactoeEnv
from tqdm.notebook import trange


class Agent(ABC):
    def __init__(
        self,
        epsilon: Union[float, Tuple[float, float]] = 0.2,
        player: str = "X",
        learning_rate: float = 0.05,
        discount_factor: float = 0.99,
        n_max: int = 100,
    ):
        self.epsilon: float
        self.epsilon_min: float
        self.epsilon_max: float
        if isinstance(epsilon, tuple):
            self.epsilon_min, self.epsilon_max = epsilon
            self.epsilon = self.epsilon_max
        else:
            self.epsilon = epsilon
            self.epsilon_min = epsilon
            self.epsilon_max = epsilon
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor

        self.state: Optional[np.ndarray] = None
        self.action: Optional[Tuple[int, int]] = None

        self.n: int = 0
        self.n_max: int = n_max

        self.isLearning: bool = True

        self.player: str = player  # 'X' or 'O'

    def decrease_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - self.n / self.n_max))

    def set_player(self, player: str = "X", j: int = -1) -> None:
        self.player = player
        if j != -1:
            self.player = "X" if j % 2 == 0 else "O"

    @staticmethod
    def empty(state: np.ndarray) -> List[Tuple[int, int]]:
        """Return all empty positions."""
        available_actions: List[Tuple[int, int]] = []
        for x in range(3):
            for y in range(3):
                position: Tuple[int, int] = (x, y)
                if state[position] == 0:
                    available_actions.append(position)
        return available_actions

    def random_action(self, state: np.ndarray) -> Tuple[int, int]:
        """Choose a random action from the available options."""
        available_actions: List[Tuple[int, int]] = self.empty(state)

        return random.choice(available_actions)

    @abstractmethod
    def best_action(self, state: np.ndarray) -> Tuple[int, int]:
        pass

    @abstractmethod
    def act(self, state: np.ndarray) -> Tuple[int, int]:
        pass

    @abstractmethod
    def learn(self, s_prime: np.ndarray, reward: float, end: bool = False) -> None:
        pass


class Qtable:
    def __init__(self, q_tab: Optional[Dict[int, float]] = None, default_value: float = 0.0):
        if q_tab is None:
            q_tab = {}
        self.q_tab: Dict[int, float] = q_tab
        self.default_value: float = default_value

    @staticmethod
    def hash(state_action: Tuple[np.ndarray, Tuple[int, int]]) -> int:
        """Hash a state-action tuple to a unique integer.

        Each element of the grid (state) has 3 states, same for the action position. Then, we
        consider the state-action pair as a base 3 number, and we hash it to an integer. Can be
        easily recover from hash to state-action pair.
        """
        state, action = state_action

        hash_state: int = ((state.flatten() + 1) @ np.array([3**i for i in range(9)])).item()
        hash_action: int = action[0] * 3**9 + action[1] * 3**10
        return hash_state + hash_action

    @staticmethod
    def reverse_hash(h: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        state_reverse: np.ndarray = np.zeros((3, 3))
        for i in range(9):
            state_reverse[i // 3, i % 3] = h // 3**i % 3
        state_reverse -= 1
        action_reverse = (h // 3**9 % 3, h // 3**10 % 3)
        return state_reverse, action_reverse

    def __getitem__(self, key: Tuple[np.ndarray, Tuple[int, int]]) -> float:
        return self.q_tab.get(self.hash(key), self.default_value)

    def __setitem__(self, key: Tuple[np.ndarray, Tuple[int, int]], value: float) -> None:
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
        epsilon: float = 0.2,
        player: str = "X",
        learning_rate: float = 0.05,
        discount_factor: float = 0.99,
        n_max: int = 100,
        q: Qtable = Qtable(),
    ):
        Agent.__init__(self, epsilon, player, learning_rate, discount_factor, n_max)

        self.q: Qtable = q

    def best_action(self, state: np.ndarray) -> Tuple[int, int]:
        """Choose the available actions which have a maximum expected future reward.

        If there are multiple actions with the same maximum expected future reward, choose one of
        them at random.
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

    def act(self, state: np.ndarray) -> Tuple[int, int]:
        """Epsilon-greedy action selection, according to the Q-table."""
        self.state = state

        # whether move in random or not
        if random.random() < self.epsilon:
            self.action = self.random_action(state)
        else:
            # Get the best move
            self.action = self.best_action(state)

        return self.action

    def learn(self, s_prime: np.ndarray, reward: float, end: bool = False) -> None:
        """Q-learning update.

        If it's the end of a game, we set Q(s',a') = 0.
        """
        if self.isLearning:
            self.state = cast(np.ndarray, self.state)
            self.action = cast(Tuple[int, int], self.action)
            if not end:
                # Get the best move
                a_prime: Tuple[int, int] = self.best_action(s_prime)

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
    def __init__(
        self,
        input_size: int = 18,
        hidden_size1: int = 128,
        hidden_size2: int = 128,
        output_size: int = 9,
    ):
        super().__init__()
        self.flattener = nn.Flatten()
        self.inputLayer = nn.Linear(input_size, hidden_size1)
        self.fullyConnected = nn.Linear(hidden_size1, hidden_size2)
        self.outputLayer = nn.Linear(hidden_size2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flattener(x)
        x = nn_functional.relu(self.inputLayer(x))
        x = nn_functional.relu(self.fullyConnected(x))
        x = self.outputLayer(x)
        return x


class DQNAgent(Agent):
    """Our Q-network will be a simple linear neural network with two hidden layers."""

    def __init__(
        self,
        epsilon: float = 0.2,
        player: str = "X",
        learning_rate: float = 0.0005,
        discount_factor: float = 1.0,
        n_max: int = 100,
        q_model: nn.Module = QNetwork(),
        batch_size: int = 64,
        c: int = 500,
        r: deque = deque(maxlen=10_000),
        criterion: nn.Module = nn.HuberLoss(),
        second_player: bool = False,
    ):
        super().__init__(epsilon, player, learning_rate, discount_factor, n_max)

        # If a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_model: nn.Module = q_model.to(self.device)
        self.q_target: nn.Module = copy.deepcopy(q_model).to(self.device)

        # self.r is a deque of tuples of the form
        # Tuple[
        #     self.state: torch.Tensor,
        #     self.action: Tuple[int, int],
        #     reward: float,
        #     s_prime: torch.Tensor
        # ]
        self.r: deque = r
        self.batch_size: int = min(batch_size, cast(int, r.maxlen))

        self.t: int = 0
        self.c: int = c

        self.criterion: nn.Module = criterion.to(self.device)

        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=learning_rate)

        self.loss_curve: List[float] = []

        self.second_player: bool = second_player

    def best_action(self, state: torch.Tensor) -> Tuple[int, int]:
        """Choose the available actions which have a maximum expected future reward using the
        Q-network."""
        # convert state to tensor, adding batch dimension
        with torch.no_grad():
            q_values = self.q_model.forward(state)
        action: int = q_values.argmax(dim=1).item()
        return action // 3, action % 3

    def grid_to_state(self, grid: np.ndarray) -> torch.Tensor:
        state = torch.tensor(grid, dtype=torch.int64)
        state = nn_functional.one_hot(state + 1, 3)
        if not self.second_player:
            state = state[:, :, (2, 0)]
        else:
            state = state[:, :, (0, 2)]
        state = state.unsqueeze(0)
        state = state.type(torch.float).to(self.device)
        return state

    def act(self, grid: np.ndarray) -> Tuple[int, int]:
        """Epsilon-greedy action selection, according to the Q-table."""
        self.state: torch.Tensor = self.grid_to_state(grid)

        # whether move in random or not
        if random.random() < self.epsilon:
            self.action = self.random_action(grid)
        else:
            # Get the best move
            self.action = self.best_action(self.state)

        return self.action

    def learn(self, grid: np.ndarray, reward: float, end: bool = False) -> None:
        if self.isLearning:
            if not end:
                s_prime: torch.Tensor = self.grid_to_state(grid)

                self.r.append((self.state, self.action, reward, s_prime))
            else:
                self.r.append((self.state, self.action, reward, None))

                self.state = None
                self.action = None

                self.n += 1
                self.decrease_epsilon()
            # self.r is a deque with max length equal to buffer_size so it auto pop

            batch: deque
            max_q_target: torch.Tensor
            if len(self.r) < self.batch_size:
                batch = self.r
                max_q_target = torch.zeros(len(self.r)).to(self.device)
            else:
                # sample random minibatch from self.r
                batch = cast(deque, random.sample(self.r, self.batch_size))
                max_q_target = torch.zeros(self.batch_size).to(self.device)

            # convert to tensor
            states: torch.Tensor = torch.cat([x[0] for x in batch]).to(self.device)
            actions: List[Tuple[int, int]] = [x[1] for x in batch]
            rewards: torch.Tensor = torch.tensor([x[2] for x in batch], dtype=torch.float).to(
                self.device
            )

            self.optimizer.zero_grad()

            q_theta_sj_aj: torch.float = self.q_model.forward(states)[
                torch.arange(len(actions)), actions
            ]

            s_prime_mask: torch.Tensor = torch.tensor(
                [x[3] is not None for x in batch], dtype=torch.bool
            ).to(self.device)
            if s_prime_mask.any():
                s_primes: torch.Tensor = torch.cat([x[3] for x in batch if x[3] is not None]).to(
                    self.device
                )

                max_q_target[s_prime_mask] = (
                    self.q_target.forward(s_primes).max(dim=1).values.detach()
                )
            loss: torch.Tensor = self.criterion(
                q_theta_sj_aj, rewards + self.discount_factor * max_q_target
            )

            loss.backward()
            self.optimizer.step()

            self.loss_curve.append(loss.item())

            self.t += 1
            if self.t == self.c:
                self.t = 0
                self.q_target.load_state_dict(self.q_model.state_dict())

        elif end:
            self.state = None
            self.action = None


def play_games(
    player_opt: OptimalPlayer,
    agent: Agent,
    max_games: int = 20_000,
    env: TictactoeEnv = TictactoeEnv(),
) -> np.ndarray:
    turns: np.ndarray = np.array(["X", "O"])
    winner_list: np.ndarray = np.zeros(max_games)

    p_bar = trange(max_games)
    for nbGames in range(max_games):
        env.reset()
        grid: np.ndarray
        grid, _, __ = env.observe()

        player_opt.player = turns[nbGames % 2]
        agent.player = turns[(nbGames + 1) % 2]

        for roundGame in range(9):
            move: Tuple[int, int]
            if env.current_player == player_opt.player:
                if roundGame > 1 and isinstance(player_opt, Agent):
                    player_opt.learn(grid, 0)
                move = player_opt.act(grid)
            else:
                if roundGame > 1 and isinstance(agent, Agent):
                    agent.learn(grid, 0)
                move = agent.act(grid)

            bad_move: bool = env.grid[move] != 0
            if bad_move:
                env.end = True
                end: bool = True
                env.num_step += 1
                env.current_player = "X" if env.num_step % 2 == 0 else "O"
                winner: str = env.current_player
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


def play_games_with_m(
    player_opt: OptimalPlayer,
    agent: Agent,
    max_games_total: int = 20_000,
    delta_m: int = 250,
    env: TictactoeEnv = TictactoeEnv(),
) -> Tuple[np.ndarray, List[float], List[float]]:
    m_opt: List[float] = []
    m_random: List[float] = []
    player_real_opt = OptimalPlayer(epsilon=0.0)
    player_random = OptimalPlayer(epsilon=1.0)
    winner_list: List[np.ndarray] = []

    p_bar = trange(max_games_total)
    for _ in range(max_games_total // delta_m):
        winner_list.append(play_games(player_opt, agent, delta_m, env=env))

        current_epsilon = agent.epsilon
        agent.epsilon = 0
        agent.isLearning = False

        m_opt.append(play_games(player_real_opt, agent, 500, env=env).mean().item())
        m_random.append(play_games(player_random, agent, 500, env=env).mean().item())

        agent.isLearning = True
        agent.epsilon = current_epsilon

        p_bar.update(delta_m)
    return np.concatenate(winner_list), m_opt, m_random
