import numpy as np

import random

from tic_env import TictactoeEnv, OptimalPlayer

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from collections import deque

from tqdm.notebook import trange

def play_games(player_opt, agent, maxGames=20_000, env=TictactoeEnv()):
    Turns = np.array(['X','O'])
    winnerList = np.zeros(maxGames)

    pBar = trange(maxGames)
    for nbGames in range(maxGames):
        env.reset()
        grid, _, __ = env.observe()

        player_opt.player = Turns[nbGames%2]
        agent.player = Turns[(nbGames+1)%2]

        for roundGame in range(9):
            if env.current_player == player_opt.player:
                if roundGame > 1 and isinstance(player_opt, Agent):
                    player_opt.learn(grid, 0)
                move = player_opt.act(grid)
            else:
                if roundGame > 1 and isinstance(agent, Agent):
                    agent.learn(grid, 0)
                move = agent.act(grid)

            badMove = env.grid[move] != 0  
            if badMove:
                env.end = True
                end = True
                env.num_step += 1
                env.current_player = 'X' if env.num_step % 2 == 0 else  'O'
                winner = env.current_player
            else:
                grid, end, winner = env.step(move, print_grid=False)

            if end:
                if winner == agent.player:
                    winnerList[nbGames] = 1
                    if isinstance(player_opt, Agent):
                        player_opt.learn(grid, -1, end=True)
                    if isinstance(agent, Agent):
                        agent.learn(grid, 1, end=True)
                elif winner == player_opt.player:
                    winnerList[nbGames] = -1
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
        pBar.update(1)
    pBar.close()
    env.reset()
    return winnerList

def play_games_with_M(player_opt, agent, maxGamesTotal=20_000, deltaM=250, env=TictactoeEnv()):
    Mopt = []
    Mrandom = []
    player_realOpt = OptimalPlayer(epsilon=0.0)
    player_random = OptimalPlayer(epsilon=1.0)
    winnerList = []

    pBar = trange(maxGamesTotal)
    for _ in range(maxGamesTotal//deltaM):
        winnerList.append(play_games(player_opt, agent, deltaM))

        currentEpsilon = agent.epsilon
        agent.epsilon = 0
        agent.isLearning = False

        Mopt.append(play_games(player_realOpt, agent, 500).mean())
        Mrandom.append(play_games(player_random, agent, 500).mean())

        agent.isLearning = True
        agent.epsilon = currentEpsilon

        pBar.update(deltaM)
    winnerList = np.concatenate(winnerList)
    return winnerList, Mopt, Mrandom

class Agent:
    def __init__(self, epsilon=0.2, player='X', learningRate=0.05, discountFactor=0.99, n_max=100):
        if isinstance(epsilon, tuple):
            self.epsilon_min, self.epsilon_max = epsilon
            self.epsilon = self.epsilon_max
        else:
            self.epsilon = epsilon
            self.epsilon_min = epsilon
            self.epsilon_max = epsilon
        self.learningRate = learningRate
        self.discountFactor = discountFactor

        self.state = None
        self.action = None

        self.n = 0
        self.n_max = n_max

        self.isLearning = True

        self.player = player # 'X' or 'O'

    def decrease_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - self.n / self.n_max))

    def set_player(self, player = 'X', j=-1):
        self.player = player
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def empty(self, state):
        """ Return all empty positions. """
        availableActions = []
        for x in range(3):
            for y in range(3):
                position = (x, y)
                if state[position] == 0:
                    availableActions.append(position)
        return availableActions

    def randomAction(self, state):
        """ Choose a random action from the available options. """
        availableActions = self.empty(state)

        return random.choice(availableActions)
    
    def bestAction(self, state):
        pass

    def act(self,state):
        pass

    def learn(self, s_prime, reward, end=False):
        pass

class Qtable:
    def __init__(self, Qtab={}, defaultValue=0.0):
        self.Qtab = Qtab
        self.defaultValue = defaultValue

    def hash(self, state_action):
        """
        hash a state-action tuple to a unique integer. Each element of the grid (state) has 3 states, 
        same for the action position.
        Then, we consider the state-action pair as a base 3 number, and we hash it to an integer.
        Can be easily recover from hash to state-action pair.
        """
        state, action = state_action

        hashState = (state.flatten()+1)@np.array([3**0,3**1,3**2,3**3,3**4,3**5,3**6,3**7,3**8])
        hashAction = action[0]*3**9 + action[1]*3**10
        return hashState + hashAction

    def reverseHash(self, h):
        state_reverse = np.zeros((3,3))
        state_reverse[0,0] = h//3**0%3
        state_reverse[0,1] = h//3**1%3
        state_reverse[0,2] = h//3**2%3
        state_reverse[1,0] = h//3**3%3
        state_reverse[1,1] = h//3**4%3
        state_reverse[1,2] = h//3**5%3
        state_reverse[2,0] = h//3**6%3
        state_reverse[2,1] = h//3**7%3
        state_reverse[2,2] = h//3**8%3
        state_reverse -= 1
        action_reverse = [None, None]
        action_reverse[0] = h//3**9%3
        action_reverse[1] = h//3**10%3
        action_reverse = tuple(action_reverse)
        return state_reverse, action_reverse
    
    def __getitem__(self, key: tuple):
        return self.Qtab.get(self.hash(key), self.defaultValue)

    def __setitem__(self, key: tuple, value):
        self.Qtab[self.hash(key)] = value

class QlearningAgent(Agent):
    """
    Description:
        A class to implement an epsilon-greedy Qlearning Agent in Tic-tac-toe.

    Parameters:
        epsilon: float, in [0, 1]. This is a value between 0-1 that indicates the
            probability of making a random action instead of the greedy action
            at any given time.

        learningRate: float, in [0, 1]. Setting it to 0 means that the Q-values are
            never updated, hence nothing is learned. Setting a high value such as 0.9 
            means that learning can occur quickly. 
        
        discountFactor: float, in [0, 1]. This models the fact that future rewards are
            worth less than immediate rewards. Setting it to 0 means that the agent
            will only learn from immediate rewards. Setting it to 1 means that the
            agent will learn from all rewards equally.
    """

    def __init__(self, epsilon=0.2, player='X', learningRate=0.05, discountFactor=0.99, n_max=100, Q=Qtable()):
        Agent.__init__(self, epsilon, player, learningRate, discountFactor, n_max)

        self.Q = Q

    def bestAction(self, state):
        """
        Choose the available actions which have a maximum expected future reward. 
        If there are multiple actions with the same maximum expected future reward,
        choose one of them at random.
        """
        # Get the available moves
        availableActions = self.empty(state)

        # Get the best move
        bestActions = []
        bestValue = -999.0
        for action in availableActions:
            Qsa = self.Q[state, action]
            if Qsa == bestValue:
                bestActions.append(action)
            if Qsa > bestValue:
                bestActions = [action]
                bestValue = Qsa

        return random.choice(bestActions)

    def act(self, state):
        """
        epsilon-greedy action selection, according to the Q-table.
        """
        self.state = state

        # whether move in random or not
        if random.random() < self.epsilon:
            self.action = self.randomAction(state)
        else:
            # Get the best move
            self.action = self.bestAction(state)

        return self.action

    def learn(self, s_prime, reward, end=False):
        """
        Q-learning update. If it's the end of a game, we set Q(s',a') = 0.
        """
        if self.isLearning:
            if not end:
                # Get the best move
                a_prime = self.bestAction(s_prime)

                # Update the Q-value
                self.Q[self.state, self.action] += self.learningRate * (reward + self.discountFactor * self.Q[s_prime, a_prime] - self.Q[self.state, self.action])
            else:
                self.Q[self.state, self.action] += self.learningRate * (reward - self.Q[self.state, self.action])

                self.state = None
                self.action = None

                self.n += 1
                self.decrease_epsilon()
        elif end:
            self.state = None
            self.action = None

class Qnetwork(nn.Module):
    def __init__(self, input_size=18, hidden_size1=128, hidden_size2=128, output_size=9):
        super(Qnetwork, self).__init__()
        self.flattener = nn.Flatten()
        self.inputLayer = nn.Linear(input_size, hidden_size1)
        self.fullyConnected = nn.Linear(hidden_size1, hidden_size2)
        self.outputLayer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.flattener(x)
        x = F.relu(self.inputLayer(x))
        x = F.relu(self.fullyConnected(x))
        x = self.outputLayer(x)
        return x

class DQN_agent(Agent):
    """
    Our Q-network will be a simple linear neural network with two hidden layers.
    """
    def __init__(self, epsilon=0.2, player='X', learningRate=0.0005, discountFactor=1.0 , n_max=100, Qmodel=Qnetwork(), batch_size=64, C=500 , R=deque(maxlen=10_000), criterion=nn.HuberLoss(), secondPlayer = False):
        super(DQN_agent, self).__init__(epsilon, player, learningRate, discountFactor, n_max)

        # If a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Qmodel = Qmodel.to(self.device)
        self.Qtarget = copy.deepcopy(Qmodel).to(self.device)

        self.R = R
        self.batch_size = min(batch_size, R.maxlen)

        self.t = 0
        self.C = C
        
        self.criterion = criterion.to(self.device)

        self.optimizer = torch.optim.Adam(self.Qmodel.parameters(), lr=learningRate)

        self.lossCurve = []

        self.secondPlayer = secondPlayer

    def bestAction(self, state):
        """
        Choose the available actions which have a maximum expected future reward
        using the Q-network.
        """
        # convert state to tensor, adding batch dimension
        with torch.no_grad():
            q_values = self.Qmodel.forward(state)
        return q_values.argmax(dim=1).item()

    def act(self, grid):
        """
        epsilon-greedy action selection, according to the Q-table.
        """
        state = torch.tensor(grid, dtype=torch.int64)
        state = F.one_hot(state+1,3)
        if not self.secondPlayer:
            state = state[:,:,(2,0)]
        else:
            state = state[:,:,(0,2)]
        state = state.unsqueeze(0)
        state = state.type(torch.float).to(self.device)
        self.state = state

        # whether move in random or not
        if random.random() < self.epsilon:
            action = self.randomAction(grid)
            self.action = action[0] * 3 + action[1]
        else:
            # Get the best move
            self.action = self.bestAction(self.state)
            # action is a tuple of (x, y) from self.action
            action = (self.action // 3, self.action % 3)


        return action

    def learn(self, grid, reward, end=False):
        if self.isLearning:
            if not end:
                s_prime = torch.tensor(grid, dtype=torch.int64)
                s_prime = F.one_hot(s_prime+1,3)

                if not self.secondPlayer:
                    s_prime = s_prime[:,:,(2,0)]
                else:
                    s_prime = s_prime[:,:,(0,2)]
                s_prime = s_prime.unsqueeze(0)
                s_prime = s_prime.type(torch.float).to(self.device)

                self.R.append((self.state, self.action, reward, s_prime))
            else:
                self.R.append((self.state, self.action, reward, None))

                self.state = None
                self.action = None

                self.n += 1
                self.decrease_epsilon()
            # self.R is a deque with maxlen=buffer_size so it auto pop

            if len(self.R) < self.batch_size:
                batch = self.R
                maxQtarget = torch.zeros(len(self.R)).to(self.device)
            else:
                # sample random minibatch from self.R
                batch = random.sample(self.R, self.batch_size)
                maxQtarget = torch.zeros(self.batch_size).to(self.device) 

            # convert to tensor
            states = torch.cat([x[0] for x in batch]).to(self.device)
            actions = [x[1] for x in batch]
            rewards = torch.tensor([x[2] for x in batch], dtype=torch.float).to(self.device)

            self.optimizer.zero_grad()

            Q_theta_sj_aj = self.Qmodel.forward(states)[torch.arange(len(actions)),actions]

            s_prime_mask = torch.tensor([x[3] is not None for x in batch], dtype=torch.bool).to(self.device)
            if s_prime_mask.any():
                s_primes = torch.cat([x[3] for x in batch if x[3] is not None]).to(self.device)
                
                maxQtarget[s_prime_mask] = self.Qtarget.forward(s_primes).max(dim=1).values.detach()
            loss = self.criterion(Q_theta_sj_aj, rewards + self.discountFactor*maxQtarget)

            loss.backward()
            self.optimizer.step()

            self.lossCurve.append(loss.item())

            self.t += 1
            if self.t == self.C:
                self.t = 0
                self.Qtarget.load_state_dict(self.Qmodel.state_dict())
        
        elif end:
            self.state = None
            self.action = None