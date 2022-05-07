import numpy as np
import random

class Qtable:
    def __init__(self, Qtab={}, defaultValue=0.0):
        self.Qtab = Qtab
        self.defaultValue = defaultValue

    def hash(self, state_action):
        """
        TODO
        """
        state, action = state_action

        hashState = (state.flatten()+1)@np.array([3**0,3**1,3**2,3**3,3**4,3**5,3**6,3**7,3**8])
        hashAction = action[0]*3**9 + action[1]*3**10
        return hashState + hashAction
    
    def __getitem__(self, key: tuple):
        return self.Qtab.get(self.hash(key), self.defaultValue)

    def __setitem__(self, key: tuple, value):
        self.Qtab[self.hash(key)] = value
        

class QlearningAgent:
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

    def __init__(self, epsilon=0.2, player='X', learningRate=0.05, discountFactor=0.99, Q={}, Q_defaultValue=0.0):
        if isinstance(epsilon, tuple):
            self.epsilon_min, self.epsilon_max = epsilon
            self.epsilon = self.epsilon_max
        else:
            self.epsilon = epsilon
            self.epsilon_min = epsilon
            self.epsilon_max = epsilon
        self.player = player # 'X' or 'O'
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.Q = Qtable(Q, Q_defaultValue)
        self.state = None
        self.action = None

    def decrease_epsilon(self, n, n_max):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - n / n_max))


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
            Qsa = self.Q[(state, action)]
            if Qsa == bestValue:
                bestActions.append(action)
            if Qsa > bestValue:
                bestActions = [action]
                bestValue = Qsa

        return random.choice(bestActions)

    def act(self, state):
        """
        TODO
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
        TODO
        """
        if not end:
            # Get the best move
            a_prime = self.bestAction(s_prime)

            # Update the Q-value
            self.Q[(self.state, self.action)] += self.learningRate * (reward + self.discountFactor * self.Q[(s_prime, a_prime)] - self.Q[(self.state, self.action)])
        else:
            self.Q[(self.state, self.action)] += self.learningRate * (reward - self.Q[(self.state, self.action)])

            self.state = None
            self.action = None