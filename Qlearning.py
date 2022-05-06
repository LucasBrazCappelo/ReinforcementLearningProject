import numpy as np
import random

class QlearningAgent:
    '''
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
    '''
    
    def __init__(self, epsilon=0.2, player='X', learningRate=0.05, discountFactor=0.99):
        self.epsilon = epsilon
        self.player = player # 'X' or 'O'
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.Q = {} # use get(key, default)

    def set_player(self, player = 'X', j=-1):
        self.player = player
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def empty(self, grid):
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i/3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def randomMove(self, grid):
        """ Chose a random move from the available options. """
        avail = self.empty(grid)

        return avail[random.randint(0, len(avail)-1)]

    def train(self, grid, **kwargs):
        """
        Goes through a hierarchy of moves, making the best move that
        is currently available each time (with probabitity 1-self.epsilon).
        A touple is returned that represents (row, col).
        """
        # whether move in random or not
        if random.random() < self.epsilon:
            return self.randomMove(grid)

        # random move
        return self.randomMove(grid)