# Reinforcement Learning Project

(README clean to do)

Project of RL (Q-learning) and Deep RL (DQN Algorithm) at EPFL.    
The idea was to implement a simple game (Tic Tac Toe) where we have an optimal agent serving as a reference in order to compare the performances (and learning speed) of different agents, in particular an agent provided with a Q-table and learning by Q-learning and another one, more advanced, provided with an Artificial Neural Network simulating a Q-table (named a Q-network) and learning due to the DQN algorithm.    
    
- tic_env.py: Contains the game environment.
- utils.py: Contains classes corresponding to the different Agents (inheritance used from an abstract Agent class), the Q-network and the Q-table.
- tic_tac_toe.ipynb: The implementation and results analysis.
