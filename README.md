# Reinforcement Learning Project
### basic README. Will be improved.

RL (Q-learning) and Deep RL (*DQN Algorithm*) project at EPFL for the ANN course.
The idea was to implement a simple game (Tic Tac Toe) where we have an optimal agent serving as a reference in order to compare the performances (and learning speed) of different agents, in particular an agent provided with a Q-table and learning by Q-learning and another one, more advanced, provided with an Artificial Neural Network simulating a Q-table (named a Q-network) and learning due to the DQN algorithm.

Algorithms:
- Q-learning
- **DQN**

The results obtained are satisfactory and the DQN used could easily be adapted to be trained on more complex games or systems.

- tic_env.py: Contains the game environment.
- utils.py: Contains classes corresponding to the different Agents (inheritance used from an abstract Agent class), the Q-network and the Q-table.
- tic_tac_toe.ipynb: The implementation and results analysis.
