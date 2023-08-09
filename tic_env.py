import random
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np


class TictactoeEnv:
    """
    Description:
        Classical Tic-tac-toe game for two players who take turns marking the spaces in a
        three-by-three grid with X or O.
        The player who succeeds in placing three of their marks in a horizontal, vertical, or
        diagonal row is the winner.

        The game is played by two players: player 'X' and player 'O'. Player 'x' moves first.

        The grid is represented by a 3x3 numpy array, with value in {0, 1, -1}, with corresponding
        values:
            0 - place unmarked
            1 - place marked with X
            -1 - place marked with O

        The game environment will receive movement from two players in turn and update the grid.

    self.step:
        receive the movement of the player, update the grid

    The action space is [0-8], representing the 9 positions on the grid.

    The reward is 1 if you win the game, -1 if you lose, and 0 besides.
    """

    def __init__(self) -> None:
        self.grid: np.ndarray = np.zeros((3, 3))
        self.end: bool = False
        self.winner: Optional[str] = None
        self.player2value: Dict[str, int] = {"X": 1, "O": -1}
        self.num_step: int = 0
        self.current_player: str = "X"  # By default, player 'X' goes first

    def _position_to_tuple(self, position: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if self.end:
            raise ValueError("This game has ended, please reset it!")
        if type(position) is int:
            position = (int(position / 3), position % 3)
        elif type(position) is not tuple:
            position = cast(Tuple[int, int], tuple(cast(List[int], position)))

        return position

    def check_valid(self, position: Union[int, Tuple[int, int]]) -> bool:
        """Check whether the current action is valid or not"""

        return False if self.grid[self._position_to_tuple(position)] != 0 else True

    def step(
        self, position: Union[int, Tuple[int, int]], print_grid: bool = False
    ) -> Tuple[np.ndarray, bool, Optional[str]]:
        """Receive the movement from two players in turn and update the grid"""
        # check the position and value are valid or not
        # position should be a tuple like (0, 1) or int [0-8]
        position = self._position_to_tuple(position)

        if self.grid[position] != 0:
            raise ValueError("There is already a chess on position {}.".format(position))

        # place a chess on the position
        self.grid[position] = self.player2value[self.current_player]
        # update
        self.num_step += 1
        self.current_player = "X" if self.num_step % 2 == 0 else "O"
        # check whether the game ends or not
        self.check_end()

        if print_grid:
            self.render()

        return self.grid.copy(), self.end, self.winner

    def get_current_player(self) -> str:
        return self.current_player

    def check_end(self) -> None:
        # check rows and cols
        if np.any(np.sum(self.grid, axis=0) == 3) or np.any(np.sum(self.grid, axis=1) == 3):
            self.end = True
            self.winner = "X"
        elif np.any(np.sum(self.grid, axis=0) == -3) or np.any(np.sum(self.grid, axis=1) == -3):
            self.end = True
            self.winner = "O"
        # check diagonals
        elif (
            self.grid[[0, 1, 2], [0, 1, 2]].sum() == 3 or self.grid[[0, 1, 2], [2, 1, 0]].sum() == 3
        ):
            self.end = True
            self.winner = "X"
        elif (
            self.grid[[0, 1, 2], [0, 1, 2]].sum() == -3
            or self.grid[[0, 1, 2], [2, 1, 0]].sum() == -3
        ):
            self.end = True
            self.winner = "O"
        # check if all the positions are filled
        elif np.sum(self.grid == 0) == 0:
            self.end = True
            self.winner = None  # no one wins
        else:
            self.end = False
            self.winner = None

    def reset(self) -> Tuple[np.ndarray, bool, Optional[str]]:
        # reset the grid
        self.grid = np.zeros((3, 3))
        self.end = False
        self.winner = None
        self.num_step = 0
        self.current_player = "X"

        return self.grid.copy(), self.end, self.winner

    def observe(self) -> Tuple[np.ndarray, bool, Optional[str]]:
        return self.grid.copy(), self.end, self.winner

    def reward(self, player: str = "X") -> int:
        if self.end:
            if self.winner is None:
                return 0
            else:
                return 1 if player == self.winner else -1
        else:
            return 0

    def render(self) -> None:
        # print current grid
        value2player: Dict[int, str] = {0: "-", 1: "X", -1: "O"}
        for i in range(3):
            print("|", end="")
            for j in range(3):
                print(value2player[int(self.grid[i, j])], end=" " if j < 2 else "")
            print("|")
        print()


class OptimalPlayer:
    """
    Description:
        A class to implement an epsilon-greedy optimal player in Tic-tac-toe.

    About optimal policy:
        There exists an optimal policy for game Tic-tac-toe. A player ('X' or 'O') can win or at
        least draw with optimal strategy.
        See the wikipedia page for details https://en.wikipedia.org/wiki/Tic-tac-toe
        In short, an optimal player choose the first available move from the following list:
            [Win, block_win, Fork, block_fork, Center, Corner, Side]

    Parameters:
        epsilon: float, in [0, 1]. This is a value between 0-1 that indicates the
            probability of making a random action instead of the optimal action
            at any given time.

    """

    def __init__(self, epsilon: float = 0.2, player: str = "X") -> None:
        self.epsilon: float = epsilon
        self.player: str = player  # 'x' or 'O'

    def set_player(self, player: str = "X", j: int = -1) -> None:
        self.player = player
        if j != -1:
            self.player = "X" if j % 2 == 0 else "O"

    @staticmethod
    def empty(grid: np.ndarray) -> List[Tuple[int, int]]:
        """return all empty positions"""
        avail: List[Tuple[int, int]] = []
        for i in range(9):
            pos = (int(i / 3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def center(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Pick the center if its available,
        if it's the first step of the game, center or corner are all optimal.
        """
        if np.abs(grid).sum() == 0:
            # first step of the game
            return [(1, 1)] + self.corner(grid)

        return [(1, 1)] if grid[1, 1] == 0 else []

    @staticmethod
    def corner(grid: np.ndarray) -> List[Tuple[int, int]]:
        """Pick empty corners to move"""
        corner: List[Tuple[int, int]] = [(0, 0), (0, 2), (2, 0), (2, 2)]
        cn: List[Tuple[int, int]] = []
        # First, pick opposite corner of opponent if it's available
        for i in range(4):
            if grid[corner[i]] == 0 and grid[corner[3 - i]] != 0:
                cn.append(corner[i])
        if len(cn) > 0:
            return cn
        else:
            for idx in corner:
                if grid[idx] == 0:
                    cn.append(idx)
            return cn

    @staticmethod
    def side(grid: np.ndarray) -> List[Tuple[int, int]]:
        """Pick empty sides to move"""
        rt: List[Tuple[int, int]] = []
        for idx in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            if grid[idx] == 0:
                rt.append(idx)
        return rt

    def win(self, grid: np.ndarray, val: Optional[int] = None) -> List[Tuple[int, int]]:
        """Pick all positions that player will win after taking it"""
        if val is None:
            val = 1 if self.player == "X" else -1

        to_win: List[Tuple[int, int]] = []
        # check all positions
        for pos in self.empty(grid):
            grid_ = np.copy(grid)
            grid_[pos] = val
            if self.check_win(grid_, val):
                to_win.append(pos)

        return to_win

    def block_win(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Find the win positions of opponent and block it"""
        oppon_val: int = -1 if self.player == "X" else 1
        return self.win(grid, oppon_val)

    def fork(self, grid: np.ndarray, val: Optional[int] = None) -> List[Tuple[int, int]]:
        """Find a fork opportunity that the player will have two positions to win"""
        if val is None:
            val = 1 if self.player == "X" else -1

        to_fork: List[Tuple[int, int]] = []
        # check all positions
        for pos in self.empty(grid):
            grid_: np.ndarray = np.copy(grid)
            grid_[pos] = val
            if self.check_fork(grid_, val):
                to_fork.append(pos)

        return to_fork

    def block_fork(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Block the opponent's fork.
        If there is only one possible fork from opponent, block it.
        Otherwise, player should force opponent to block win by making two in a row or column
        Among all possible force win positions, choose positions in opponent's fork in prior
        """
        oppon_val: int = -1 if self.player == "X" else 1
        oppon_fork: List[Tuple[int, int]] = self.fork(grid, oppon_val)
        if len(oppon_fork) <= 1:
            return oppon_fork

        # force the opponent to block win
        force_block_win: List[Tuple[int, int]] = []
        val: int = 1 if self.player == "X" else -1
        for pos in self.empty(grid):
            grid_: np.ndarray = np.copy(grid)
            grid_[pos] = val
            if np.any(np.sum(grid_, axis=0) == val * 2) or np.any(np.sum(grid_, axis=1) == val * 2):
                force_block_win.append(pos)
        force_block_win_prior: List[Tuple[int, int]] = []
        for pos in force_block_win:
            if pos in oppon_fork:
                force_block_win_prior.append(pos)

        return force_block_win_prior if force_block_win_prior != [] else force_block_win

    def check_win(self, grid: np.ndarray, val: Optional[int] = None) -> bool:
        # check whether the player corresponding to the val will win
        if val is None:
            val = 1 if self.player == "X" else -1
        target: int = 3 * val
        # check rows and cols
        if np.any(np.sum(grid, axis=0) == target) or np.any(np.sum(grid, axis=1) == target):
            return True
        # check diagonals
        elif (
            grid[[0, 1, 2], [0, 1, 2]].sum() == target or grid[[0, 1, 2], [2, 1, 0]].sum() == target
        ):
            return True
        else:
            return False

    def check_fork(self, grid: np.ndarray, val: Optional[int] = None) -> bool:
        # check whether the player corresponding to the val will fork
        if val is None:
            val = 1 if self.player == "X" else -1
        target: int = 2 * val
        # check rows and cols
        rows: int = np.sum(np.sum(grid, axis=0) == target).item()
        cols: int = np.sum(np.sum(grid, axis=1) == target).item()
        diags: int = (grid[[0, 1, 2], [0, 1, 2]].sum() == target) + (
            grid[[0, 1, 2], [2, 1, 0]].sum() == target
        )
        if (rows + cols + diags) >= 2:
            return True
        else:
            return False

    def random_move(self, grid: np.ndarray) -> Tuple[int, int]:
        """Chose a random move from the available options."""
        avail: List[Tuple[int, int]] = self.empty(grid)

        return avail[random.randint(0, len(avail) - 1)]

    def act(self, grid: np.ndarray) -> Tuple[int, int]:
        """
        Goes through a hierarchy of moves, making the best move that
        is currently available each time (with probability 1-self.epsilon).
        A tuple is returned that represents (row, col).
        """
        # whether move in random or not
        if random.random() < self.epsilon:
            return self.random_move(grid)

        # --- optimal policies ---

        # Win
        win: List[Tuple[int, int]] = self.win(grid)
        if len(win) > 0:
            return win[random.randint(0, len(win) - 1)]
        # Block win
        block_win: List[Tuple[int, int]] = self.block_win(grid)
        if len(block_win) > 0:
            return block_win[random.randint(0, len(block_win) - 1)]
        # Fork
        fork: List[Tuple[int, int]] = self.fork(grid)
        if len(fork) > 0:
            return fork[random.randint(0, len(fork) - 1)]
        # Block fork
        block_fork: List[Tuple[int, int]] = self.block_fork(grid)
        if len(block_fork) > 0:
            return block_fork[random.randint(0, len(block_fork) - 1)]
        # Center
        center: List[Tuple[int, int]] = self.center(grid)
        if len(center) > 0:
            return center[random.randint(0, len(center) - 1)]
        # Corner
        corner: List[Tuple[int, int]] = self.corner(grid)
        if len(corner) > 0:
            return corner[random.randint(0, len(corner) - 1)]
        # Side
        side: List[Tuple[int, int]] = self.side(grid)
        if len(side) > 0:
            return side[random.randint(0, len(side) - 1)]

        # random move
        return self.random_move(grid)
