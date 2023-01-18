# Environments that implement tasks
# -----# -----# -----# -----# ----------
#
# Key OpenAI Gym defines:
# (1) step()
# (2) reset()

import pandas as pd
import gym

import numpy as np
from types import NoneType
from typing import List, Union

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

class TaskEnvironment(Environment, gym.Env):
    """
    Environment with task structure: there is an objective, and when the
    objective is reached, it terminates an episode, and starts a new episode
    (reset). This environment can be static or dynamic, depending on whether
    update() is implemented.

    In order to be more useful with other Reinforcement Learning pacakges, this
    environment inherits from both ratinabox.Environment and openai's widely
    used gym environment.
    """
    def __init__(self, *pos, **kws):
        super().__init__(*pos, **kws)
        self.episode_history:List[pd.Series] = [] # Written to upon completion of an episode
        self.dynamic_walls = []      # List of walls that can change
        self.dynamic_objects = []    # List of current objects
        self.dynamic_objectives = [] # List of current objectives to saitsfy

    def add_agents(self, agents:Union[List[Agent], Agent]):
        if isinstance(agents, list):
            self.agents += agents
        elif isinstance(agents, Agent):
            self.agents.append(agents)
        else:
            raise TypeError("agents must be a list of agents or an agent type")

    def finshed(self):
        """
        Whether the current state is a terminal state
        """
        raise NotImplementedError("is_done() must be implemented")

    def reset(self):
        """
        How to reset the task when finisedh
        """
        raise NotImplementedError("reset() must be implemented")

    def update(self):
        """
        If the environment is dynamic, use this
        """
        pass

    def step(self, *pos, **kws):
        """Alias to satisfy openai"""
        self.update(*pos, **kws)

    # ----------------------------------------------
    # Reading and writing episod data
    # ----------------------------------------------

    def write_episode(self, **kws):
        self.episode_history.append(pd.Series(kws))

    def read_episodes(self)->pd.DataFrame:
        pass

class SpatialGoalEnvironment(TaskEnvironment):
    """
    Creates a spatial goal-directed task
    """

    def __init__(self, *pos, 
                 possible_goal_pos:Union[List[np.ndarray], np.ndarray]=[], 
                 current_goal_state:Union[NoneType,np.ndarray,List[np.ndarray]]=None, 
                 **kws):
        super().__init__(*pos, **kws)
        self.possible_goal_pos = possible_goal_pos
        if current_goal_state is None:
            self.reset()

    def reset(self):
        """
            reset

        resets the environement
        """
        if len(self.possible_goal_pos):
            self.goal_pos = np.random.choice(self.possible_goal_pos, 1)
        else:
            self.goal_pos = [] # No goal state (this could be, e.g., a lockout time)
        self.dynamic_objectives.append(self.one_agent_reached_target)
        self.terminal_state_reached = False

    def one_agent_reached_target(self):
        """
        any agent reaches one of the goal positions
        """
        agents_reached_goal = [(pos == self.goal_pos).all(axis=1).any() for Agent in self.agents]
        return any(agents_reached_goal)

    def finished(self):
        # Check our objectives
        checked_position = 0
        while check_position < len(self.dynamic_objectives):
            if self.dynamic_objectives[checked_position]():
                self.dynamic_objectives.pop(checked_position)
            else:
                checked_position += 1
        # Return if no objectives left
        no_objectives_left = len(self.dynamic_objectives) == 0
        return no_objectives_left


if __name__ == "__main__":
    pass
