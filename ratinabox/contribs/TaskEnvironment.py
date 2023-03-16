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

class Objective():
    """
    Abstract `Objective` class that can be used to define finishing coditions
    for a task
    """
    def __init__(self, env:TaskEnvironment, *pos, **kws):
        self.env = env

    def check(self):
        """
        Check if the objective is satisfied
        """
        raise NotImplementedError("check() must be implemented")


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
    def __init__(self, Agents:List[Agent]|Agent, *pos, **kws):
        super().__init__(*pos, **kws)
        self.episode_history:List[pd.Series] = [] # Written to upon completion of an episode
        self.objectives:List[Objective] = [] # List of current objectives to saitsfy
        self.dynamic_walls = []      # List of walls that can change/move
        self.dynamic_objects = []    # List of current objects that can move
        self.Agents = Agents if isinstance(Agents, list) else [Agents]

    def add_agents(self, agents:Union[List[Agent], Agent]):
        if isinstance(agents, list):
            self.agents += agents
        elif isinstance(agents, Agent):
            self.agents.append(agents)
        else:
            raise TypeError("agents must be a list of agents or an agent type")

    def is_done(self):
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
        raise NotImplementedError("update() must be implemented")

    def step(self, *pos, **kws):
        """Alias to satisfy openai compatibility"""
        self.update(*pos, **kws)

    # ----------------------------------------------
    # Reading and writing episod data
    # ----------------------------------------------

    def write_episode(self, **kws):
        self.episode_history.append(pd.Series(kws))

    def read_episodes(self)->pd.DataFrame:
        pass


class SpatialGoalObjective(Objective):
    """
    Spatial goal objective: agent must reach a specific position

    Parameters
    ----------
    env : TaskEnvironment
        The environment that the objective is defined in
    goal_pos : np.ndarray | None
        The position that the agent must reach
    """
    def __init__(self, *pos, goal_pos:Union[np.ndarray,None], **kws):
        super().__init__(*pos, **kws)
        self.goal_pos = goal_pos

    def check(self, agents:List[Agent]):
        """
        Check if the objective is satisfied
        """
        agents_reached_goal = [(agent.pos == self.goal_pos).all(axis=1).any()
                               for agent in agents]
        return any(agents_reached_goal)

class SpatialGoalEnvironment(TaskEnvironment):
    """
    Creates a spatial goal-directed task

    Parameters
    ----------
    possible_goal_pos : List[np.ndarray] | np.ndarray
        List of possible goal positions
    current_goal_state : np.ndarray | None
        The current goal position
    n_goals : int
        The number of goals to set
    """

    def __init__(self, *pos, 
                 possible_goal_pos:Union[List[np.ndarray], np.ndarray]=[], 
                 current_goal_state:Union[NoneType,np.ndarray,List[np.ndarray]]=None, 
                 n_goals:int=1,
                 **kws):
        super().__init__(*pos, **kws)
        self.possible_goal_pos = possible_goal_pos
        self.n_goals = n_goals
        if current_goal_state is None:
            self.reset()

    def _propose_spatial_goal(self):
        """
        Propose a new spatial goal from the possible goal positions
        """
        if len(self.possible_goal_pos):
            goal_pos = np.random.choice(self.possible_goal_pos, 1)
        else:
            goal_pos = None # No goal state (this could be, e.g., a lockout time)
        return goal_pos

    def reset(self, n_goals=None):
        """
            reset

        resets the environement to a new episode
        """
        ng = n_goals if n_goals is not None else self.n_goals
        for _ in range(ng):
            self.goal_pos = self._propose_spatial_goal()
            objective = SpatialGoalObjective(self, goal_pos=self.goal_pos)
            self.objectives.append(objective)

    def is_done(self):
        """
        Whether the current state is a terminal state
        """
        # Check our objectives
        i_objective = 0
        # Loop through objectives, checking if they are satisfied
        while i_objective < len(self.objectives):
            if self.objectives[i_objective].check():
                self.objectives.pop(i_objective)
            else:
                i_objective += 1
        # Return if no objectives left
        no_objectives_left = len(self.objectives) == 0
        return no_objectives_left

    def update(self, *pos, **kws):
        """
        Update the environment
        """
        # Check if we are done
        if self.is_done():
            self.reset()

if __name__ == "__main__":
    pass
