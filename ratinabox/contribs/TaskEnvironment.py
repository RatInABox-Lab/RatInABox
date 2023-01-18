# Environments that implement tasks
# -----# -----# -----# -----# ----------

from ratinabox.Environment import Environment
import numpy as np
import pandas as pd
import gym
from types import NoneType
from typing import List, Union

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
        pass

    def write_episode(**kws):
        self.episode_history.append(pd.Series(kws))

    def episodes_to_df():
        pass

class SpatialGoalEnvironment(TaskEnvironment):
    """
    Creates a spatial goal-directed task
    """

    def __init__(self, *pos, 
                 possible_goal_locations:Union[List[np.ndarray], np.ndarray]=[], 
                 current_goal_state:Union[NoneType,np.ndarray,List[np.ndarray]]=None, 
                 **kws):
        super().__init__(*pos, **kws)
        self.possible_goal_locations = possible_goal_locations
        if current_goal_state is None:
            self.reset()
        else:
            self.goal_state = current_goal_state

    def reset(self):
        """
            reset

        resets the environement
        """
        if len(self.possible_goal_locations):
            self.goal_state = np.random.choice(self.possible_goal_locations, 1)
        else:
            self.goal_state = [] # No goal state (this could be, e.g., a lockout time)
        self.terminal_state_reached = False

