# Environments that implement tasks
# -----# -----# -----# -----# ----------
#
# Key OpenAI Gym defines:
# (1) step()
# (2) reset()

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import gym
from gym.spaces import Box, Space, Dict

from types import NoneType
from typing import List, Union
import warnings

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

    # Inputs
    --------
    verbose : bool
        Whether to print out information about the environment
    render_mode : str
        How to render the environment. Options are 'matplotlib', 'pygame', or
        'none'
    render_every : int
        How often to render the environment (in time steps)
    """
    def __init__(self, *pos, verbose=False,
                 render_mode='matplotlib', render_every=2, **kws):
        super().__init__(*pos, **kws)
        self.episode_history:List[pd.Series] = [] # Written to upon completion of an episode
        self.objectives:List[Objective] = [] # List of current objectives to saitsfy
        self.dynamic_walls = []      # List of walls that can change/move
        self.dynamic_objects = []    # List of current objects that can move
        self.Agents:List[Agent] = [] # List of agents in the environment
        self.t = 0                   # Current time step
        self.render_every = render_every # How often to render
        self.verbose = verbose
        self.render_mode:str = render_mode # options 'matplotlib'|'pygame'|'none'
        self._stable_render_objects:dict = {} # objects that are stable across
                                         # a rendering type

        # ----------------------------------------------
        # Setup gym primatives
        # ----------------------------------------------
        # Setup observation space from the Environment space
        ext = [self.extent[i:i+2] for i in np.arange(0, len(self.extent), 2)]
        lows, highs = np.array(list(zip(*ext)), dtype=np.float_)
        self.observation_space:Space = \
                Box(low=lows, high=highs, dtype=np.float_)
        self.action_space:List[Space] = Dict({})
        self.reward_space:List[float] = []
        self.info:dict = {} # gynasiym returns an info dict in step()

    def add_agents(self, agents:Union[List[Agent], Agent],
                   names=None, maxvel:float=50.0, **kws):
        """
        Add agents to the environment

        For each agent, we add its action space (expressed as velocities it can
        take) to the environment's action space.

        Parameters
        ----------
        agents : List[Agent] | Agent
            The agents to add to the environment
        names : List[str] | None
            The names of the agents. If None, then the names are generated
        maxvel : float
            The maximum velocity that the agents can take
        """
        if not isinstance(agents, (list, Agent)):
            raise TypeError("agents must be a list of agents or an agent type")
        if isinstance(agents, Agent):
            agents = [agents]
        # Enlist agents
        if names is None:
            start = len(self.Agents)
            names = ["agent_" + str(start+i) for i in range(len(agents))]
        for (name, agent) in zip(names, agents):
            self.Agents.append(agent)
            # Add the agent's action space to the environment's action space
            D = int(self.dimensionality[0])
            self.action_space[name] = Box(low=0, high=maxvel, shape=(D,))
            self.reward_space.append(0.0)

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

    def update(self, update_agents=True):
        """
        How to update the task over time
        """
        self.t += 1 # base task class only has a clock
        # ---------------------------------------------------------------
        # Task environments in OpenAI's gym interface update their agents
        # ---------------------------------------------------------------
        if update_agents:
            for agent in self.Agents:
                agent.update()
        # ---------------------------------------------------------------

    def step(self, drift_velocity, dt=None, 
             drift_to_random_strength_ratio=1, *pos, **kws):
        """
            step()

        step() functions in Gynasium paradigm usually take an action space
        action, and return the next state, reward, whether the state is
        terminal, and an information dict

        different from update(), which updates this environment. this function
        executes a full step on the environment with an action from the agents
        """
        # Udpate the environment
        self.update(*pos, **kws)

        # If the user passed drift_velocity, update the agents
        drift_velocity = drift_velocity if \
                isinstance(drift_velocity, list) \
                else [drift_velocity]*len(self.Agents)
        for (agent, drift_velocity) in zip(self.Agents, drift_velocity):
            dt = dt if dt is not None else agent.dt
            agent.update(dt=dt, drift_velocity=drift_velocity,
                         drift_to_random_strength_ratio= \
                                 drift_to_random_strength_ratio)

        # Return the next state, reward, whether the state is terminal,
        return self.get_state(), self.reward, self.is_done(), self.info
        
    def get_state(self):
        """
        Get the current state of the environment
        """
        return [agent.pos for agent in self.Agents]

    # ----------------------------------------------
    # Reading and writing episod data
    # ----------------------------------------------

    def write_episode(self, **kws):
        self.episode_history.append(pd.Series(kws))

    def read_episodes(self)->pd.DataFrame:
        pass

class Objective():
    """
    Abstract `Objective` class that can be used to define finishing coditions
    for a task
    """
    def __init__(self, env:TaskEnvironment):
        self.env = env

    def check(self):
        """
        Check if the objective is satisfied
        """
        raise NotImplementedError("check() must be implemented")

    def __call__(self):
        """
        Can be used to report its value to the environment
        (Not required -- just a convenience)
        """
        raise NotImplementedError("__call__() must be implemented")

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
    def __init__(self, *pos, goal_pos:Union[np.ndarray,None], 
                 goal_radius=None, **kws):
        super().__init__(*pos, **kws)
        if self.env.verbose:
            print("new SpatialGoalObjective: goal_pos = {}".format(goal_pos))
        self.goal_pos = goal_pos
        self.radius = self.env.dx * 10 if goal_radius is None else goal_radius
    
    def _in_goal_radius(self, pos, goal_pos):
        """
        Check if a position is within a radius of a goal position
        """
        radius = self.radius
        return np.linalg.norm(pos - goal_pos, axis=1) < radius

    def check(self, agents:List[Agent]):
        """
        Check if the objective is satisfied

        Parameters
        ----------
        agents : List[Agent]
            The agents to check the objective for (usually just one)
        """
        agents_reached_goal = [
            self._in_goal_radius(agent.pos, self.goal_pos).all().any()
                               for agent in agents]
        return any(agents_reached_goal)

    def __call__(self)->np.ndarray:
        """
        Can be used to report its value to the environment
        (Not required -- just a convenience)
        """
        return self.goal_pos

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
                 possible_goal_pos:List[np.ndarray]|np.ndarray|str='random_4', 
                 current_goal_state:Union[NoneType,np.ndarray,List[np.ndarray]]=None, 
                 n_goals:int=1,
                 **kws):
        super().__init__(*pos, **kws)
        self.possible_goal_pos = self._init_poss_goals(possible_goal_pos)
        self.n_goals = n_goals
        self.objectives:List[SpatialGoalObjective] = []
        if current_goal_state is None:
            self.reset()

    def _init_poss_goals(self, possible_goal_pos:List[np.ndarray]|np.ndarray|str):
        """ 
        Initialize the possible goal positions 


        Parameters
        ----------
        possible_goal_pos : List[np.ndarray] | np.ndarray | str
            List of possible goal positions or a string to generate random
            goal positions

        Returns
        -------
        possible_goal_pos : np.ndarray
        """

        if isinstance(possible_goal_pos, str):
            if possible_goal_pos.startswith('random'):
                n = int(possible_goal_pos.split('_')[1])
                ext = [self.extent[i:i+2] for i in np.arange(0, len(self.extent), 2)]
                possible_goal_pos = [np.random.random(n) * \
                                     (ext[i][1] - ext[i][0]) + ext[i][0]
                                     for i in range(len(ext))]
                possible_goal_pos = np.array(possible_goal_pos).T
            else:
                raise ValueError("possible_goal_pos string must start with "
                                 "'random'")
        else:
            raise ValueError("possible_goal_pos must be a list of np.ndarrays, "
                         "a string, or None")
        return possible_goal_pos

    def _propose_spatial_goal(self):
        """
        Propose a new spatial goal from the possible goal positions
        """
        if len(self.possible_goal_pos):
            g = np.random.choice(np.arange(len(self.possible_goal_pos)), 1)
            goal_pos = self.possible_goal_pos[g]
        else:
            warnings.warn("No possible goal positions specified yet")
            goal_pos = None # No goal state (this could be, e.g., a lockout time)
        return goal_pos

    def reset(self, goal_locations:np.ndarray|None=None, n_goals=None):
        """
            reset

        resets the environement to a new episode
        """
        # How many goals to set?
        if goal_locations is not None:
            ng = len(goal_locations)
        elif n_goals is not None:
            ng = n_goals
        else:
            ng = self.n_goals
        self.objectives = [] # blank slate
        # Set the number of required spatial spatial goals
        for g in range(ng):
            if goal_locations is None:
                self.goal_pos = self._propose_spatial_goal()
            else:
                self.goal_pos = goal_locations[g]
            objective = SpatialGoalObjective(self, 
                                             goal_pos=self.goal_pos)
            self.objectives.append(objective)
        # Clear rendering cache
        self.clear_render_cache()

    def render(self, render_mode=None, *pos, **kws):
        """
        Render the environment
        """
        if render_mode is None:
            render_mode = self.render_mode
        if self.verbose:
            print("rendering environment with mode: {}".format(render_mode))
        if render_mode == 'matplotlib':
            self._render_matplotlib(*pos, **kws)
        elif render_mode == 'pygame':
            self._render_pygame(*pos, **kws)
        elif render_mode == 'none':
            pass
        else:
            raise ValueError("method must be 'matplotlib' or 'pygame'")

    def _render_matplotlib(self, *pos, **kws):
        """
        Render the environment using matplotlib
        """
        ag_annotate_default={'fontsize':10}
        ag_scatter_default={'marker':'o'}
        sg_scatter_default={'marker':'x', 'c':'r'}

        if np.mod(self.t, self.render_every) != 0:
            # Skip rendering unless this is redraw time
            return None

        if "matplotlib" not in self._stable_render_objects:
            R = self._stable_render_objects["matplotlib"] = {}
        else:
            R = self._stable_render_objects["matplotlib"]

        if "fig" not in R:
            fig, ax = plt.subplots(1,1)
            R["fig"] = fig
            R["ax"] = ax
        else:
            fig, ax = R["fig"], R["ax"]

        if "environment" not in R:
            R["environment"] = self.plot_environment(fig=fig, ax=ax)
            R["title"] = fig.suptitle("t={}".format(self.t))
        else:
            R["title"].set_text("t={}".format(self.t))

        if "agents" not in R:
            R["agents"] = []
            R["agent_history"] = []
            for agent in self.Agents:
                # set 🐀 location
                pos = agent.pos
                poshist = np.vstack((
                    np.reshape(agent.history["pos"],(-1,2)),
                    np.atleast_2d(pos)))
                if not len(agent.history['pos']):
                    his = plt.plot(*poshist.T, 'k', linewidth=0.2,
                                   linestyle='dotted')
                R["agent_history"].append(his)
                ag = plt.scatter(*pos.T, **ag_scatter_default)
                R["agents"].append(ag)
        else:
            for i, agent in enumerate(self.Agents):
                scat = R["agents"][i]
                scat.set_offsets(agent.pos)
                his = R["agent_history"][i]
                his[0].set_data(*np.array(agent.history["pos"]).T)

        if "spat_goals" not in R:
            R["spat_goals"] = []
            R["spat_goal_radius"] = []
            for spat_goal in self.objectives:                
                sg = plt.scatter(*spat_goal().T, **sg_scatter_default)
                ci = plt.Circle(spat_goal().ravel(), spat_goal.radius,
                                facecolor="red", alpha=0.2)
                ax.add_patch(ci)
                R["spat_goals"].append(sg)
                R["spat_goal_radius"].append(sg)
        else:
            for i, obj in enumerate(self.objectives):
                scat = R["spat_goals"][i]
                scat.set_offsets(obj())
                ci = R["spat_goal_radius"][i]
                # TODO blitting circles?
                # ci.set_center(obj().ravel())
                # ci.set_radius(obj.radius)

    def _render_pygame(self, *pos, **kws):
        pass

    def clear_render_cache(self):
        """
            clear_render_cache
        
        clears the cache of objects held for render()
        """
        if "matplotlib" in self._stable_render_objects:
            R = self._stable_render_objects["matplotlib"]
            R["ax"].cla()
            for item in (set(R.keys()) - set(("fig","ax"))):
                R.pop(item)

    def is_done(self):
        """
        Whether the current state is a terminal state
        """
        # Check our objectives
        i_objective = 0
        # Loop through objectives, checking if they are satisfied
        while i_objective < len(self.objectives):
            if self.objectives[i_objective].check(self.Agents):
                self.objectives.pop(i_objective)
            else:
                i_objective += 1
        # Return if no objectives left
        no_objectives_left = len(self.objectives) == 0
        return no_objectives_left

    def update(self, *pos, **kws):
        """
        Update the environment

        if drift_velocity is passed, update the agents
        """
        # Update the environment
        super().update(*pos, **kws)

        # Check if we are done
        if self.is_done():
            self.reset()

active = True
if active and __name__ == "__main__":

    plt.close('all')
    env = SpatialGoalEnvironment(n_goals=1, params={'dimensionality':'2D'},
                                 render_every=100,
                                 verbose=False)
    Ag = Agent(env)
    env.add_agents(Ag)
    # Note: if Agent() class detects if its env is a TaskEnvironment,
    # we could have the agent's code automate this last step. In other words,
    # after setting the agent's environment, it could automatically add itself
    # to the environment's list of agents.
    env.reset()

    plt.ion()
    env.render()
    plt.show()
    plt.pause(0.1)
    
    while True:
        env.update()
        env.render()
        Ag.update()
        plt.pause(0.1)
        if env.is_done():
            print("goal reached!")
            break
    
    # env.render()
