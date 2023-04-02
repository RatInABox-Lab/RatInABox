# Environments that implement tasks
# -----# -----# -----# -----# ----------
#
# Key OpenAI Gym defines:
# (1) step()
# (2) reset()

import numpy as np

import matplotlib.pyplot as plt

import pettingzoo
import gymnasium as gym
from gymnasium.spaces import Box, Space, Dict
# see https://gymnasium.farama.org/
# see https://

from types import NoneType, FunctionType
from typing import List, Union
from functools import partial
import warnings
from copy import copy

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

class Reward():
    """
    When an task objective is triggered, reward object is attached an Agent's
    reward:list. This object tracks the dynamics of the reward applied to the
    agent.

    This implementation allows rewards to be applied:
        - externally (through a task environment) 
        - or internally (through the agent's internal neuronal dynamics),
          e.g. through a set of neurons tracking rewards, attached to the agent

    This tracker specifies what the animals reward value should be at a given
    time while the reward is activate
    """
    decay_preset = {
        "constant":    lambda a, x: a,
        "linear":      lambda a, x: a * x,
        "exponential": lambda a, x: a * np.exp(x),
        "none":        lambda a, x: 0
    }
    decay_knobs_preset = {
        "linear":      [1],
        "constant":    [1],
        "exponential": [2],
        "none":        []
    }
    def __init__(self, init_state, dt,
                 expire_clock=None, decay=None,
                 decay_knobs=[], 
                 external_drive:Union[FunctionType,None]=None,
                 external_drive_strength=1):
        """
        Parameters
        ----------
        init_state : float
            initial reward value
        dt : float
            timestep
        expire_clock : float|None
            time until reward expires, if None, reward never expires
        decay : str|function|None
            decay function, or decay preset name, or None
        decay_knobs : list
            decay function knobs
        external_drive : function|None
            external drive function, or None. can be used to attach a goal
            gradient or reward ramping signal
        external_drive_strength : float
            strength of external drive, how quickly the reward follows the
            external drive
        """
        self.state = init_state if not isinstance(init_state, FunctionType) \
                               else init_state()
        self.dt = dt
        self.expire_clock = expire_clock if \
                isinstance(expire_clock, (int,float)) else \
                dt
        if isinstance(decay, str):
            self.preset = decay
            self.decay_knobs = decay_knobs or self.decay_knobs_preset[self.preset]
            self.decay = partial(self.decay_preset[self.preset], *self.decay_knobs)
        else:
            self.preset = "custom" if decay is not None \
                                   else "constant"
            self.decay_knobs = decay_knobs or \
                                        self.decay_knobs_preset[self.preset]
            self.decay = decay or self.decay_preset["constant"]
        self.external_drive = external_drive
        self.external_drive_strength = external_drive_strength
        self.history = {'state':[], 'expire_clock':[]}

    def update(self):
        """
        update reward, 

        grows towards the gradient target value from its initial value, if a
        target_value() function is defined. otherwise, reward is only
        controlled by decay from some initial value. if decay is 0, and target
        gradient is not defined then its constant, until the reward expire time
        is reached.

        # Returns
        True if reward is still active, False if reward has expired
        """
        self.state = self.state + self.get_delta() * self.dt
        self.expire_clock -= self.dt
        self.history['state'].append(self.state)
        self.history['expire_clock'].append(self.expire_clock)
        return not (self.expire_clock <= 0)

    def get_delta(self, state=None):
        """ \delta(reward) for a dt """
        state = self.state if state is None else state
        if self.external_drive is not None:
            target_gradient = self.external_drive()
            strength = self.external_drive_strength
            change = (strength*(target_gradient - state) - self.decay(state)) 
        else:
            change = -(self.decay(state)) 
        return change

    def plot_theoretical_reward(self, timerange=(0,1)):
        """
        plot the reward dynamics : shows the user how their parameters of
        interest setup reward dynamics, without updating the object
        """
        rewards = [self.state]
        timesteps = np.arange(timerange[0], timerange[1], self.dt)
        for t in timesteps[1:]:
            r = rewards[-1] + self.get_delta(state=rewards[-1]) * self.dt
            rewards.append(r)
        plt.plot(timesteps, rewards[:len(timesteps)],
               label=f"reward={self.preset}, " 
               f"knobs={self.decay_knobs}")
        plt.axvspan(self.expire_clock, plt.gca().get_ylim()[-1], color='r', alpha=0.2)
        plt.text(np.mean((plt.gca().get_xlim()[0], self.expire_clock)), 
                 np.mean(plt.gca().get_ylim()), "reward\nactive",
                 backgroundcolor='black', color="white")
        plt.text(np.mean((self.expire_clock, plt.gca().get_xlim()[-1])), 
                 np.mean(plt.gca().get_ylim()), "reward\nexpires",
                 backgroundcolor='black', color="white")
        return plt.gcf(), plt.gca()

class RewardCache():
    """
    RewardCache

    A cache of all `active` rewards attached to an agent
    """
    def __init__(self, verbose=False):
        self.cache:List[Reward] = []
        self.verbose = verbose

    def append(self, reward:Reward, copymode=True):
        if copymode:
            self.cache.append(copy(reward))
        else:
            self.cache.append(reward)

    def update(self):
        """
            Update
        """
        for reward in self.cache:
            reward_still_active = reward.update()
            if not reward_still_active:
                self.cache.remove(reward)
                if self.verbose:
                    print("Reward removed from cache")
    
    def get_total(self):
        """
        If there are any active rewards, return the sum of their values.
        """
        return sum([reward.state for reward in self.cache])


class TaskEnvironment(Environment, pettingzoo.ParallelEnv):
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
    *pos : list
        Positional arguments to pass to Environment
    verbose : bool
        Whether to print out information about the environment
    render_mode : str
        How to render the environment. Options are 'matplotlib', 'pygame', or
        'none'
    render_every : int
        How often to render the environment (in time steps)
    **kws :
        Keyword arguments to pass to Environment
    """
    metadata = {
        "render_modes": ["matplotlib", "none"],
        "name": "TaskEnvironment-RiAB"
    }

    def __init__(self, *pos, verbose=False,
                 render_mode='matplotlib', 
                 render_every=None, render_every_framestep=2,
                 dt=0.01, teleport_on_reset=False, 
                 save_expired_rewards=False, **kws):
        super().__init__(*pos, **kws)
        self.episode_history:dict = {} # Written to upon completion of an episode
        self.objectives:List[Objective] = [] # List of current objectives to saitsfy
        self.dynamic_walls = []      # List of walls that can change/move
        self.dynamic_objects = []    # List of current objects that can move
        self.Agents:List[Agent] = [] # List of agents in the environment
        self.t = 0                   # Current time
        self.dt = dt                 # Time step

        self.render_every = render_every_framestep # How often to render
        self.verbose = verbose
        self.render_mode:str = render_mode # options 'matplotlib'|'pygame'|'none'
        self._stable_render_objects:dict = {} # objects that are stable across
                                         # a rendering type

        # ----------------------------------------------
        # Agent-related task config
        # ----------------------------------------------
        self.teleport_on_reset = teleport_on_reset # Whether to teleport 
                                                   # agents to random

        # ----------------------------------------------
        # Setup gym primatives
        # ----------------------------------------------
        # Setup observation space from the Environment space
        self.observation_spaces:Dict[Space] = Dict({})
        self.action_spaces:Dict[Space]      = Dict({})
        self.reward_caches:List[RewardCache] = []
        self.agent_names:List[str]     = []
        self.info:dict                 = {} # gymnasium returns info in step()

        # Reward cache specifics
        self.save_expired_rewards = save_expired_rewards
        self.expired_rewards:List[RewardCache] = []

    def observation_space(self, agent_name:str):
        return self.observation_spaces[agent_name]

    def action_space(self, agent_name:str):
        return self.action_spaces[agent_name]

    def add_agents(self, agents:Union[dict, List[Agent], Agent],
                   names:None|list=None, maxvel:float=50.0, **kws):
        """
        Add agents to the environment

        For each agent, we add its action space (expressed as velocities it can
        take) to the environment's action space.

        Parameters
        ----------
        agents : Dict[Agent] | List[Agent] | Agent
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
        if not ([agent.dt == self.dt for agent in agents]):
            raise NotImplementedError("Does not yet support agents with different dt from envrionment")
        if isinstance(agents, dict):
            names  = list(agents.keys())
            agents = list(agents.values())
        elif names is None:
            start = len(self.Agents)
            names = ["agent_" + str(start+i) for i in range(len(agents))]
        # Enlist agents
        for (name, agent) in zip(names, agents):
            self.Agents.append(agent)
            self.agent_names.append(name)
            # Add the agent's action space to the environment's action spaces
            # dict
            D = int(self.dimensionality[0])
            self.action_spaces[name] = Box(low=0, high=maxvel, shape=(D,))
            # Add the agent's observation space to the environment's 
            # observation spaces dict
            ext = [self.extent[i:i+2] 
                   for i in np.arange(0, len(self.extent), 2)]
            lows, highs = np.array(list(zip(*ext)), dtype=np.float_)
            self.observation_spaces[name] = \
                    Box(low=lows, high=highs, dtype=np.float_)
            cache = RewardCache()
            self.reward_caches.append(cache)
            agent.reward = cache

    def _dict(self, V):
        """
        Convert a list of values to a dictionary of values keyed by agent name
        """
        return {name:v for (name, v) in zip(self.agent_names, V)} \
                if hasattr(V,'__iter__') else \
                {name:V for name in self.agent_names}

    def _is_terminal_state(self):
        """
        Whether the current state is a terminal state
        """
        raise NotImplementedError("_is_terminated() must be implemented")
    
    def _is_truncated_state(self):
        """ 
        whether the current state is a truncated state, 
        see https://gymnasium.farama.org/api/env/#gymnasium.Env.step

        default is false: an environment by default will have a terminal state,
        ending the episode, whereon users should call reset(), but not a
        trucation state ending the mdp.
        """
        return False

    def reset(self, seed=None, return_info=False, options=None):
        """
        How to reset the task when finisedh
        """
        # Clear rendering cache
        self.clear_render_cache()
        # If teleport on reset, randomly pick new location for agents
        if self.teleport_on_reset:
            for agent in self.Agents:
                agent.update()
                agent.pos = self.observation_spaces.sample()
                agent.history['pos'][-1] = agent.pos

    def update(self, update_agents=True):
        """
        How to update the task over time
        """
        self.t += self.dt # base task class only has a clock
        # ---------------------------------------------------------------
        # Task environments in OpenAI's gym interface update their agents
        # ---------------------------------------------------------------
        if update_agents:
            for agent in self.Agents:
                agent.update()
        # ---------------------------------------------------------------

    def step(self, actions:Union[dict,np.array]=None, dt=None, 
             drift_to_random_strength_ratio=1, *pos, **kws):
        """
            step()

        step() functions in Gynasium paradigm usually take an action space
        action, and return the next state, reward, whether the state is
        terminal, and an information dict

        different from update(), which updates this environment. this function
        executes a full step on the environment with an action from the agents

        https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.step
        """

        # If the user passed drift_velocity, update the agents
        actions = actions if isinstance(actions, dict) else \
                  self._dict(actions)
        for (agent, action) in zip(self.Agents, actions.values()):
            dt = dt if dt is not None else agent.dt
            action = np.array(action).ravel()
            agent.update(dt=dt, drift_velocity=action,
                         drift_to_random_strength_ratio= \
                                 drift_to_random_strength_ratio)

        # Update the reward caches for time decay of existing rewards
        for reward_cache in self.reward_caches:
            reward_cache.update()

        # Udpate the environment, which can add new rewards to caches
        self.update(*pos, **kws)
        
        # Return the next state, reward, whether the state is terminal,
        return (self.get_observation(), 
                self.get_reward(), 
                self._dict(self._is_terminal_state()),
                self._dict(self._is_truncated_state()), 
                self._dict([self.info])
                )

    def step1(self, action, *pos, **kws):
        """
        shortcut for stepping when only 1 agent exists...makes it behave
        like gymnasium instead of pettingzoo
        """
        results = self.step({self.agent_names[0]:action}, *pos, **kws)
        results = [x[self.agent_names[0]] for x in results]
        return results
        
    def get_observation(self):
        """ Get the current state of the environment """
        return {name:agent.pos 
                for name, agent in zip(self.agent_names, self.Agents)}

    def get_reward(self):
        """ Get the current reward state of each agent """
        return {name:agent.reward.get_total()
                for name, agent in zip(self.agent_names, self.Agents)}

    # ----------------------------------------------
    # Reading and writing episod data
    # ----------------------------------------------

    def write_episode(self, **kws):
        pass

    def read_episodes(self):
        pass

    # ----------------------------------------------
    # Rendering
    # ----------------------------------------------
    def render(self, render_mode=None, *pos, **kws):
        """
        Render the environment
        """
        if render_mode is None:
            render_mode = self.render_mode
        # if self.verbose:
        #     print("rendering environment with mode: {}".format(render_mode))
        if render_mode == 'matplotlib':
            self._render_matplotlib(*pos, **kws)
        elif render_mode == 'pygame':
            self._render_pygame(*pos, **kws)
        elif render_mode == 'none':
            pass
        else:
            raise ValueError("method must be 'matplotlib' or 'pygame'")

    def _render_matplotlib(self, *pos, agentkws:dict=dict(), **kws):
        """
        Render the environment using matplotlib
        `
        Inputs
        ------
        agentkws: dict
            keyword arguments to pass to the agent's render method
        """

        if np.mod(self.t, self.render_every) < self.dt:
            # Skip rendering unless this is redraw time
            return False

        R = self._get_mpl_render_cache()
    
        # Render the environment
        self._render_mpl_env()

        # Render the agents
        self._render_mpl_agents(**agentkws)

        return True

    def _get_mpl_render_cache(self):
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
        return R, fig, ax
    
    def _render_mpl_env(self):
        R, fig, ax = self._get_mpl_render_cache()
        if "environment" not in R:
            R["environment"] = self.plot_environment(fig=fig, ax=ax)
            R["title"] = fig.suptitle("t={:.2f}".format(self.t))
        else:
            R["title"].set_text("t={:.2f}".format(self.t))

    def _render_mpl_agents(self, **kws):
        """
        Render the agents

        TODO:
        -----
        - `skiprate`
        - plot_trajectory draws a new environment each time it plots,
          which slowly accumulates environment background objects. this
          can drag on performance after lots of episodes.
        """
        R, fig, ax = self._get_mpl_render_cache()
        initialize = "agents" not in R
        if initialize:
            R["agents"]        = []
            for (i, agent) in enumerate(self.Agents):
                # set 🐀 location
                _, a = agent.plot_trajectory(fig=fig, ax=ax)
                R["agents"].append(a.collections[-1])
        else:
            for i, agent in enumerate(self.Agents):
                scat = R["agents"][i]
                scat.set_offsets(agent.history['pos'])
                c, s = self._agent_style(agent, color=i, **kws)
                scat.set_facecolors(c)
                scat.set_edgecolors(c)
                scat.set_sizes(s)

    @staticmethod
    def _agent_style(agent:Agent, color=0,
                     point_size:bool=15, decay_point_size:bool=False, 
                     plot_agent:bool=True, decay_point_timescale:int=10):
        time = agent.history['t']
        if isinstance(color, int):
            color=plt.rcParams['axes.prop_cycle'].by_key()['color'][color]
        s = point_size * np.ones_like(time)
        if decay_point_size == True:
            s = point_size * np.exp((time - time[-1]) / decay_point_timescale)
            s[(time[-1] - time) > (1.5 * decay_point_timescale)] *= 0
        c = [color] * len(time)
        if plot_agent == True:
            s[-1] = 40
            c[-1] = "r"
        return c, s

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

    def close(self):
        """ gymnasium close() method """
        self.clear_render_cache()
        if "fig" in self._stable_render_objects:
            if isinstance(self._stable_render_objects["fig"], plt.Figure):
                plt.close(self._stable_render_objects["fig"])

reward_default=Reward(1, 0.01, expire_clock=1, decay="linear")

class Objective():
    """
    Abstract `Objective` class that can be used to define finishing coditions
    for a task
    """
    def __init__(self, env:TaskEnvironment, 
                 reward=reward_default):
        self.env = env
        self.reward = reward

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
    reward_value : float
        The reward value that the objective gives
    goal_pos : np.ndarray | None
        The position that the agent must reach
    goal_radius : float | None
        The radius around the goal position that the agent must reach
    """
    def __init__(self, *pos, reward=reward_default, goal_pos:Union[np.ndarray,None], 
                 goal_radius=None, **kws):
        super().__init__(*pos, **kws)
        if self.env.verbose:
            print("new SpatialGoalObjective: goal_pos = {}".format(goal_pos))
        self.goal_pos = goal_pos
        self.radius = np.min((self.env.dx * 10, np.ptp(self.env.extent)/10)) if goal_radius is None else goal_radius
    
    def _in_goal_radius(self, pos, goal_pos):
        """
        Check if a position is within a radius of a goal position
        """
        radius = self.radius
        distance = lambda x,y : self.env.get_distances_between___accounting_for_environment(x, y, wall_geometry="line_of_sight")
        # return np.linalg.norm(pos - goal_pos, axis=1) < radius \
        #         if np.ndim(goal_pos) > 1 \
        #         else np.abs((pos - goal_pos)) < radius
        return distance(pos, goal_pos) < radius 

    def check(self, agents:List[Agent]):
        """
        Check if the objective is satisfied

        Parameters
        ----------
        agents : List[Agent]
            The agents to check the objective for (usually just one)

        Returns
        -------
        rewards : List[float]
            The rewards for each agent
        which_agents : np.ndarray
            The indices of the agents that reached the goal
        """
        agents_reached_goal = [
            self._in_goal_radius(agent.pos, self.goal_pos).all().any()
                               for agent in agents]
        rewarded_agents = np.where(agents_reached_goal)[0]
        rewards = [self.reward] * len(rewarded_agents)
        if self.env.verbose:
            pass
            # print("SpatialGoalObjective.check(): ",
            #       "rewarded_agents = {}".format(rewarded_agents),
            #       "reward attach = {}".format(rewards))
        return rewards, rewarded_agents

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
    *pos : 
        Positional arguments for TaskEnvironment
            - n_agents : int
            - 
    possible_goal_pos : List[np.ndarray] | np.ndarray
        List of possible goal positions
    current_goal_state : np.ndarray | None
        The current goal position
    n_goals : int
        The number of goals to set
    """
    # --------------------------------------
    # Some reasonable default render settings
    # --------------------------------------
    ag_annotate_default={'fontsize':10}
    ag_scatter_default ={'marker':'o'}

    def __init__(self, *pos, 
                 reward=reward_default, objectivekws=dict(),
                 possible_goal_pos:List|np.ndarray|str='random_5', 
                 current_goal_state:Union[NoneType,np.ndarray,List[np.ndarray]]=None, 
                 n_goals:int=1,
                 **kws):
        super().__init__(*pos, **kws)
        self.possible_goal_pos = self._init_poss_goals(possible_goal_pos)
        self.n_goals = n_goals
        self.objectives:List[SpatialGoalObjective] = []
        self.objkws = dict()
        self.objkws.update({'reward':reward})
        self.objkws.update(objectivekws)
        if current_goal_state is None:
            self.reset()

    def _init_poss_goals(self, possible_goal_pos:List|np.ndarray|str):
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
                ext = [self.extent[i:i+2] 
                       for i in np.arange(0, len(self.extent), 2)]
                possible_goal_pos = [np.random.random(n) * \
                                     (ext[i][1] - ext[i][0]) + ext[i][0]
                                     for i in range(len(ext))]
                possible_goal_pos = np.array(possible_goal_pos).T
            else:
                raise ValueError("possible_goal_pos string must start with "
                                 "'random'")
        elif not isinstance(possible_goal_pos, (list, np.ndarray)):
            raise ValueError("possible_goal_pos must be a list of np.ndarrays, "
                         "a string, or None")
        return np.array(possible_goal_pos)

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

    def get_goals(self):
        """
        Get the current goal positions

        (shortcut func)
        """
        return [obj.goal_pos for obj in self.objectives]

    def reset(self, goal_locations:np.ndarray|None=None, n_goals=None) ->Dict:
        """
            reset

        resets the environement to a new episode
        https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.reset
        """
        super().reset()

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
                                             goal_pos=self.goal_pos,
                                             **self.objkws)
            self.objectives.append(objective)

        return self.get_observation()

    def _render_matplotlib(self, goalkws=dict(), **kws):
        """
        Take existing mpl render and add spatial goals
        """
        if super()._render_matplotlib(**kws):
            # Render the spatial goals
            self._render_mpl_spat_goals(**goalkws)

    def _render_mpl_spat_goals(self, facecolor="red", alpha=0.1,
                               marker="x", c="red"):
        R, fig, ax = self._get_mpl_render_cache()
        initialize = "spat_goals" not in R or \
                len(self.objectives) != len(R["spat_goals"])
        if initialize:
            R["spat_goals"]       = []
            R["spat_goal_radius"] = []
            for spat_goal in self.objectives:                
                if self.dimensionality == "2D":
                    x, y = spat_goal().T
                else:
                    x, y = np.zeros(np.shape(spat_goal())), spat_goal()
                sg = plt.scatter(x,y, marker=marker, c=c)
                ci = plt.Circle(spat_goal().ravel(), spat_goal.radius,
                                facecolor=facecolor, alpha=alpha)
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


    def _is_terminal_state(self):
        """
        Whether the current state is a terminal state
        """
        # Check our objectives
        test_objective = 0
        # Loop through objectives, checking if they are satisfied
        while test_objective < len(self.objectives):
            rewards, agents = self.objectives[test_objective].check(self.Agents)
            if len(agents):
                self.objectives.pop(test_objective)
                # Set the reward for the agent(s)
                for (agent, reward) in zip(agents, rewards):
                    self.reward_caches[agent].append(reward)
                # Verbose debugging
                if self.verbose:
                    print("objective {} satisfied by agents {}".format(
                    test_objective, agents),
                    "remaining objectives {}".format(len(self.objectives)))
            else:
                test_objective += 1
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
        if self._is_terminal_state():
            self.reset()

active = True
if active and __name__ == "__main__":

    plt.close('all')

    # Test reward class
    r1=Reward(1, 0.01, expire_clock=1, decay="linear")
    r1.plot_theoretical_reward()

    # Test the environment
    env = SpatialGoalEnvironment(n_goals=2, params={'dimensionality':'2D'},
                                 reward=r1, render_every=1, 
                                 teleport_on_reset=False,
                                 verbose=True)
    Ag = Agent(env)
    env.add_agents(Ag)
    # Note: if Agent() class detects if its env is a TaskEnvironment,
    # we could have the agent's code automate this last step. In other words,
    # after setting the agent's environment, it could automatically add itself
    # to the environment's list of agents.
    env.reset()

    # Prep the rendering figure
    plt.ion(); env.render(); plt.show()
    plt.pause(0.1)

    # Define some helper functions
    # get_goal_vector   = lambda: env.get_goals()[0][0] - Ag.pos
    get_goal_vector   = lambda: env.get_vectors_between___accounting_for_environment(
            env.get_goals()[0][0], Ag.pos)[0]
    # get_goal_distance = lambda: np.linalg.norm(get_goal_vector())


    # -----------------------------------------------------------------------
    # TEST 1 AGENT
    # -----------------------------------------------------------------------
    # Run the simulation, with the agent drifting towards the goal when
    # it is close to the goal
    s = input("Single Agent. Press enter to start")
    resets = 0
    while True:
        dir_to_reward = get_goal_vector()
        # print("dir_to_reward", dir_to_reward)
        drift_velocity = 3 * Ag.speed_mean * \
                (dir_to_reward / np.linalg.norm(dir_to_reward))
        observation, reward, terminate_episode, _, info = \
                env.step1(drift_velocity)
        if reward > 0:
            print("Reward_value:", reward)
        env.render()
        plt.pause(0.00001)
        if terminate_episode:
            resets += 1
            print("done! reward:", reward)
            env.reset()
            if resets >= 0:
                break

    # -----------------------------------------------------------------------
    # TEST Multi-agent (2 agents, second agent not pointed at goal)
    # -----------------------------------------------------------------------
    Ag2 = Agent(env)
    env.add_agents(Ag2)
    s = input("Two agents. Press enter to start")
    resets = 0
    while True:
        dir_to_reward = get_goal_vector()
        # print("dir_to_reward", dir_to_reward)
        drift_velocity = 3 * Ag.speed_mean * \
                (dir_to_reward / np.linalg.norm(dir_to_reward))
        observation, reward, terminate_episode, _, info = \
                env.step(drift_velocity)
        env.render()
        plt.pause(0.00001)
        if any(terminate_episode.values()):
            resets += 1
            print("done! reward:", reward)
            env.reset()
            if resets >= 10:
                break

