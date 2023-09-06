# ======================================
# Environments that implement tasks
# ======================================
#
# Key OpenAI Gym defines:
# (1) step()
# (2) reset()

import numpy as np
import time

import matplotlib.pyplot as plt

import pettingzoo
from gymnasium.spaces import Box, Space, Dict

# https://github.com/Farama-Foundation/PettingZoo

from types import FunctionType
from typing import List, Union
from functools import partial
import warnings
from copy import copy, deepcopy
import random

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent


class TaskEnvironment(Environment, pettingzoo.ParallelEnv):
    """
    Environment with task structure: there is a goal, and when the
    goal is reached, it terminates an episode, and starts a new episode
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
    render_every_framestep : int
        How often to render the environment (in framesteps)
    teleport_on_reset : bool
        Whether to teleport agents to random positions on reset
    save_expired_rewards : bool
        Whether to save expired rewards in the environment
    goals : list
        List of goals to replenish the goal cache with on reset
    goalcachekws : dict
        Keyword arguments to pass to GoalCache
    episode_termination_delay : float
        How long to wait before terminating an episode after the goal is reached
    **kws :
        Keyword arguments to pass to Environment
    """
    default_params = {} #for RatInABox 
    metadata = {"render_modes": ["matplotlib", "none"], "name": "TaskEnvironment-RiaB"}

    def __init__(
        self,
        *pos,
        dt=0.01,
        render_mode="matplotlib",
        render_every=None,
        render_every_framestep=2,
        teleport_on_reset=False,
        save_expired_rewards=False,
        goals=[],  # one can pass in goal objects directly here
        goalcachekws=dict(),
        rewardcachekws=dict(),
        episode_terminate_delay=0,
        verbose=False,
        **kws,
    ):
        super().__init__(*pos, **kws)
        self.dynamic = {"walls": [], "objects": []}
        self.Ags: dict[str, Agent] = {}  # dict of agents in the environment

        self.goal_cache: GoalCache = GoalCache(self, **goalcachekws)
        # replenish from this list of goals on reset
        self.goal_cache.reset_goals = goals if isinstance(goals, list) else [goals]

        self.t = 0  # current time
        self.dt = dt  # time step
        self.history = {"t": []}  # history of the environment

        if render_every is None and render_every_framestep is not None:
            self.render_every = render_every_framestep  # How often to render
        elif render_every is not None:
            self.render_every = render_every / self.dt
        self.verbose = verbose
        self.render_mode: str = render_mode  # options 'matplotlib'|'pygame'|'none'
        self._stable_render_objects: dict = {}  # objects that are stable across
        # a rendering type

        # ----------------------------------------------
        # Agent-related task config
        # ----------------------------------------------
        self.teleport_on_reset = teleport_on_reset  # Whether to teleport
        # agents to random

        # ----------------------------------------------
        # Setup gym primatives
        # ----------------------------------------------
        # Setup observation space from the Environment space
        self.observation_spaces: Dict[Space] = Dict({})
        self.action_spaces: Dict[Space] = Dict({})
        self.agent_names: List[str] = []
        self.agents: List[str] = []  # pettingzoo variable
        # that tracks all agents who are
        # still active in an episode
        self.infos: dict = {}  # pettingzoo returns infos in step()
        self.observation_lambda = {}  # lambda functions to attain an agents
        # observation information -- a vector
        # of whatever info in the agent defines
        # its current observation -- DEFAULT: pos

        # Episode history
        self.episodes: dict = {}  # Written to upon completion of an episode
        self.episodes["episode"] = []
        self.episodes["start"] = []
        self.episodes["end"] = []
        self.episodes["duration"] = []
        self.episodes["meta_info"] = []
        self.episode = 0
        # Episode state and option
        self.episode_state = {"delayed_term": False}
        self.episode_terminate_delay = episode_terminate_delay

        # Reward cache specifics
        self.reward_caches: dict[str, RewardCache] = {}
        self.save_expired_rewards = save_expired_rewards
        self.expired_rewards: List[RewardCache] = []
        self.rewardcachekws = rewardcachekws

    def observation_space(self, agent_name: str):
        return self.observation_spaces[agent_name]

    def action_space(self, agent_name: str):
        return self.action_spaces[agent_name]

    def add_agents(
        self,
        agents: Union[dict, List[Agent], Agent],
        names: Union[None, List] = None,
        maxvel: float = 50.0,
        **kws,
    ):
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
            raise NotImplementedError(
                "Does not yet support agents with different dt from envrionment"
            )
        if isinstance(agents, dict):
            names = list(agents.keys())
            agents = list(agents.values())
        elif names is None:
            start = len(self.Ags)
            names = ["agent_" + str(start + i) for i in range(len(agents))]
        # Enlist agents
        for i, (name, agent) in enumerate(zip(names, agents)):
            self.Ags[name] = agent
            self.agent_names.append(name)
            agent.name = name  # attach name to agent
            # Add the agent's action space to the environment's action spaces
            # dict
            D = int(self.dimensionality[0])
            self.action_spaces[name] = Box(low=-maxvel, high=maxvel, shape=(D,))

            # Add the agent's observation space to the environment's
            # observation spaces dict
            ext = [self.extent[i : i + 2] for i in np.arange(0, len(self.extent), 2)]
            lows, highs = np.array(list(zip(*ext)), dtype=np.float_)
            self.observation_spaces[name] = Box(low=lows, high=highs, dtype=np.float_)
            self.observation_lambda[name] = lambda agent: agent.pos

            # Attach a reward cache for the agent
            cache = RewardCache(**self.rewardcachekws)
            self.reward_caches[name] = cache
            agent.reward = cache
            # Ready the goal_cache for the agent
            self.goal_cache.add_agent(agent)
            # Set the agents time to the environment time
            agent.t = self.t  # agent clock is aligned to environment,
            # in case a new agent is placed in the env
            # on a later episode
            self.infos[name] = {}  # pettingzoo requirement

        self.reset()  # reset the environment with new agent

    def remove_agents(self, agents):
        """
        Remove agents from the environment
        Parameters
        ----------
        agents
        """
        agents = self._agentnames(agents)
        for name in agents:
            self.reward_caches.pop(name)
            self.observation_spaces.spaces.pop(name)
            self.action_spaces.spaces.pop(name)
            self.Ags.pop(name)
            self.agent_names.remove(name)
            if name in self.agents:
                self.agents.remove(name)
        self.reset()

    def _agentnames(self, agents=None) -> list[str]:
        """
        Convenience function for generally hanlding all the ways that
        users might want to specify agents, names, numbers, or objects
        themselves. Also as a "scalar" or list of such thing. This makes
        several functions that call this robust to ways users specify
        agents.
        """
        if isinstance(agents, Agent):
            agents: list[str] = [agents.name]
        if isinstance(agents, int):
            agents = [self.agent_names[agents]]
        elif isinstance(agents, str):
            agents = [agents]
        elif isinstance(agents, list):
            new: list[str] = []
            for agent in agents:
                if isinstance(agent, int):
                    new.append(self.agent_names[agent])
                elif isinstance(agent, Agent):
                    new.append(agent.name)
                elif isinstance(agent, str):
                    new.append(agent)
                else:
                    raise TypeError("agent must be an Agent, int, or str")
            agents = new
        elif agents is None:
            agents = self.agent_names
        return agents

    def _dict(self, V) -> dict:
        """
        Convert a list of values to a dictionary of values keyed by agent name
        """
        return (
            {name: v for (name, v) in zip(self.agent_names, V)}
            if hasattr(V, "__iter__")
            else {name: V for name in self.agent_names}
        )

    def _is_terminal_state(self):
        """Whether the current state is a terminal state"""
        # Check our objectives
        test_goal = 0
        # Loop through objectives, checking if they are satisfied
        rewards, agents = self.goal_cache.check(remove_finished=True)
        for reward, agent in zip(rewards, agents):
            self.reward_caches[agent].append(reward)
        if self.verbose >= 2:
            print("GOALS:", self.goal_cache.goals)
        # Return if no objectives left
        no_objectives_left = len(self.goal_cache) == 0
        return no_objectives_left

    def _is_truncated_state(self):
        """
        whether the current state is a truncated state,
        see https://gymnasium.farama.org/api/env/#gymnasium.Env.step

        default is false: an environment by default will have a terminal state,
        ending the episode, whereon users should call reset(), but not a
        trucation state ending the mdp.
        """
        return False

    def seed(self, seed=None):
        """Seed the random number generator"""
        np.random.seed(seed)

    def reset(self, seed=None, episode_meta_info=False, options=None):
        """How to reset the task when finished"""
        if seed is not None:
            self.seed(seed)
        if self.verbose:
            print("Resetting")
        if len(self.episodes["start"]) > 0:
            self.write_end_episode(episode_meta_info=episode_meta_info)

        # Reset active non-terminated agents
        self.agents = copy(self.agent_names)

        # Clear rendering cache
        self.clear_render_cache()

        # If teleport on reset, randomly pick new location for agents
        if self.teleport_on_reset:
            for agent_name, agent in self.Ags.items():
                # agent.update()
                agent.pos = self.sample_positions(1)[
                    0
                ]  # random position in the environment
                if len(agent.history["pos"]) > 0:
                    agent.history["pos"][-1] = agent.pos

        # Increment episode counter
        if len(self.episodes["duration"]) and self.episodes["duration"][-1] == 0:
            for key in self.episodes:
                self.episodes[key].pop()
        else:
            self.episode += 1
        self.write_start_episode()

        # Restore agents to active state (pettingzoo variable)
        self.agents = copy(self.agent_names)
        # print("Active agents: ", self.agents)

        # Reset goals
        self.goal_cache.reset()

        # Episode state trackers
        # we have not applied a delayed terminate
        self.episode_state["delayed_term"] = False

        return self.get_observation(), self.infos

    def update(self, update_agents=False):
        """
        How to update the task over time --- update things
        directly connected to the task
        """
        self.t += self.dt  # base task class only has a clock
        self.history["t"].append(self.t)

    def step(
        self,
        actions: Union[dict, np.array, None] = None,
        dt=None,
        drift_to_random_strength_ratio=1,
        *pos,
        **kws,
    ):
        """
            step()

        step() functions in Gymnasium paradigm usually take an action space
        action, and return the next state, reward, whether the state is
        terminal, and an information dict

        different from update(), which updates this environment. this function
        executes a full step on the environment with an action from the agents

        https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.step
        """

        # If the user passed drift_velocity, update the agents
        if actions is not None:
            if len(self.agents) == 0:
                raise AttributeError(
                    "Action is given, but there are no "
                    "active agents. If there are no agents, try adding an "
                    "agent with .add_agents(). If there are agents, "
                    "try .reset() to restore inactive agents w/o goals to "
                    "active."
                )
            actions = actions if isinstance(actions, dict) else self._dict(actions)
        else:
            # Move agents randomly on None
            actions = self._dict([None for _ in range(len(self.Ags))])

        if not isinstance(drift_to_random_strength_ratio, dict):
            drift_to_random_strength_ratio = self._dict(drift_to_random_strength_ratio)
        for agent, action in zip(self.agents, actions.values()):
            Ag = self.Ags[agent]
            dt = dt if dt is not None else Ag.dt
            if action is not None:
                action = np.array(action).ravel()
                action[np.isnan(action)] = 0
            strength = drift_to_random_strength_ratio[agent]
            Ag.update(
                dt=dt, drift_velocity=action, drift_to_random_strength_ratio=strength
            )

        # Update the reward caches for time decay of existing rewards
        for reward_cache in self.reward_caches.values():
            reward_cache.update()

        # Udpate the environment, which can add new rewards to caches
        self.update(*pos, **kws)

        # Return the next state, reward, whether the state is terminal,
        terminal = self._is_terminal_state()

        # Episode termination delay?
        if (
            terminal
            and self.episode_terminate_delay
            and self.episode_state["delayed_term"] == False
        ):
            unrewarded_episode_padding = TimeElapsedGoal(
                self,
                reward=no_reward_default,
                wait_time=self.episode_terminate_delay,
                verbose=False,
            )
            self.episode_state["delayed_term"] = True
            self.goal_cache.append(unrewarded_episode_padding)
            terminal = self._is_terminal_state()

        # If any terminal agents, remove from set of active agents
        truncations = self._dict(self._is_truncated_state())
        for agent, term in self._dict(self._is_terminal_state()).items():
            if term and agent in self.agents or truncations[agent]:
                self.agents.remove(agent)

        # Create pettingzoo outputs
        outs = (
            self.get_observation(),
            self.get_reward(),
            self._dict(terminal),
            self._dict(self._is_truncated_state()),
            self._dict([self.infos]),
        )
        if self.verbose:
            print(f"🐀 action @ {self.t}:", actions)
            print(f"🌍 step @ {self.t}:", outs)
        return outs

    def step1(self, action=None, *pos, **kws):
        """
        shortcut for stepping when only 1 agent exists...makes it behave
        like gymnasium instead of pettingzoo
        """
        results = self.step({self.agent_names[0]: action}, *pos, **kws)
        results = [x[self.agent_names[0]] for x in results]
        return results

    def get_observation(self):
        """Get the current state of the environment"""
        return {
            name: self.observation_lambda[name](agent)
            for name, agent in self.Ags.items()
        }

    def get_reward(self):
        """Get the current reward state of each agent"""
        return {name: agent.reward.get_total() for name, agent in self.Ags.items()}

    def set_observation(
        self,
        agents: Union[List, str, Agent],
        spaces: Union[List, Space],
        observation_lambdass: Union[List, FunctionType],
    ):
        """
        Set the observation space and observation function for an agent(s)

        - The space is a gym.Space that describes the set of possible
        values an agents observation can take
        - The lambda takes an agent argument and returns a tuple/list of
        numbers regardings the agents position. users can set the lambda to
        extract whatever attributes of the agent encode its state

        The default for agents is there position. But if you would like to
        change the observation to cell firing or velocity, you can do that
        here.

        Input
        ----
        agents: List, str, Agent
            The agent(s) to change the observation space for
        spaces: List, gym.Space
            The observation space(s) to change to. If a list, then it
            must be a list of gymnasium spaces (these just describe the
            full range of values an observation can take, and RL libraries
            often use these to sample the space.)
        observation_lambdass: List, Function
            The observation function(s) to change to...these should take an
            agent and output the vector of numbers describing what you
            consider your agents' state. you can set the function to grab
            whatever you'd like about the agent: it's position, velocity,
        """
        agents = self._agentnames(agents)
        if not isinstance(spaces, list):
            spaces = [spaces]
        if not isinstance(observation_lambdass, list):
            observation_lambdass = [observation_lambdass]
        if len(spaces) != len(observation_lambdass):
            raise ValueError(
                "observation space and observation lambda " "must be the same length"
            )
        for ag, sp, obs in zip(agents, spaces, observation_lambdass):
            print("Changing observation space for {ag}")
            self.observation_spaces[ag] = sp
            self.observation_lambda[ag] = obs

    # ----------------------------------------------
    # Reading and writing episode data
    # ----------------------------------------------
    def _current_episode_start(self):
        return 0 if not len(self.episodes["start"]) else self.episodes["end"][-1]

    def write_start_episode(self):
        self.episodes["episode"].append(self.episode)
        self.episodes["start"].append(self._current_episode_start())
        if self.verbose:
            print("starting episode {}".format(self.episode))
            print("episode start time: {}".format(self.episodes["start"][-1]))

    def write_end_episode(self, episode_meta_info="none"):
        self.episodes["end"].append(self.t)
        self.episodes["duration"].append(self.t - self.episodes["start"][-1])
        self.episodes["meta_info"].append(episode_meta_info)
        if self.verbose:
            print("ending episode {}".format(self.episode))
            print("episode end time: {}".format(self.episodes["end"][-1]))
            print("episode duration: {}".format(self.episodes["duration"][-1]))

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
        if render_mode == "matplotlib":
            out = self._render_matplotlib(*pos, **kws)
            assert out is not None
            return out
        elif render_mode == "pygame":
            return self._render_pygame(*pos, **kws)
        elif render_mode == "none":
            pass
        else:
            raise ValueError("method must be 'matplotlib' or 'pygame'")

    def _render_matplotlib(self, *pos, agentkws: dict = dict(), **kws):
        """
        Render the environment using matplotlib
        `
        Inputs
        ------
        agentkws: dict
            keyword arguments to pass to the agent's render method
        """

        R, fig, ax = self._get_mpl_render_cache()

        if np.mod(self.t, self.render_every) < self.dt:
            # Skip rendering unless this is redraw time
            return fig, ax

        else:
            # Render the environment
            self._render_mpl_env()

            # Render the agents
            self._render_mpl_agents(**agentkws)

            return fig, ax

    def _get_mpl_render_cache(self):
        if "matplotlib" not in self._stable_render_objects:
            R = self._stable_render_objects["matplotlib"] = {}
        else:
            R = self._stable_render_objects["matplotlib"]
        if "fig" not in R:
            fig, ax = plt.subplots(1, 1)
            R["fig"] = fig
            R["ax"] = ax
        else:
            fig, ax = R["fig"], R["ax"]
        return R, fig, ax

    def _render_mpl_env(self):
        R, fig, ax = self._get_mpl_render_cache()
        if "environment" not in R:
            R["environment"] = self.plot_environment(fig=fig, ax=ax, autosave=False)
            R["title"] = fig.suptitle(
                "t={:.2f}\nepisode={}".format(self.t, self.episode)
            )
        else:
            R["title"].set_text("t={:.2f}\nepisode={}".format(self.t, self.episode))

    def _render_mpl_agents(self, framerate=60, alpha=0.7, t_start="episode", **kws):
        """
        Render the agents 🐀

        Inputs
        ------
        framerate: float
            the framerate at which to render the agents
        alpha: float
            the alpha value to use for the agents
        t_start: float
            the time at which to start rendering the agents
                - "episode" : start at the beginning of the current episode
                - "all" : start at the beginning of the first episode
                - float : start at the given time
        **kws
            keyword arguments to pass to the agent's style (point size, color)
            see _agent_style
        """
        R, fig, ax = self._get_mpl_render_cache()
        initialize = "agents" not in R
        if t_start == "episode":
            t_start = self.episodes["start"][-1]
        elif t_start == "all" or t_start is None:
            t_start = self.episodes["start"][0]

        def get_agent_props(agent, color):
            t = np.array(agent.history["t"])
            startid = np.nanargmin(np.abs(t - (t_start)))
            skiprate = int((1.0 / framerate) // agent.dt)
            trajectory = np.array(agent.history["pos"][startid::skiprate])
            t = t[startid::skiprate]
            c, s = self._agent_style(
                agent, t, color, startid=startid, skiprate=skiprate, **kws
            )
            return trajectory, c, s

        if initialize or len(R["agents"]) != len(self.Ags):
            R["agents"] = []
            for i, agent in enumerate(self.Ags.values()):
                if len(agent.history["t"]):
                    trajectory, c, s = get_agent_props(agent, i)
                    ax.scatter(
                        *trajectory.T, s=s, alpha=alpha, zorder=0, c=c, linewidth=0
                    )
                    R["agents"].append(ax.collections[-1])
        else:
            for i, agent in enumerate(self.Ags.values()):
                scat = R["agents"][i]
                trajectory, c, s = get_agent_props(agent, i)
                scat.set_offsets(trajectory)
                scat.set_facecolors(c)
                scat.set_edgecolors(c)
                scat.set_sizes(s)

    @staticmethod
    def _agent_style(
        agent: Agent,
        time,
        color=0,
        skiprate=1,
        startid=0,
        point_size: bool = 15,
        decay_point_size: bool = False,
        plot_agent: bool = True,
        decay_point_timescale: int = 10,
    ):
        if isinstance(color, int):
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][color]
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
            for item in set(R.keys()) - set(("fig", "ax")):
                R.pop(item)

    def close(self):
        """gymnasium close() method"""
        self.clear_render_cache()
        if "fig" in self._stable_render_objects:
            if isinstance(self._stable_render_objects["fig"], plt.Figure):
                plt.close(self._stable_render_objects["fig"])


class Reward:
    """
    When an task goal is triggered, reward goal is attached an Agent's
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
        "constant": lambda a, x: a,
        "linear": lambda a, x: a * x,
        "exponential": lambda a, x: a * np.exp(x),
        "none": lambda a, x: 0,
    }
    decay_knobs_preset = {
        "linear": [1],
        "constant": [1],
        "exponential": [2],
        "none": [0],
    }

    def __init__(
        self,
        init_state=1,
        dt=0.01,
        expire_clock=None,
        decay=None,
        decay_knobs=[],
        external_drive: Union[FunctionType, None] = None,
        external_drive_strength=1,
        name=None,
    ):
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
        self.state = (
            init_state if not isinstance(init_state, FunctionType) else init_state()
        )
        self.dt = dt
        self.expire_clock = (
            expire_clock if isinstance(expire_clock, (int, float)) else dt
        )
        if isinstance(decay, str):
            self.preset = decay
            self.decay_knobs = decay_knobs or self.decay_knobs_preset[self.preset]
            self.decay = partial(self.decay_preset[self.preset], *self.decay_knobs)
        else:
            self.preset = "custom" if decay is not None else "constant"
            self.decay_knobs = decay_knobs or self.decay_knobs_preset[self.preset]
            self.decay = decay or self.decay_preset["constant"]
        self.external_drive = external_drive
        self.external_drive_strength = external_drive_strength
        self.history = {"state": [], "expire_clock": []}
        self.name = (
            name
            if name is not None
            else self.__class__.__name__ + " " + str(hash(self))[:5]
        )
        # if a goal provides a reward, then this attribute is used to track
        # the goal that provided the reward
        self.goal: Union[None, Goal] = None  # optional store goal linked to
        # reward

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
        self.history["state"].append(self.state)
        self.history["expire_clock"].append(self.expire_clock)
        return not (self.expire_clock <= 0)

    def get_delta(self, state=None):
        """\delta(reward) for a dt"""
        state = self.state if state is None else state
        if self.external_drive is not None:
            target_gradient = self.external_drive()
            strength = self.external_drive_strength
            change = strength * (target_gradient - state) - self.decay(state)
        else:
            change = -(self.decay(state))
        return change

    def plot_theoretical_reward(self, timerange=(0, 1), name=None):
        """
        plot the reward dynamics : shows the user how their parameters of
        interest setup reward dynamics, without updating the object
        """
        rewards = [self.state]
        name = self.name if name is None else name
        timesteps = np.arange(timerange[0], timerange[1], self.dt)
        pre_expire_timesteps = np.arange(
            timerange[0], self.expire_clock + self.dt, self.dt
        )
        for t in pre_expire_timesteps[1:]:
            r = rewards[-1] + self.get_delta(state=rewards[-1]) * self.dt
            rewards.append(r)
        plt.plot(
            pre_expire_timesteps,
            rewards[: len(timesteps)],
            label=f"reward={self.preset}, " f"knobs={self.decay_knobs}",
        )
        y1 = np.min((self.state, 0, np.min(plt.gca().get_ylim())))
        y2 = np.max((self.state, 0, np.max(plt.gca().get_ylim())))
        plt.ylim((y1, y2))
        plt.xlim((timerange[0], timerange[1]))
        plt.axvspan(0, self.expire_clock, color="r", alpha=0.2)
        plt.text(
            np.mean((plt.gca().get_xlim()[0], self.expire_clock)),
            np.mean(plt.gca().get_ylim()),
            f"{name}\nactive",
            backgroundcolor="black",
            color="white",
        )
        plt.text(
            np.mean((self.expire_clock, plt.gca().get_xlim()[-1])),
            np.mean(plt.gca().get_ylim()),
            f"{name}\nexpires",
            backgroundcolor="black",
            color="white",
        )
        plt.gca().set(xlabel="time (s)", ylabel=f"{name} signal")
        return plt.gcf(), plt.gca()


class RewardCache:
    """
    RewardCache

    A cache of all `active` rewards attached to an agent

    # Parameters
    default_reward_level : float
        default reward level for all rewards in the cache
    verbose : bool, int
        print reward cache related details
    """

    def __init__(self, default_reward_level=0, verbose=False):
        self.default_reward_level = default_reward_level
        self.cache: List[Reward] = []
        self.verbose = verbose
        self.stats = {
            "total_steps_active": 0,
            "total_steps_inactive": 0,
            "max": -np.inf,
            "min": np.inf,
            "uniq_rewards": [],
            "uniq_goals": [],
        }

    def append(self, reward: Reward, copymode=True):
        assert isinstance(reward, Reward), "reward must be a Reward object"
        if reward is not None:
            if copymode:
                reward = copy(reward)
            if reward.name not in self.stats["uniq_rewards"]:
                self.stats["uniq_rewards"].append(reward.name)
            if reward.goal.name not in self.stats["uniq_goals"]:
                self.stats["uniq_goals"].append(reward.goal.name)
            self.cache.append(reward)

    def update(self):
        """Update"""
        # If any rewards ...
        if self.cache:
            self.stats["total_steps_active"] += 1
            # Iterate through each reward, updating
            for reward in self.cache:
                reward_still_active = reward.update()
                if not reward_still_active:
                    self.cache.remove(reward)
                    if self.verbose:
                        print("Reward removed from cache")
        # Else, increment inactivity tracker
        else:
            self.stats["total_steps_inactive"] += 1

    def get_total(self):
        """
        If there are any active rewards, return the sum of their values.
        """
        r = sum([reward.state for reward in self.cache]) + self.default_reward_level
        assert not np.isnan(r), "reward is nan"
        if r > self.stats["max"]:
            self.stats["max"] = r
        if r < self.stats["min"]:
            self.stats["min"] = r
        return r

    def get_fraction(self):
        """Return the fraction of the total reward value relative to the max
        and min values so far experienced."""
        r = self.get_total()
        return (r - self.stats["min":]) / (self.stats["max"] - self.stats["min"])


reward_default = Reward(1, 0.01, expire_clock=1, decay="linear")
no_reward_default = Reward(
    0, 0.01, expire_clock=0.1, decay="none"
)  # A reward object which doesn't give any reward (for use in goals where no reward is then given)


class Goal:
    """
    Abstract `Objective` class that can be used to define finishing coditions
    for a task
    """

    def __init__(
        self,
        env: Union[None, TaskEnvironment] = None,
        reward=reward_default,
        name=None,
        **kws,
    ):
        self.env = env
        self.reward = reward
        self.reward.goal = self
        self.name = (
            name
            if name is not None
            else self.__class__.__name__ + " " + str(hash(random.random()))[:5]
        )

    def __hash__(self):
        """hash for uniquely identifying a goal"""
        hashes = []
        for value in self.__dict__.values():
            try:
                hashes.append(hash(value))
            except:
                pass
        return hash(tuple(hashes))

    def check(self, agents=None):
        """
        Check if the goal is satisfied for agents and report which agents
        satisfied the goal and if any rewards are rendered
        """
        raise NotImplementedError("check() must be implemented")

    def __call__(self):
        """
        Can be used to report its value to the environment
        (Not required -- just a convenience)
        """
        pass


class GoalCache:
    """
    Organizes a collection of goals shared across agents.
    Can agents tackle goals in series or parallel?
    Does each agent have to finish the goal? Or first agent to a goal
    consumes it? The cache handles the logic of how several goals
    are handled by several agents.

    # Inputs
    --------
    goalorder : str
        Goal relations
        - "sequential"    : goal must be accomplished in sequence
        - "nonsequential" : any order
        - "custom" : provide a function that gives valid successor
                     indices for a given goal
    agentmode : str
        Agent handling
        - "interact" :: an goal is consumed/satisfied if any
                        one agent satisfies it
        - "noninteract" :: all agents must satisfy goal
    verbose : bool
        Print debug statements?
    """

    def __init__(
        self,
        env,
        goalorder="nonsequential",
        agentmode="interact",
        reset_goals: List[Goal] = [],
        reset_n_goals: int = 1,
        reset_orders_goal: bool = False,
        verbose=False,
        **kws,
    ):
        self.env = env
        self.goals: dict[str, list[Goal]] = {name: [] for name in self.env.Ags.keys()}
        self.goalorder = goalorder  # sequential | nonsequential
        self.agentmode = agentmode  # interact | noninteract
        self.reset_goals: List[Goal] = reset_goals  # list of goals to reset from
        self.reset_n_goals: int = (
            reset_n_goals  # if >0, then randomly select from reservoir on .reset()
        )
        self.reset_orders_goal: bool = (
            reset_orders_goal  # if True, then keep the ordering of
        )
        # goals chosen from the pool
        if self.reset_n_goals <= 0:
            raise ValueError("reset_n_goals must be > 0")
        # list of current goals to satisfy per agent, these are drawn
        # from reset_goals list
        # records the last goal that was achieved, if sequential
        self._if_sequential__last_acheived = {
            agent: -1 for agent in self.env.Ags.keys()
        }
        self.verbose = verbose
        if self.goalorder not in ["sequential", "nonsequential", "custom"]:
            raise ValueError(
                "goalorder must be 'sequential', 'nonsequential'" ", or 'custom'"
            )
        if self.agentmode not in ["interact", "noninteract"]:
            raise ValueError("agentmode must be 'interact' or 'noninteract'")

    def add_agent(self, agent):
        """
        Add a new agent to the goal cache
        """
        if isinstance(agent, str):
            self.goals[agent] = []
        elif isinstance(agent, Agent):
            self.goals[agent.name] = []
        else:
            raise ValueError("agent must be a string or Agent object")

    def check(self, remove_finished: bool = True):
        """
        Check if any goals are satisfied
        --------
        # Inputs
        --------
        remove_finished : bool
            Remove finished goals from the cache?
        --------
        # Returns
        --------
        rewards : list of rewards
        agents : list of agents that satisfied the goal
        """
        verbose = self.verbose

        # If no goals, no rewards and no agents
        if len(self.goals) == 0:
            return [], []

        if verbose:
            print("entering check")
            pre = repr(self.goals)
        # -------------------------------------------------------
        # Agents must accomplish goal, following the sequence of the
        # goal list?
        # -------------------------------------------------------
        if self.goalorder == "sequential":
            rewards, agents = [], []
            for agent in self.env.agent_names:
                assert agent in self.goals.keys(), (
                    f"agent {agent} not in goal cache ... "
                    "try adding it with GoalCache.add_agent"
                )
                if len(self.goals[agent]) == 0:
                    continue
                this: int = self._if_sequential__last_acheived[agent] + 1
                solution_agents = self.goals[agent][this].check(agent)
                for agent, reward in solution_agents.items():
                    rewards.append(reward if reward is not None else None)
                    agents.append(agent)
                    self._if_sequential__last_acheived[agent] = this
                    if remove_finished:
                        self.pop(agent, this)
        # -------------------------------------------------------
        # Agents do not have to accomplish goal in sequence, any
        # order is fine
        # -------------------------------------------------------
        elif self.goalorder == "nonsequential":
            rewards, agents = [], []
            for agent in self.env.agent_names:
                if len(self.goals[agent]) == 0:
                    continue
                g = 0  # goal index
                while g < len(self.goals[agent]):
                    solution_agents = self.goals[agent][g].check(agent)
                    for agent, reward in solution_agents.items():
                        if not isinstance(reward, Reward):
                            import pdb

                            pdb.set_trace()
                        rewards.append(reward if reward is not None else None)
                        agents.append(agent)
                        if remove_finished:
                            self.pop(agent, g)
                    g += 1
        else:
            raise ValueError("Unknown mode: {}".format(self.goalorder))
        if verbose:
            post = repr(self.goals)
            if pre != post:
                print("---- PRE-check, GoalCache ----")
                print(pre)
                print("---- POST-check, GoalCache ----")
                print(post)
                print("exiting check")
        return rewards, agents

    def pop(self, agent_name: str, goal_index: int):
        """Remove a goal from the cache"""
        if self.agentmode == "noninteract":
            if self.verbose:
                print(f"popping {agent_name}.{goal_index}!")
            # print("Non-interact mode, pop only agent's index")
            self.goals[agent_name].pop(goal_index)
            if self.goalorder == "sequential":
                s = self._if_sequential__last_acheived[agent_name]
                self._if_sequential__last_acheived[agent_name] = max(s - 1, -1)
        elif self.agentmode == "interact":
            if self.verbose:
                print(f"popping all.{goal_index}!")
            # print("Interacting mode")
            for agent in self.env.agent_names:
                self.goals[agent].pop(goal_index)
                if self.goalorder == "sequential":
                    s = self._if_sequential__last_acheived[agent]
                    self._if_sequential__last_acheived[agent] = max(s - 1, -1)

    def is_empty():
        """checks if empty cache"""
        return [len(g) == 0 for g in self.goals.values()]

    def get_goals(self) -> tuple:
        """Get all unique goals"""
        from itertools import chain

        if self.agentmode == "noninteract":
            goals = tuple(self.goals.values())
            return goals[0] if not len(goals) == 0 else goals
        elif self.agentmode == "interact":
            return tuple(set(chain(*self.goals.values())))
        else:
            raise ValueError("Unknown mode: {}".format(self.agentmode))

    def get_agent_goals(self, agent) -> dict:
        """Get all goals for each agent
        Inputs
        ------
        agent : str | list of str | int | list of int |
                    agent | list of agent | None
        """
        agent = self.env._agentnames(agent)
        return {a: self.goals[a] for a in agent}

    def __len__(self):
        return len(self.get_goals())

    def append(self, goal: Goal, agent=None):
        if self.verbose:
            print(f"appending {goal} to agents={agent}")
        names = self.env._agentnames(agent)
        for name in names:
            if name not in self.goals:
                self.goals[name] = []
            self.goals[name].append(goal)
            self._if_sequential__last_acheived[name] = -1

    def clear(self):
        if self.verbose:
            print("Clearing goals")
        self.goals.clear()

    def reset(self, seed=None):
        """
        Reset the goal cache for the next episode -- drawing from
        self.reset_goals. It selects self.reset_n_goals from the pool
        and adds them to the cache. If self.reset_orders_goal is True,
        it maintains the ordering from the pool.
        """
        import random

        # Clear existing goals
        self.clear()

        # Validate that we have enough goals to reset
        if len(self.reset_goals) < self.reset_n_goals:
            warnings.warn(
                "Not enough goals to replenish "
                "n={self.reset_n_goals} "
                f"\nlen(goals)={len(self.reset_goals)}"
            )
            n_reset = len(self.reset_goals)
        else:
            n_reset = self.reset_n_goals

        # Select goals
        if self.reset_orders_goal:
            # Maintain the ordering from the pool
            selected_goals = self.reset_goals[:n_reset]
        else:
            # Pick goals randomly without replacement
            selected_goals = random.sample(self.reset_goals, n_reset)

        # Append selected goals to each agent in the environment
        for agent_name in self.env.Ags.keys():
            for goal in selected_goals:
                self.append(goal, agent_name)

    def find(self, other_goal: Goal, agents=None) -> dict[str, Goal]:
        """Find an goal in the cache"""
        agentnames = self.env._agentnames(agents)
        results = dict()
        raise NotImplementedError("find() not implemented")
        return results


class TimeElapsedGoal(Goal):
    def __init__(self, *args, wait_time=1, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = self.env.t
        self.wait_time = wait_time
        self.verbose = verbose
        if verbose:
            print(f"time elapsed with {wait_time}")

    def check(self, agents=None):
        current_time = self.env.t
        if self.verbose:
            print(f"waited {current_time - self.start_time}")
        if current_time - self.start_time >= self.wait_time:
            return {agent: self.reward for agent in self.env._agentnames(agents)}
        else:
            return {}


class SpatialGoal(Goal):
    """
    Spatial goal objective: agent must reach a specific position

    Parameters
    ----------
    env : TaskEnvironment
        The environment that the objective is defined in
    reward_value : float
        The reward value that the objective gives
    pos : np.ndarray | None
        The position that the agent must reach
    goal_radius : float | None
        The radius around the goal position that the agent must reach

    see also : Goal
    """

    def __init__(
        self,
        *positionals,
        pos: Union[np.ndarray, None] = None,
        goal_radius=None,
        **kws,
    ):
        super().__init__(*positionals, **kws)
        if self.env is not None and self.env.verbose:
            print("new SpatialGoal: goal_pos = {}".format(pos))
        if pos is not None:
            self.pos = np.array(pos)
        else:
            self.pos = np.random.rand(int(len(self.env.extent) / 2))
        self.radius = (
            np.min((self.env.dx * 10, np.ptp(self.env.extent) / 10))
            if goal_radius is None
            else goal_radius
        )

    def _in_goal_radius(self, pos, goal_pos):
        """
        Check if a position is within a radius of a goal position
        """
        radius = self.radius
        distance = (
            lambda x, y: self.env.get_distances_between___accounting_for_environment(
                x, y, wall_geometry="line_of_sight"
            )
        )
        # return np.linalg.norm(pos - goal_pos, axis=1) < radius \
        #         if np.ndim(goal_pos) > 1 \
        #         else np.abs((pos - goal_pos)) < radius
        return distance(pos, goal_pos) < radius

    def __hash__(self):
        return super().__hash__()

    def check(self, agents=None):
        """
        Check if the goal is satisfied

        Parameters
        ----------
        agents : List[Agent] | List[int] | int | string | List[string]
            The agents to check the goal for (usually just one), None=All
            see TaskEnvironment._agentnames

        Returns
        -------
        dict[str, Reward] dict of rule-triggering agents and
                          their "reward" objects
        """
        agents = self.env._agentnames(agents)
        agents_reached_goal = []
        for agent in agents:
            agent_pos = self.env.Ags[agent].pos
            goal_pos = self.pos

            if self._in_goal_radius(agent_pos, goal_pos).all().any():
                agents_reached_goal.append(agent)
        return {agent: self.reward for agent in agents_reached_goal}

    def __eq__(self, other: Union[Goal, np.ndarray, list]):
        if isinstance(other, SpatialGoal):
            return np.all(self.pos == other.pos)
        elif isinstance(other, (np.ndarray, list)):
            return np.all(self.pos == np.array(other))

    def __call__(self) -> np.ndarray:
        """
        Can be used to report its value to the environment
        (Not required -- just a convenience)
        """
        return np.array(self.pos)


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

    See `TaskEnvronment` for more kws
    """

    default_params = {} #for RatInABox 

    # --------------------------------------
    # Some reasonable default render settings
    # --------------------------------------
    def __init__(
        self,
        *pos,
        possible_goals: Union[None, List[SpatialGoal]] = None,
        possible_goal_positions: Union[List, np.ndarray, str] = "random_5",
        current_goal_state: Union[None, np.ndarray, List[np.ndarray]] = None,
        goalkws=dict(),
        **kws,
    ):
        super().__init__(*pos, **kws)
        if possible_goals is None:
            self.goalkws = goalkws
            self.goal_cache.reset_goals = self._init_poss_goal_positions(
                possible_goal_positions
            )
        else:
            self.goal_cache.reset_goals = possible_goals

    def _init_poss_goal_positions(
        self, possible_goal_position: Union[List, np.ndarray, str]
    ) -> list[SpatialGoal]:
        """
        Shortcut function for intializing a set of goals from numpy
        array or from a string saying how many goals to randomly create our
        reservoir from. The reservoir is the pool used to replenish the
        list of goals from on .reset()

        Parameters
        ----------
        possible_goal_pos : List[np.ndarray] | np.ndarray | str
            List of possible goal positions or a string to generate random
            goal positions

        Returns
        -------
        possible_goal_pos : np.ndarray
        """

        if isinstance(possible_goal_position, str):
            if possible_goal_position.startswith("random"):
                n = int(possible_goal_position.split("_")[1])
                ext = [
                    self.extent[i : i + 2] for i in np.arange(0, len(self.extent), 2)
                ]
                possible_goal_position = [
                    np.random.random(n) * (ext[i][1] - ext[i][0]) + ext[i][0]
                    for i in range(len(ext))
                ]
                possible_goal_position = np.array(possible_goal_position).T
            else:
                raise ValueError("possible_goal_pos string must start with " "'random'")
        if isinstance(possible_goal_position, (list, np.ndarray)):
            possible_goal_position = np.array(possible_goal_position)
        # Create possible Goal
        possible_goals = [
            SpatialGoal(self, pos=pos, **self.goalkws) for pos in possible_goal_position
        ]
        return possible_goals

    def get_goal_positions(self) -> np.ndarray:
        """
        Shortcut for getting an numpy array of goal positions
        dimensions: (n_goals, n_dimensions)
        """
        return np.array(
            [
                goal.pos
                for goal in self.goal_cache.get_goals()
                if isinstance(goal, SpatialGoal)
            ]
        )

    def reset(
        self, goal_locations: Union[np.ndarray, None] = None, n_objectives=None, **kws
    ) -> dict:
        """
            reset

        resets the environement to a new episode
        """

        # How many goals to set?
        if goal_locations is not None:
            self.goal_cache.reset_n_goals = len(goal_locations)
        elif n_objectives is not None:
            self.goal_cache.reset_n_goals = n_objectives
        # Did user pass new goals?
        if goal_locations is not None:
            self.goal_cache.reset_goals = self._init_poss_goal_positions(
                goal_locations
            )

        # Reset the TaskEnvironment parent class (which picks new goals etc)
        # and returns pettingzoo.reset() required objects
        return super().reset(**kws)

    # --------------------------------------
    # Rendering
    # --------------------------------------

    def _render_matplotlib(self, goalkws=dict(), **kws):
        """
        Take existing mpl render and add spatial goals

        Parameters
        ----------
        goalkws : dict
            keyword arguments to pass to the spatial goal rendering
        """
        out = super()._render_matplotlib(**kws)
        # Render the spatial goals
        self._render_mpl_spat_goals(**goalkws)
        return out

    def _render_mpl_spat_goals(self, facecolor="red", alpha=0.1, marker="x", c="red"):
        R, fig, ax = self._get_mpl_render_cache()
        initialize = "spat_goals" not in R or len(self.goal_cache) != len(
            R["spat_goals"]
        )
        if initialize:
            R["spat_goals"] = []
            R["spat_goal_radius"] = []
            for spat_goal in self.goal_cache.get_goals():
                if not isinstance(spat_goal, SpatialGoal):
                    continue
                if self.dimensionality == "2D":
                    x, y = spat_goal().T
                else:
                    x, y = np.zeros(np.shape(spat_goal())), spat_goal()
                sg = plt.scatter(x, y, marker=marker, c=c)
                ci = plt.Circle(
                    spat_goal().ravel(),
                    spat_goal.radius,
                    facecolor=facecolor,
                    alpha=alpha,
                )
                ax.add_patch(ci)
                R["spat_goals"].append(sg)
                R["spat_goal_radius"].append(sg)
        else:
            for i, obj in enumerate(self.goal_cache.get_goals()):
                if not isinstance(obj, SpatialGoal):
                    continue
                scat = R["spat_goals"][i]
                scat.set_offsets(obj())
                ci = R["spat_goal_radius"][i]
                # TODO blitting circles?
                # ci.set_center(obj().ravel())
                # ci.set_radius(obj.radius)


# ======================== #
### HELPER FUNTIONS ########
# ======================== #


def get_goal_vector(Ag=None):
    """
    Direction vector to nearest goal
    (Primarily for testing)
    """
    import warnings

    if isinstance(Ag, Agent):
        goals = Ag.Environment.get_goal_positions()
        if (goals.shape == np.array(0)).any():
            warnings.warn(f"no goals for Agent={Ag}, emitting (0,0)")
            return np.array([0, 0])
        vecs = Ag.Environment.get_vectors_between___accounting_for_environment(
            goals, Ag.pos
        )
        if Ag.Environment.goal_cache.goalorder == "sequential":
            shortest = 0
        elif Ag.Environment.goal_cache.goalorder == "nonsequential":
            shortest = np.argmin(np.linalg.norm(vecs, axis=2))
        else:
            raise ValueError("Unknown goalorder")
        return vecs[shortest].squeeze()
    elif isinstance(Ag, list) and isinstance(Ag[0], Agent):
        agents = Ag
        agentnames = Ag[0].Environment._agentnames(Ag)
        return {name: get_goal_vector(agent) for name, agent in zip(agentnames, agents)}
    elif isinstance(Ag, dict):
        return {name: get_goal_vector(agent) for name, agent in Ag.items()}
    else:
        raise TypeError("Unknown input type")


def test_environment_loop(
    env, episodes=6, pausetime=0.0000001, speed=11.0  # pause time in plot
):
    # Prep the rendering figure
    plt.ion()
    fig, ax = env.render()
    plt.show()
    plt.pause(pausetime)

    print("Agent names: ", env.agent_names)
    while env.episode < episodes:
        # Get the direction to the goal
        dir_to_reward = {name: get_goal_vector(Ag) for name, Ag in env.Ags.items()}
        drift_velocity = {
            agent: speed
            * env.Ags[agent].speed_mean
            * (dir_to_reward / np.linalg.norm(dir_to_reward))
            for (agent, dir_to_reward) in dir_to_reward.items()
        }
        # Step the environment with actions
        observation, reward, terminate_episode, _, info = env.step(drift_velocity)
        # Render environment
        env.render()
        plt.pause(pausetime)
        # Check if we are done
        if any(terminate_episode.values()):
            print("done! reward:", reward)
            env.reset()

    return fig, ax


# ======================== #
### BASIC TESTING CODE #####
# ======================== #
# See ./tests/test_taskenv.py for more tests

active = True
if active and __name__ == "__main__":
    plt.close("all")

    #################################################################
    #                   GOAL AND REWARD OPTIONS
    #################################################################
    # Create a reward that goals emit (this is optional, there is a default
    #                                   reward, but this could be set to None)
    r1 = Reward(1, expire_clock=0.5, decay="linear", decay_knobs=[6])
    pun = Reward(-1, expire_clock=0.5, decay="linear", decay_knobs=[6], name="punish")
    r1.plot_theoretical_reward()
    # Any options for the goal objects that track whether animals satisfy a
    # goal? These can be either reward, position and radius of the goal.
    goalkws = dict(reward=r1)
    # Any options for the object that tracks all activate goals across all
    # agents?
    #   agentmode = "interact" means that one agent completing a goal completes
    #                       for all agents. "noninteract" means that each agent
    #                       must complete the goal individually.
    #   goalorder = "nonsequential" means that goals can be completed in any order
    #                           "sequential" means that goals must be completed
    #                           in the order they are set
    goalcachekws = dict(
        agentmode="noninteract",
        goalorder="nonsequential",
        reset_n_goals=2,
        verbose=False,
    )

    #################################################################
    #                   ENVIRONMENT
    #################################################################
    # Create a test environment
    env = SpatialGoalEnvironment(
        params={"dimensionality": "2D"},
        render_every=1,
        teleport_on_reset=False,
        goalkws=goalkws,
        goalcachekws=goalcachekws,
        episode_terminate_delay=2,  # seconds
        verbose=False,
    )
    #################################################################
    #                   AGENT
    #################################################################
    # Create an agent
    Ag = Agent(env)
    env.add_agents(Ag)  # add it to the environment
    Ag2 = Agent(env)
    env.add_agents(Ag2)
    #################################################################
    #################################################################

    test_environment_loop(env, episodes=4, speed=8.0)
