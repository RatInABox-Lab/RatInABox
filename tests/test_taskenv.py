import pytest

from types import FunctionType

from ratinabox.contribs.TaskEnvironment import SpatialGoalEnvironment, get_goal_vector
from ratinabox.Agent import Agent

import numpy as np
from pettingzoo.test import parallel_api_test

import matplotlib.pyplot as plt

plt.ion()

speed_const = 18  # dials how fast agent runs
records = []


@pytest.fixture(params=[dict(dt=dt) for dt in (0.01,)])
def Ag(request, Env):
    """
    Returns an Agent with a given dt
    """
    return Agent(Environment=Env, params=request.param)


@pytest.fixture(params=[1, 3, 10])
def Agents(request, Ag):
    """
    Returns a list of n Agents
    """
    return [Ag] * request.param


@pytest.fixture(params=[1, 2])
def reset_n_goals(request):
    """
    Returns a list of n goals
    """
    return request.param


@pytest.fixture(
    params=[
        [[0.00001, 0.00001], [0, 0.999]],  # list of lists
        np.array([[0.00001, 0.00001]]),
        "random_2",  # string
    ]
)
def possible_goal_positions(request):
    return request.param


@pytest.fixture(params=["interact", "noninteract"])
def agentmode(request):
    return request.param


@pytest.fixture(params=[{"dimensionality": "2D"}])
def Env(request, reset_n_goals, possible_goal_positions, agentmode):
    """
    Returns an Environment with a given dimensionality
    """
    env = SpatialGoalEnvironment(
        possible_goal_positions=possible_goal_positions,
        render_every=1,
        goalcachekws=dict(agentmode=agentmode, reset_n_goals=reset_n_goals),
        params=request.param,
    )
    return env


@pytest.fixture
def EnvWithAgents(Agents, Env):
    Env.add_agents(Agents)
    return Env


@pytest.fixture
def drift_velocity(EnvWithAgents):
    agents = list(EnvWithAgents.Ags.values())
    return {
        name: (val * agent.speed_mean * speed_const)
        for agent, (name, val) in zip(agents, get_goal_vector(agents).items())
    }


def test_agent_position_not_nan(Agents):
    assert not np.any(np.isnan([A.pos for A in Agents]))


def test_drift_velocity_not_nan(drift_velocity):
    assert not any((np.isnan(driftvec).any() for driftvec in drift_velocity.values()))


def test_reward_is_not_nan(EnvWithAgents, drift_velocity):
    _, reward, _, _, _ = EnvWithAgents.step(drift_velocity)
    assert not np.isnan(list(reward.values())).any()


def test_agent_can_reach_goal(
    EnvWithAgents: SpatialGoalEnvironment, less_than_steps=10_000
):
    """Test that the agent can reach the goal"""
    done = False
    steps = 0
    Env, Agents = EnvWithAgents, list(EnvWithAgents.Ags.values())
    Env.reset()
    while not done:
        if np.any(np.isnan([A.pos for A in Agents])):
            raise ValueError("Agent pos is NaN")
        action = {
            name: val * agent.speed_mean * speed_const
            for agent, (name, val) in zip(Agents, get_goal_vector(Agents).items())
        }
        if np.any(np.isnan((*action.values(),))):
            raise ValueError("Action is NaN")
        _, reward, terminate_episode, _, _ = Env.step(action)
        done = all(terminate_episode.values())
        steps += 1
        if steps > less_than_steps:
            # This should be reached before because we're taking the best
            # action
            break
    records.append(dict(steps=steps, reward=reward))
    assert all(terminate_episode.values())

# def test_parallel_api(EnvWithAgents):
#     """
#     Test that the environment fits the parallel pettingzoo API. This
#     runs the official pettingzoo test suite.
#     """
#     parallel_api_test(EnvWithAgents, num_cycles=10)
