import pytest

from types import FunctionType

from ratinabox.contribs.TaskEnvironment import SpatialGoalEnvironment
from ratinabox.Agent import Agent

import numpy as np
import pandas as pd

records = pd.DataFrame()

@pytest.fixture(params=[dict(dt=dt) for dt in (0.01,)])
def Ag(request, Env):
    """
    Returns an Agent with a given dt
    """
    return Agent(Environment=Env, params=request.param )


@pytest.fixture(params=[1,3,10])
def Agents(request, Ag):
    """
    Returns a list of n Agents
    """
    return [Ag] * request.param

@pytest.fixture(params=[1,3,5])
def n_goals(request):
    """
    Returns a list of n goals
    """
    return request.param

@pytest.fixture(params=[ 
                        [[0.00001,0.00001],[0,0.999]],   # list of lists
                        np.array([0.00001,0.00001]),
                        "random_5",       # string
                        ]) 
def possible_goal_pos(request):
    return request.param


@pytest.fixture(params=[
    {"dimensionality":"1D"},{"dimensionality":"2D"}])
def Env(request, n_goals, possible_goal_pos):
    """
    Returns an Environment with a given dimensionality
    """
    env = SpatialGoalEnvironment( possible_goal_pos=possible_goal_pos, render_every=1, n_goals=n_goals, params=request.param)
    return env

@pytest.fixture
def EnvWithAgents(Agents, Env):
    Env.add_agents(Agents)
    return Env

def best_drift_velocity(Ag:Agent, Env:SpatialGoalEnvironment):
    # Define some helper functions
    get_goal_vector = lambda: \
            Env.get_goals()[0][0] - np.array([A.pos for A in Ag])
    get_goal_distance = lambda: \
        np.linalg.norm(get_goal_vector(), axis=1)
    dir_to_reward = get_goal_vector()
    drift_velocity = 2 * np.array([A.speed_mean for A in Ag])[...,None] * \
            (dir_to_reward / np.linalg.norm(dir_to_reward,axis=1)[...,None])
    return drift_velocity

@pytest.fixture
def drift_velocity(EnvWithAgents):
    return best_drift_velocity(EnvWithAgents.Agents, EnvWithAgents)

def test_agent_position_not_nan(Agents):
    assert not np.any(np.isnan([A.pos for A in Agents]))

def test_drift_velocity_not_nan(drift_velocity):
    assert not np.any(np.isnan(drift_velocity))

def test_agent_can_reach_goal(EnvWithAgents: SpatialGoalEnvironment,
                              less_than_steps=5000):
    """ Test that the agent can reach the goal """
    done = False
    steps = 0
    while not done:
        if np.any(np.isnan([A.pos for A in EnvWithAgents.Agents])):
            raise ValueError("Agent pos is NaN")
        action = best_drift_velocity(EnvWithAgents.Agents, EnvWithAgents)
        if np.any(np.isnan(action)):
            raise ValueError("Action is NaN")
        _, reward, done, _ = EnvWithAgents.step(action)
        steps += 1
        if steps > less_than_steps:
            # This should be reached before because we're taking the best
            # action
            break
    records.append(dict(steps=steps, reward=reward), ignore_index=True)
    assert all([done, all(np.abs(reward) > 0)])
    

