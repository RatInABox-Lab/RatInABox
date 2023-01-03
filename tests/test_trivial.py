import pytest

from ratinabox.Environment import *
from ratinabox.Agent import *
from ratinabox.Neurons import *
from ratinabox.utils import *


def test_pass():
    assert 1 + 1 == 2

def test_fail():
    assert 1 + 1 == 1


@pytest.fixture
def environment_agent():
    """Returns a default Environment and Agent"""
    return Environment()


def test_env_init():
    Env = Environment()

def test_agent_init():
    Env = Environment()
    Ag = Agent(Env)

def test_run_agent():
    Env = Environment()
    Ag = Agent(Env)
    for i in range(100):
        Ag.update()


