import pytest

from ratinabox.Environment import *
from ratinabox.Agent import *
from ratinabox.Neurons import *
from ratinabox.utils import *


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
