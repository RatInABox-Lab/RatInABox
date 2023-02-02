import pytest

import ratinabox
import matplotlib
import numpy
from ratinabox.Environment import Environment

@pytest.fixture
def Env1D():
    return Environment(params={'dimensionality':'1D'})

@pytest.fixture
def Env2D():
    return Environment(params={'dimensionality':'2D'})

def test_env_init(Env1D,Env2D):
    assert type(Env1D) == Environment
    assert type(Env2D) == Environment

def test_end_add_wall(Env2D):
    n_walls = len(Env2D.walls)
    Env2D.add_wall([[0.2,0.2],[0.2,0.2]])
    assert(len(Env2D.walls) == n_walls+1)

def test_plot_environment(Env1D,Env2D):
    fig1, ax1 = Env1D.plot_environment()
    fig2, ax2 = Env2D.plot_environment()
    assert(type(fig1) == matplotlib.figure.Figure)   
    assert(type(fig2) == matplotlib.figure.Figure)

def test_sample_positions(Env1D,Env2D):
    pos_list = []
    for method_ in ['uniform','random','uniform_random']:
        for Env in [Env1D,Env2D]:
            pos_list.append(Env.sample_positions(5,method=method_))
    assert all([pos.shape[0] == 5 for pos in pos_list])

def test_discretise_environment(Env1D,Env2D):
    coords1 = Env1D.discretise_environment(dx=0.01)
    coords2 = Env2D.discretise_environment(dx=0.01)
    assert all([type(c) is numpy.ndarray for c in [coords1,coords2]])