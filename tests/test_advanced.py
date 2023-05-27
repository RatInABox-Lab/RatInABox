# TO DO: write more detailed tests 
# For now, test_advanced will simply try to run versions of the "simple", "extensive" and "decoding_position" demo scripts. Assuming no error is raised the test has passed. It will also test whether external data (Sargolini) can be imported to test data availability. 
# This is an okay safety net for now but should be improved in future versions

#this test requires scikitlearn 

import pytest
import numpy as np 
import matplotlib.pyplot as plt 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge


def test_simple():
    Env = Environment()
    Ag = Agent(Env)
    PCs = PlaceCells(Ag)
    for i in range(int(60/Ag.dt)):
        Ag.update()
        PCs.update()

    print("Timestamps:",Ag.history['t'][:10],"\n")
    print("Positions:",Ag.history['pos'][:10],"\n")
    print("Firing rate timeseries:",PCs.history['firingrate'][:10],"\n")
    print("Spikes:",PCs.history['spikes'][:10],"\n")

    Ag.plot_trajectory()
    PCs.plot_rate_timeseries()

    return

def test_extensive():
    # 1 Initialise environment.
    Env = Environment(
    params = {'aspect':2,
               'scale':1})

    # 2 Add walls. 
    Env.add_wall([[1,0],[1,0.35]])
    Env.add_wall([[1,0.65],[1,1]])

    # 3 Add Agent.
    Ag = Agent(Env)
    Ag.pos = np.array([0.5,0.5])
    Ag.speed_mean = 0.2

    # 4 Add place cells. 
    PCs = PlaceCells(Ag,
                    params={'n':20,
                            'description':'gaussian_threshold',
                            'widths':0.40,
                            'wall_geometry':'line_of_sight',
                            'max_fr':10,
                            'min_fr':0.1,
                            'color':'C1'})
    PCs.place_cell_centres[-1] = np.array([1.1,0.5])

    # 5 Add boundary vector cells.
    BVCs = BoundaryVectorCells(Ag,
                    params = {'n':10,
                            'color':'C2'})

    # 6 Simulate. 
    dt = 50e-3 
    T = 5*60
    for i in range(int(T/dt)):
        Ag.update(dt=dt)
        PCs.update()
        BVCs.update()

    # 7 Plot trajectory. 
    fig, ax = Ag.plot_position_heatmap()
    fig, ax = Ag.plot_trajectory(t_start=50,t_end=60,fig=fig,ax=ax)

    # 8 Plot timeseries. 
    fig, ax = BVCs.plot_rate_timeseries(t_start=0,t_end=60,chosen_neurons='12',spikes=True)

    # 9 Plot place cells. 
    fig, ax = PCs.plot_place_cell_locations()

    # 10 Plot rate maps. 
    fig, ax = PCs.plot_rate_map(chosen_neurons='3',method='groundtruth')
    fig, ax = PCs.plot_rate_map(chosen_neurons='3',method='history',spikes=True)

    # 11 Display BVC rate maps and polar receptive fields
    fig, ax = BVCs.plot_rate_map(chosen_neurons='2')
    fig, ax = BVCs.plot_BVC_receptive_field(chosen_neurons='2')

    # 12 Multipanel figure 
    fig, axes = plt.subplots(2,8,figsize=(24,6))
    Ag.plot_trajectory(t_start=0, t_end=60,fig=fig,ax=axes[0,0])
    axes[0,0].set_title("Trajectory (last minute)")
    Ag.plot_position_heatmap(fig=fig,ax=axes[1,0])
    axes[1,0].set_title("Full trajectory heatmap")
    PCs.plot_rate_timeseries(t_start=0,t_end=60,chosen_neurons='6',spikes=True,fig=fig, ax=axes[0,1])
    axes[0,1].set_title("Place cell activity")
    axes[0,1].set_xlabel("")
    BVCs.plot_rate_timeseries(t_start=0,t_end=60,chosen_neurons='6',spikes=True,fig=fig, ax=axes[1,1])
    axes[1,1].set_title("BVC activity")
    PCs.plot_rate_map(chosen_neurons='6',method='groundtruth',fig=fig,ax=axes[0,2:])
    axes[0,2].set_title("Place cell receptive fields")
    BVCs.plot_rate_map(chosen_neurons='6',method='groundtruth',fig=fig,ax=axes[1,2:])
    axes[1,2].set_title("BVC receptive fields")



def test_decoding_position():
    def train_decoder(Neurons,t_start=None,t_end=None):
        """t_start and t_end allow you to pick the poritions of the saved data to train on."""
        #Get training data
        t = np.array(Neurons.history['t'])
        if t_start is None: i_start = 0
        else: i_start = np.argmin(np.abs(t-t_start))
        if t_end is None: i_end = -1
        else: i_end = np.argmin(np.abs(t-t_end))
        t = t[i_start:i_end][::5] #subsample data for training (most of it is redundant anyway)
        fr = np.array(Neurons.history['firingrate'])[i_start:i_end][::5]
        pos = np.array(Neurons.Agent.history['pos'])[i_start:i_end][::5]
        #Initialise and fit model
        from sklearn.gaussian_process.kernels import RBF
        model_GP = GaussianProcessRegressor(alpha=0.01, kernel=RBF(1
        *np.sqrt(Neurons.n/20), #<-- kernel size scales with typical input size ~sqrt(N)
        length_scale_bounds="fixed"
        ))
        model_LR = Ridge(alpha=0.01)
        model_GP.fit(fr,pos)    
        model_LR.fit(fr,pos)    
        #Save models into Neurons class for later use
        Neurons.decoding_model_GP = model_GP
        Neurons.decoding_model_LR = model_LR
        return 

    def decode_position(Neurons,t_start=None,t_end=None):
        """t_start and t_end allow you to pick the poritions of the saved data to train on.
        Returns a list of times and decoded positions"""
        #Get testing data
        t = np.array(Neurons.history['t'])
        if t_start is None: i_start = 0
        else: i_start = np.argmin(np.abs(t-t_start))
        if t_end is None: i_end = -1
        else: i_end = np.argmin(np.abs(t-t_end))
        t = t[i_start:i_end]
        fr = np.array(Neurons.history['firingrate'])[i_start:i_end]
        #decode position from the data and using the decoder saved in the  Neurons class 
        decoded_position_GP = Neurons.decoding_model_GP.predict(fr)
        decoded_position_LR = Neurons.decoding_model_LR.predict(fr)
        return (t, decoded_position_GP, decoded_position_LR)

    Env = Environment()
    Env.add_wall(np.array([[0.4,0],[0.4,0.4]]))
    Ag = Agent(Env, params={'dt':50e-3})

    PCs = PlaceCells(Ag,params={'description':'gaussian_threshold','widths':0.4,'n':20,'color':'C1'})
    GCs = GridCells(Ag,params={'n':20,'color':'C2'},)

    for i in range(int(5*60/Ag.dt)):
        Ag.update()
        PCs.update()
        GCs.update()

    fig_t, ax_t = Ag.plot_trajectory(alpha=0.5)

    fig, ax = PCs.plot_rate_map(chosen_neurons='all')
    fig, ax = GCs.plot_rate_map(chosen_neurons='all')

    train_decoder(PCs)
    train_decoder(GCs)

    np.random.seed(10)
    for i in range(int(60/Ag.dt)):
        Ag.update()
        PCs.update()
        GCs.update()

    fig_t, ax_t = Ag.plot_trajectory(fig=fig_t, ax=ax_t,t_start=Ag.t-60,color='black',alpha=0.5)

    t, pos_PCs_GP, pos_PCs_LR = decode_position(PCs,t_start=Ag.t-60)
    t, pos_GCs_GP, pos_GCs_LR = decode_position(GCs,t_start=Ag.t-60)

    fig, ax = plt.subplots(2,2,figsize=(8,8))
    Ag.plot_trajectory(t_start=Ag.t-60,fig=fig, ax=ax[0,0],color='black',alpha=0.5)
    ax[0,0].scatter(pos_PCs_GP[:,0],pos_PCs_GP[:,1],s=5,c='C1',alpha=0.2,zorder=3.1)
    Ag.plot_trajectory(t_start=Ag.t-60,fig=fig, ax=ax[1,0],color='black', alpha=0.5)
    ax[1,0].scatter(pos_PCs_LR[:,0],pos_PCs_LR[:,1],s=5,c='C1',alpha=0.2,zorder=3.1)
    ax[0,0].set_title("Place cells")

    Ag.plot_trajectory(t_start=Ag.t-60,fig=fig, ax=ax[0,1],color='black', alpha=0.5)
    ax[0,1].scatter(pos_GCs_GP[:,0],pos_GCs_GP[:,1],s=5,c='C2',alpha=0.2,zorder=3.1)
    Ag.plot_trajectory(t_start=Ag.t-60,fig=fig, ax=ax[1,1],color='black', alpha=0.5)
    ax[1,1].scatter(pos_GCs_LR[:,0],pos_GCs_LR[:,1],s=5,c='C2',alpha=0.2,zorder=3.1)
    ax[0,1].set_title("GAUSSIAN PROCESSS REGRESSION\n\nGrid cells")
    ax[1,1].set_title("LINEAR REGRESSION")


def test_data_importing():
    Env = Environment()
    Ag = Agent(Env)
    Ag.import_trajectory(dataset="sargolini")
    assert Ag.use_imported_trajectory == True