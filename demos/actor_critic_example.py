# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:35:17 2022

@author: Daniel
"""

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells
from ratinabox.Critic import Critic, Actor
import numpy as np
import matplotlib.pyplot as plt
import sys

"""PARAMETERS AND INITIALIZATION"""
np.random.seed(100)
physical_dimension = 1  # in m
physical_resolution = 0.01  # in m
simulation_time = 10  # in s
n_place_cells = 121

Env = Environment(params = {'boundary_conditions': 'solid',
                            'dimensionality': '2D',
                            'scale': physical_dimension,
                            'dx': physical_resolution})

Ag = Agent(Env)

Ag.dt = 0.0005
Ag.sim_time = simulation_time
Ag.speed_mean = 0.08 * physical_dimension #m/s
Ag.speed_coherence_time = 0.7
Ag.rotation_velocity_std = 120 * np.pi/180 #radians 
Ag.rotational_velocity_coherence_time = 0.08

sim_range = range(int(simulation_time/Ag.dt))

PCs = PlaceCells(Ag, params={"n": n_place_cells,
                             "widths": 0.1 * physical_dimension,
                             "max_fr": 400,
                             'history_flag': False})

Reward = PlaceCells(Ag, params={'n':1,
                                 'place_cell_centres':np.array([[0.9,0.05]]),
                                 'description':'top_hat',
                                 'widths':0.5,
                                 'max_fr':1,
                                 'history_flag': False},)

critic = Critic(Ag, PCs, Reward, params={'history_flag': False})
actor = Actor(Ag, PCs, critic, params={'history_flag': False})
actions_history = []
value_history = []
td_error_history = []

"""CHECK PRETRAINED VALUE ESTIMATE AND ACTION DIRECTION"""
x_pos = y_pos = np.arange(0.1, 0.9, 0.05)

pos_check_time = 0.1  # Stay at pos for 1 second

average_value_estimate_pre = np.empty((x_pos.size, y_pos.size))
average_value_estimate_pre[:] = np.nan

average_action_selection_pre = np.empty((x_pos.size, y_pos.size, 2))
average_action_selection_pre[:] = np.nan

loc = []

print("Pre training assessment")
for idx_x, x in enumerate(x_pos):
    for idx_y, y in enumerate(y_pos):
        print(f'Position {x}, {y}')
        timer = 0
        curr_value_estimate = []
        curr_action_selection = []
        Ag.pos[:] = [x, y]
        loc.append([x, y])
        while True:
            Ag.t += Ag.dt
            PCs.update()
            actor.update()
            critic.update()
            Reward.update()
            curr_value_estimate.append(critic.value_estimate())
            curr_action_selection.append(actor.action_selection())
            timer += Ag.dt

            if timer > pos_check_time:
                break

        average_value_estimate_pre[idx_x, idx_y] = np.array(curr_value_estimate).mean()
        average_action_selection_pre[idx_x, idx_y] = np.array(curr_action_selection).mean(axis=0)

"""SIMULATE TRAJECTORY"""
"""REINITIALIZE OBJECTS TO BE SAFE"""
np.random.seed(100)

Env = Environment(params = {'boundary_conditions': 'solid',
                            'dimensionality': '2D',
                            'scale': physical_dimension,
                            'dx': physical_resolution})

Ag = Agent(Env)

Ag.dt = 0.0005
Ag.sim_time = simulation_time
Ag.speed_mean = 0.08 * physical_dimension #m/s
Ag.speed_coherence_time = 0.7
Ag.rotation_velocity_std = 120 * np.pi/180 #radians 
Ag.rotational_velocity_coherence_time = 0.08

sim_range = range(int(simulation_time/Ag.dt))

PCs = PlaceCells(Ag, params={"n": n_place_cells,
                             "widths": 0.1 * physical_dimension,
                             "max_fr": 400})

Reward = PlaceCells(Ag, params={'n':1,
                                 'place_cell_centres':np.array([[0.9,0.05]]),
                                 'description':'top_hat',
                                 'widths':0.5,
                                 'max_fr':1,})

critic = Critic(Ag, PCs, Reward)
actor = Actor(Ag, PCs, critic)
action = (0, 0)
actions_history = []
value_history = []
td_error_history = []

"""RUN SIMULATION"""
print("Start trajectory simulation")
for i in sim_range:
    Ag.update(drift_velocity=action, drift_to_random_strength_ratio=100)
    PCs.update()
    actor.update()
    critic.update()
    Reward.update()
    value_history.append(critic.value_estimate())
    action = actor.action_selection()
    actions_history.append(action)
    td_error_history.append(critic.td_error())
    critic.r_ltp()
    actor.r_ltp()

critic_input_trained = critic.W_input

"""PLOTTING TRAJECTORY RELATED"""
fig = plt.figure()
gs = fig.add_gridspec(2, 4)
ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
ax2 = fig.add_subplot(gs[0, 1], aspect='equal')
ax3 = fig.add_subplot(gs[0, 2], aspect='equal')
ax4 = fig.add_subplot(gs[0, 3], aspect='equal')
ax5 = fig.add_subplot(gs[1, :2])
ax6 = fig.add_subplot(gs[1, 2], aspect='equal')
ax7 = fig.add_subplot(gs[1, 3], aspect='equal')

fig, ax = Ag.plot_trajectory(ax=ax1)
Reward.plot_rate_map(ax=ax2)

ax5.plot(np.array(critic.history['firing_rates'])[:, 0])
ax5.plot(np.array(critic.history['spikes'])[:, 0])
ax5.plot(critic.history['epsps'][:, 0, 3])
ax5.legend(("Rate", "Spikes", "Single Cell EPSP"))

"""CHECK TRAINED VALUE ESTIMATE AND ACTION DIRECTION"""
np.random.seed(100)

Env = Environment(params = {'boundary_conditions': 'solid',
                            'dimensionality': '2D',
                            'scale': physical_dimension,
                            'dx': physical_resolution})

Ag = Agent(Env)

Ag.dt = 0.0005
Ag.sim_time = simulation_time
Ag.speed_mean = 0.08 * physical_dimension #m/s
Ag.speed_coherence_time = 0.7
Ag.rotation_velocity_std = 120 * np.pi/180 #radians 
Ag.rotational_velocity_coherence_time = 0.08

sim_range = range(int(simulation_time/Ag.dt))

PCs = PlaceCells(Ag, params={"n": n_place_cells,
                             "widths": 0.1 * physical_dimension,
                             "max_fr": 400,
                             'history_flag': False})

Reward = PlaceCells(Ag, params={'n':1,
                                 'place_cell_centres':np.array([[0.9,0.05]]),
                                 'description':'top_hat',
                                 'widths':0.5,
                                 'max_fr':1,
                                 'history_flag': False})

critic = Critic(Ag, PCs, Reward, params={'history_flag': False})
actor = Actor(Ag, PCs, critic, params={'history_flag': False})
critic.W_input = critic_input_trained

x_pos = y_pos = np.arange(0.1, 0.9, 0.05)

pos_check_time = 0.1  # Stay at pos for 1 second

average_value_estimate_post = np.empty((x_pos.size, y_pos.size))
average_value_estimate_post[:] = np.nan

average_action_selection_post = np.empty((x_pos.size, y_pos.size, 2))
average_action_selection_post[:] = np.nan

print("Pre training assessment")
for idx_x, x in enumerate(x_pos):
    for idx_y, y in enumerate(y_pos):
        print(f'Position {x}, {y}')
        curr_value_estimate = []
        Ag.pos[:] = [x, y]
        timer = 0
        while True:
            Ag.t += Ag.dt
            PCs.update()
            actor.update()
            critic.update()
            Reward.update()

            timer += Ag.dt

            curr_value_estimate.append(critic.value_estimate())
            curr_action_selection.append(actor.action_selection())

            if timer > pos_check_time:
                break 
        average_value_estimate_post[idx_x, idx_y] = np.array(curr_value_estimate).mean()
        average_action_selection_post[idx_x, idx_y] = np.array(curr_action_selection).mean(axis=0)

loc = np.array(loc)

ax3.quiver(loc[:, 0], loc[:, 1], average_action_selection_pre.reshape(16*16,2)[:,0], average_action_selection_pre.reshape(16*16,2)[:,1])
ax3.set_title("Pre Training Actor")
ax4.quiver(loc[:, 0], loc[:, 1], average_action_selection_post.reshape(16*16,2)[:,0], average_action_selection_post.reshape(16*16,2)[:,1])
ax4.set_title("Post Train Actor")

ax6.imshow(average_value_estimate_pre.T, origin='lower')
ax6.set_title("Pre training critic")
ax7.imshow(average_value_estimate_post.T, origin='lower')
ax7.set_title("Post training critic")

"""
plt.figure()
plt.quiver(loc[:, 0], loc[:, 1], average_action_selection_pre.reshape(16*16,2)[:,0], average_action_selection_pre.reshape(16*16,2)[:,1])
plt.figure()
plt.quiver(loc[:, 0], loc[:, 1], average_action_selection_post.reshape(16*16,2)[:,0], average_action_selection_post.reshape(16*16,2)[:,1])



plt.figure()
plt.imshow(average_value_estimate_pre.T, origin='lower')
plt.figure()
plt.imshow(average_value_estimate_post.T, origin='lower')
"""

"""
chosen_neurons = [69,65,89,91,57]

plt.figure()
plt.plot(np.array(actor.history['u'])[2000, :])
plt.plot(np.array(actor.history['u'])[6000, :])
plt.plot(np.array(actor.history['u'])[8000, :])
plt.legend(("1s", "3s", "4s"))

plt.figure()


fig, ax = PCs.plot_rate_timeseries(chosen_neurons=chosen_neurons)

PCs.plot_rate_map(chosen_neurons=chosen_neurons)


fig, ax = plt.subplots(5)
# for idx, pc in enumerate(np.random.randint(low=0, high=121, size=5)):
t = np.array(PCs.history['t']) 
for idx, pc in enumerate([69,65,89,91,57]):
    ax[idx].plot(t, np.array(PCs.history['firingrate'])[:, pc])
    ax[idx].set_ylim([0, 450])
    ax[idx].fill_between(t, np.array(PCs.history['firingrate'])[:, pc])

"""