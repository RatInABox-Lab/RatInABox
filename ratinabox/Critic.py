# -*- coding: utf-8 -*-
"""
Contains the implementation of the different neurons.

The place cells used are defined in RatInABox.
"""

from ratinabox.Neurons import Neurons
import numpy as np
import pdb
import numpy.matlib

class SRM0(Neurons):
    def __init__(self, Agent, external_input, params={}):
        default_params = {
            "N_out": 40,
            "name": "SRM0",
            "tau_m": 20.0,
            "tau_s": 5.0,
            "eps0": 20.0,
            "chi": -5.0,
            "rho": 0.06,
            "theta": 16,
            "delta_u": 5,
            "tau_gamma": 50,
            "v_gamma": 20,
            'history_flag': True,
        }
        super().__init__(Agent)  # Run the init function of the parent class

        # Change the default parameters according to user provided params            
        default_params.update(params)      
        self.params = default_params
        
        # Unpack all the parameters into self
        for k, v in self.params.items():
            setattr(self, k, default_params[k])

        self.Ag = Agent
        self.external_input = external_input
        self.N_in = self.external_input.n  # How man

        # Setup history buffers that are Actor specific        
        self.history["u"] = []
        self.history["smooth_firing_rates"] = []
        self.history["firing_rates"] = np.empty((int(self.Ag.sim_time/self.Ag.dt), self.N_out))
        self.history["firing_rates"][:] = np.nan
        self.history['epsps'] = np.empty((int(self.Ag.sim_time/self.Ag.dt), self.N_out, self.N_in))
        self.history['epsps'][:] = np.nan

        if self.smooth_firing:
            self.rho_decay = np.zeros(self.N_out)
            self.rho_rise =  np.zeros(self.N_out)

        # Initialize spikes
        self.spikes = np.zeros(self.N_out)

        # Weights from place_input layer to actor
        # Mean and scale from original paper
        self.W_input = np.random.normal(0.5, 0.1, (self.N_out, self.external_input.n))
        self.W_input_initial = self.W_input.copy()

        """ACTION SPACE"""
        self.theta_actor = np.reshape(2*np.pi*np.arange(1,self.N_out+1)/self.N_out,(self.N_out,1))
        self.action_space = np.squeeze(self.a0*np.array([np.sin(self.theta_actor),np.cos(self.theta_actor)]))

        self.epsp_decay = np.zeros((self.N_out, self.external_input.n))
        self.epsp_rise = np.zeros((self.N_out, self.external_input.n))
        
        self.epsp_decay_no_reset = np.zeros((self.N_out, self.external_input.n))
        self.epsp_rise_no_reset = np.zeros((self.N_out, self.external_input.n))
        
        # Initialize the last spike
        self.last_spike_time = np.zeros(self.N_out) - 1
        self.last_spike_idx = np.zeros(self.N_out, dtype=int)

    def get_state(self):
        time_scaling_constant = 1000  # Agent time base is in seconds
        time = self.Ag.t * time_scaling_constant

        spikes_pre = np.array(self.external_input.history['spikes'][-1], dtype=int)  # Get the most recent spikes

        cat_spikes = spikes_pre
        cat_weights = self.W_input

        self.epsp_decay = self.epsp_decay - self.epsp_decay/self.tau_m + np.multiply(cat_spikes,cat_weights)
        self.epsp_rise =  self.epsp_rise  - self.epsp_rise/self.tau_s + np.multiply(cat_spikes,cat_weights)

        self.epsp_decay_no_reset = self.epsp_decay_no_reset - self.epsp_decay_no_reset/self.tau_m + np.multiply(cat_spikes,cat_weights)
        self.epsp_rise_no_reset =  self.epsp_rise_no_reset  - self.epsp_rise_no_reset/self.tau_s + np.multiply(cat_spikes,cat_weights)

        self.EPSP = (self.eps0/(self.tau_m-self.tau_s))*(self.epsp_decay-self.epsp_rise)
        self.EPSP_no_reset = (self.eps0/(self.tau_m-self.tau_s))*(self.epsp_decay_no_reset-self.epsp_rise_no_reset)

        self.u = self.EPSP.sum(axis=1) + self.chi*np.exp((-time + self.last_spike_time)/self.tau_m)
        
        self.firing_rates = self.rho*np.exp((self.u-self.theta)/self.delta_u)
        self.spikes = np.random.rand(self.N_out) <= self.firing_rates #realization spike train
    
        self.last_spike_time[self.spikes] = time #update time postsyn spike
        self.last_spike_idx[:] = self.last_spike_time // (self.Ag.dt * time_scaling_constant)
        self.epsp_decay[self.spikes] = 0
        self.epsp_rise[self.spikes] = 0

        self.rho_decay = self.rho_decay - self.rho_decay/self.tau_gamma + self.spikes
        self.rho_rise =  self.rho_rise -  self.rho_rise/self.v_gamma + self.spikes
        self.smooth_firing_rates = (self.rho_decay - self.rho_rise)*(1./(self.tau_gamma - self.v_gamma))

        if self.history_flag:
            self.history['epsps'][int(self.Ag.t / self.Ag.dt) - 1] = self.EPSP_no_reset
            self.history['firing_rates'][int(self.Ag.t / self.Ag.dt) - 1] = self.smooth_firing_rates

        return self.firing_rates, self.u, self.spikes


    def update(self):
        state = self.get_state()
        self.u = state[1]
        self.firingrate = state[0].reshape(-1)
        if self.history_flag:
            self.save_to_history()
    
    def save_to_history(self):
        cell_spikes = self.spikes
        self.history["t"].append(self.Ag.t)
        self.history["firingrate"].append(list(self.firingrate))
        self.history["spikes"].append(list(cell_spikes))
        self.history["smooth_firing_rates"].append(list(self.smooth_firing_rates))
        self.history['u'].append(self.u)


class Actor(SRM0):
    """An Actor layer as in the Actor-Critic framework. Based on a spike
    response model."""
    def __init__(self, Agent, external_input, Critic, params={}):
        """Actor network based on the SRM0 model.
        The `Agent` handles information about the position in the environment
        and actions that change the position. `place_input` relays the
        neuronal code of the environmental information. `params` contains
        all the parameters defining the actor network. Time basis of the actor
        is milliseconds.
        
        Parameters
        ---------
        N_out : int
            Number of actor neurons.
        name : str
            Desctiptive name for the actor population.
        tau_m : float
            Membrane time constant of the SRM0 model in milliseconds.
        tau_s : float
            Synaptic rise time constant in milliseconds.
        eps0 : float
            Synaptic scaling constant in millivolt * milliseconds
        chi : float
            Scaling factor for the refractory effect of the SRM0 in millivolt.
        rho : float
            Scaling constant for firing rate in events / ms
        theta : float
            Spiking threshold in millivolt.
        delta_u : float
            Spiking probability scaling constant in millivolt.
        tau_gamma : float
            Time constant to smoothen firing rate decay. Action selection
            is performed on smooth firing rate rather than instananeous.
        v_gamma : float
            Time constant to smoothen firing rate rise. Action selection
            is performed on smooth firing rate rather than instananeous.
        w_min : float
            TODO Minimum weight of inputs.
        w_max : float
            TODO Maximum weight of inputs.
        psi : float 
            Width of lateral connectivity.
        w_minus : float
            TODO
        w_plus : float
            TODO
        smooth_firing : boolean
            TODO
        lateral_weights : float
            Scale for strength of lateral weights. To disable set to zero.
        a0 : float
        TODO
        """
        default_params = {
            "N_out": 180,
            "name": "Actor",
            "tau_m": 20.0,
            "tau_s": 5.0,
            "eps0": 20.0,
            "chi": -5.0,
            "rho": 0.06,
            "theta": 16,
            "delta_u": 5,
            "tau_gamma": 50,
            "v_gamma": 20,
            "w_min": 0,
            "w_max": 3,
            "psi": 20,
            "w_minus": -300,
            "w_plus": 100,
            "smooth_firing": True,
            "lateral_weights": 1,
            "a0": 8,
            'history_flag': True,
        }
        default_params.update(params)      
        self.params = default_params
        
        super().__init__(Agent, external_input, self.params)  # Run the init function of the parent class


        """Add the lateral weights that make the actor a bump attractor"""
        diff_theta = np.matlib.repmat(self.theta_actor.T, self.N_out, 1) - np.matlib.repmat(self.theta_actor, 1, self.N_out)
        f = np.exp(self.psi * np.cos(diff_theta))  # lateral connectivity function
        f = f - np.multiply(f, np.eye(self.N_out))
        normalised = np.sum(f, axis=0)
        w_lateral = (self.w_minus / self.N_out + self.w_plus * (f / normalised))
        np.fill_diagonal(w_lateral, 0)
        self.W_lateral = w_lateral * self.lateral_weights

        self.epsp_decay = np.zeros((self.N_out, self.external_input.n + self.N_out))
        self.epsp_rise = np.zeros((self.N_out, self.external_input.n + self.N_out))
        
        # Initialize the last spike
        # self.last_spike_time = np.zeros(self.N_out) - 1
        self.last_spike_time = np.zeros(self.N_out)
        self.last_spikeidx = np.zeros(self.N_out)
        
        self.history["epsps"] = np.zeros(((int(Agent.sim_time / Agent.dt)), self.N_out, self.N_in))
        
        self.critic = Critic


    def action_selection(self):
        """Select an action in x, y space from the actors firing rates"""
        return np.dot(self.smooth_firing_rates.T, np.squeeze(self.action_space).T)/self.N_out


    def get_state(self):
        time_scaling_constant = 1000  # Agent time base is in seconds
        time = self.Ag.t * time_scaling_constant

        spikes_pre = np.array(self.external_input.history['spikes'][-1], dtype=int)  # Get the most recent spikes
        if self.W_lateral is not None:
            cat_spikes = np.concatenate([spikes_pre, self.spikes])
            cat_weights = np.concatenate([self.W_input, self.W_lateral], axis=1)
        else:
            cat_spikes = spikes_pre
            cat_weights = self.W_input

        self.epsp_decay = self.epsp_decay - self.epsp_decay/self.tau_m + np.multiply(cat_spikes,cat_weights)
        self.epsp_rise =  self.epsp_rise  - self.epsp_rise/self.tau_s + np.multiply(cat_spikes,cat_weights)
        
        self.epsp_decay_no_reset = self.epsp_decay_no_reset - self.epsp_decay_no_reset/self.tau_m + np.multiply(spikes_pre,self.W_input)
        self.epsp_rise_no_reset =  self.epsp_rise_no_reset  - self.epsp_rise_no_reset/self.tau_s + np.multiply(spikes_pre,self.W_input)

        self.EPSP = (self.eps0/(self.tau_m-self.tau_s))*(self.epsp_decay-self.epsp_rise)
        self.EPSP_no_reset = (self.eps0/(self.tau_m-self.tau_s))*(self.epsp_decay_no_reset-self.epsp_rise_no_reset)
        
        self.u = self.EPSP.sum(axis=1) + self.chi*np.exp((-time + self.last_spike_time)/self.tau_m)
        
        self.firing_rates = self.rho*np.exp((self.u-self.theta)/self.delta_u)
        self.spikes = np.random.rand(self.N_out) <= self.firing_rates #realization spike train
    
        self.last_spike_time[self.spikes] = time #update time postsyn spike
        self.last_spike_idx = self.last_spike_time / (self.Ag.dt * time_scaling_constant)
        
        self.epsp_decay[self.spikes] = 0
        self.epsp_rise[self.spikes] = 0

        self.rho_decay = self.rho_decay - self.rho_decay/self.tau_gamma + self.spikes
        self.rho_rise =  self.rho_rise -  self.rho_rise/self.v_gamma + self.spikes
        self.smooth_firing_rates = (self.rho_decay - self.rho_rise)*(1./(self.tau_gamma - self.v_gamma))
        
        if self.history_flag:
            self.history['epsps'][int(self.Ag.t / self.Ag.dt) - 1,:,:self.N_in] = self.EPSP_no_reset
            self.history['firing_rates'][int(self.Ag.t / self.Ag.dt) - 1] = self.smooth_firing_rates
        
        return self.firing_rates, self.u, self.spikes
    
    def r_ltp(self):
        """Reward based LTP TD error times coincidence."""
        """
        for i in range(self.N_out):
            for j in range(self.N_in):
                lsi = self.last_spike_idx[i]
                coincidence = np.nanmean(self.history['firing_rates'][lsi:, i] * self.history['epsps'][lsi:, i, j])
                self.W_input[i, j] += self.final_learning_rate * self.td_error() * coincidence
        """
        coincidence = self.smooth_firing_rates[None, :] @ self.EPSP_no_reset[:,:self.N_in]
        self.W_input += self.critic.final_learning_rate * -self.critic.td_error() * coincidence * self.critic.discount

class Critic(SRM0):
    """An Actor layer as in the Actor-Critic framework. Based on a spike
    response model."""
    def __init__(self, Agent, external_input, Reward, params={}):
        default_params = {
            "N_out": 100,
            "name": "Critic",
            "tau_m": 20,
            "tau_s": 5,
            "eps0": 20,
            "chi": -5,
            "rho": 0.06,
            "theta": 16,
            "delta_u": 5,
            "tau_gamma": 200,
            "v_gamma": 50,
            "w_min": 0,
            "w_max": 3,
            "psi": 20,
            "w_minus": -300,
            "w_plus": 100,
            "smooth_firing": True,
            "lateral_weights": 0,
            "a0": 8,
            "value_baseline": - 0.01,
            "value_scale": 1,
            "discount": 0.9,
            "learning_rate": 0.01,
            'history_flag': True,
        }

        default_params.update(params)
        params = default_params
        
        super().__init__(Agent, external_input, params)
    
        self.Reward = Reward
        self.final_learning_rate = (self.learning_rate * self.value_scale) / (self.N_out * self.delta_u)

        self.history["epsps"] = np.zeros(((int(Agent.sim_time / Agent.dt)), self.N_out, self.N_in))

    def value_estimate(self):
        """The value estimate is simply an average of the critic population activity."""
        return (self.value_scale * self.firingrate.sum())  / self.N_out + self.value_baseline

    def td_error(self):
        """The temporal difference (TD) error is the differensce between the
        value estimate and the actually delivered reward."""
        return self.value_estimate() - self.Reward.firingrate

    def r_ltp(self):
        """Reward based LTP TD error times coincidence."""
        """
        for i in range(self.N_out):
            for j in range(self.N_in):
                lsi = self.last_spike_idx[i]
                coincidence = np.nanmean(self.history['firing_rates'][lsi:, i] * self.history['epsps'][lsi:, i, j])
                self.W_input[i, j] += self.final_learning_rate * self.td_error() * coincidence
        """
        coincidence = self.smooth_firing_rates[None, :] @ self.EPSP_no_reset
        self.W_input += self.final_learning_rate * -self.td_error() * coincidence * self.discount
        
