from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.utils import *

import copy
import numpy as np


class ValueNeuron(FeedForwardLayer):
    """
    Contributer: Tom George tomgeorge1@btinternet.com

    The ValueNeuron class defines neuron(s) which learns the "value" of a policy using TD learning. For n > 1 the reward function is assumed to be multidimensional and n value functions (one neuron for each reward function) will be learned under the current policy.  

    The true value function, V, is approximated, \hat{V}, as a non-linearly activated (default relu) linear sum of input features (i.e. one layer neural network, in later classes we may generalise this):

    V_i(x) = \int_{t}^{\infty}e^{-\frac{t^{\prime}-t}{\tau}}R_(x(t^{\prime}))) dt^{\prime} | x(t) = x    \\
    \hat{V}_i(x) \approx \psi_{nonlinearity} ( \w_{ij} \cdot \phi_j(x) )

    For this we calculate the (temporally continuous) temporal difference error and apply it to the weights:
    
    td_error(t) = R(t) + dV(t)/dt - V(t) 
    d w_i = \eta td_error(t) \z_i(t) \psi_prime(t)

    where z is the eligibility trace of the i^th feature (psi_prime accounts for the non-linearrity in the learning rule and is caluclated by the parent class. 

    Input features can be any RatInABox Neurons class here (a set of PlaceCells, BoundaryVectorCells, GridCells etc...or more complex things). It linearly sums these inputs to calculate its firing rate (this summation is all handled by the FeedForwardLayer class). Weights are trained using TD learning, self.update_weights() should be called at each update step and passed the current reward density or reward density vector. For more infor see ratinabox/example_scripts/reinforcement_learning_example/

    Since this is a Neurons subclass, after (or even during) learning you can plot the value function just by querying the ValueNeurons rate map (ValueNeuron.plot_rate_map()), or check the estimate of value at a postion using ValueNeuron.get_state(evaluate_at=None, pos=np.array([[x,y]]))

    To see the weights try ValueNeuron.inputs['name_of_input_layer']['w']
    """

    default_params = {
        "input_layer": None,  # the features it is using as inputs
        "tau": 2,  # discount time horizon (equivalent to gamma in discrete RL)
        "tau_e": None,  # eligibility trace timescale, must be <= tau (defaults to tau/2)
        "eta": 0.001,  # learning rate
        "L2": 0.001,  # L2 regularisation
        "activation_params": {"activation": "relu"},  # non-linearity for
        "n": 1,  # how many rewards there will be and thus how many Values function (each represented by one ValueNeuron) there are
    }

    def __init__(self, Agent, params={}):

        self.params = copy.deepcopy(__class__.default_params)        
        self.params.update(params)

        self.params["input_layers"] = [self.params["input_layer"]]
        super().__init__(Agent, self.params)  # initialise parent class

        if self.tau_e == None:
            self.tau_e = self.tau / 2
        self.et = np.zeros(self.n)  # initialise eligibility trace
        self.firingrate = np.zeros(self.n)  # initialise firing rate
        self.firingrate_deriv = np.zeros(self.n)  # initialise firing rate derivative
        self.td_error = np.zeros(self.n)  # initialise td error

    def update(self):
        """Updates firing rate as weighted linear sum of feature inputs"""
        firingrate_last = self.firingrate
        # update the firing rate
        super().update()  # FeedForwardLayer builtin function. This sums the inputs from the input features over the weight matrix and saves the firingrate.

        # calculate temporal derivative of the firing rate
        self.firingrate_deriv = (self.firingrate - firingrate_last) / self.Agent.dt
        # update eligibility trace
        if self.tau_e == 0:
            self.et = self.input_layer.firingrate
        else:
            self.et = (
                self.Agent.dt * self.input_layer.firingrate
                + (1 - self.Agent.dt / self.tau_e) * self.et
            )
        return

    def update_weights(self, reward):
        """Trains the weights by implementing the TD learning rule,
        reward is the vector of reward densities"""
        reward = np.array(reward).reshape(-1)
        assert len(reward) == self.n, print(
            f"Must send same number of reward signals as value neurons (n={self.n}), you sent {len(reward)}"
        )
        w = self.inputs[self.input_layer.name]["w"]  # weights
        V = self.firingrate  # current value estimate
        dVdt = self.firingrate_deriv  # currrent value derivative estimate
        self.td_error = (
            reward + dVdt - V / self.tau
        )  # this is the continuous analog of the TD error
        dw = (
            self.Agent.dt
            * self.eta
            * (np.outer(self.td_error * self.firingrate_prime, self.et))
            - self.eta * self.Agent.dt * self.L2 * w
        )  # note L2 regularisation
        if np.linalg.norm(dw) > 10:
            print(np.linalg.norm(dw))
        self.inputs[self.input_layer.name]["w"] += dw
        return

    def reset(
        self,
    ):
        """Resets the value neuron by wiping the current eligibility trace and firing rate"""
        self.et = np.zeros(self.n)  # reinitialise eligibility trace
        self.firingrate = np.zeros(self.n)  # reinitialise firing rate
        self.firingrate_deriv = np.zeros(self.n)  # reinitialise firing rate derivative
        self.td_error = np.zeros(self.n)  # reinitialise td error

        return


if __name__ == "__main__":
    """Example use.
    A reward is placed in the middle of the environment. Agent explores and learns "value" as linear sum of 100 place cells. Afterwarrd, the results are plotted. (If successful the value function should be higher in the centre of the environment (near reward) and lower at the edges))
    """
    from ratinabox.contribs.ValueNeuron import ValueNeuron
    from tqdm import tqdm

    # initialise
    Env = Environment()
    Ag = Agent(Env, params={"speed_mean": 0.1, "dt": 0.05})
    PCs = PlaceCells(
        Ag,
        params={
            "n": 100,
            "widths": 0.1,
        },
    )
    Reward = PlaceCells(
        Ag,
        params={
            "n": 1,
            "place_cell_centres": np.array([[0.5, 0.5]]),
            "description": "gaussian_threshold",
        },
    )
    VN = ValueNeuron(
        Ag,
        params={
            "input_layer": PCs,
            "tau": 1,
        },
    )

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    Reward.plot_place_cell_locations(fig=fig, ax=ax[0])
    VN.plot_rate_map(fig=fig, ax=ax[1])

    # explore/learn for 300 seconds
    for i in tqdm(range(int(1000 / Ag.dt))):
        Ag.update()
        Reward.update()
        PCs.update()
        VN.update()
        VN.update_weights(reward=Reward.firingrate[0])

    VN.plot_rate_map(fig=fig, ax=ax[2])
    Ag.plot_trajectory(fig=fig, ax=ax[3], framerate=20)
    ax[0].set_title("Location of reward")
    ax[1].set_title("Value function \n(before)")
    ax[2].set_title("Value function \n(after)")
    ax[3].set_title("Trajectory")
    plt.show()
