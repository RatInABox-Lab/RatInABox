from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.contribs.ValueNeuron import ValueNeuron
from ratinabox.utils import *

import copy
import numpy as np
import warnings


class SuccessorFeatures(ValueNeuron):
    """
    Contributer: Tom George tomgeorge1@btinternet.com

    The SuccessorFeatures class defines neuron(s) which learns the "successor features" of a set of features {\phi_i(x)}, using TD learning. This is a special case of the ValueNeuron class where the reward function is the features (thus the value function being learned are the successor features). Successor features are approximated, {\hat{\psi}_i}, as a non-linearly activated (default relu) linear sum of a set of "basis features", {f_i(x)}.

    Note the basis features are not the same as the input features, they are a set of basis functions (e.g. place cells, grid cells, boundary vector cells etc...) which are used to represent the successor features of the input features. This TeX equation summarises the idea:

    $$\hat{\psi_i}(t) = \sigma^{\textrm{non-lin}}\bigg(\sum_j w_{ij} f_j(t)\bigg)  \approx \psi_i^{\pi}(t) = \mathbb{E} \bigg[  \int_{t}^{\infty} e^{-\frac{t^{\prime}-t}{\tau}} \phi_i(t^{\prime}) dt^{\prime}\bigg] $$

    Basis features can be any list of RatInABox Neurons class here (a set of PlaceCells, BoundaryVectorCells, GridCells etc...or more complex things). Specify basis features with the params["input_layers"] kwarg. It linearly sums these inputs to calculate the firing rate (this summation is all handled by the FeedForwardLayer class). Since successor features are a subset of value functions (where the reward is the feature activity), this class is a subclass of WalueNeuron. The only fundamental difference is it also insists you pass features, on each update_weights() call passes the activity of the feaures to super().update() as reward signals. Weights are trained using TD learning, self.update_weights() should be called at each update step. Remember 'eta' and 'tau_e' as kwargs of the ValueNeuron which you may like to play around with. For more infor see ratinabox/example_scripts/successor_features/ or the parent class ValueNeuron for details of the learning rule.

    To see the weights try SuccessorFeatures.inputs['name_of_input_layer']['w']

    Note SuccessorFeatures.update() _will not_ update the inout and basis features. These must be updated in the main loop separately by the user.
    """

    default_params = {
        "features": None,  # the features to calculate the successor features of
    }

    def __init__(self, Agent, params={}):
        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)
        if self.params["features"] is None:
            raise Exception(
                "The input parameter dictionary must contain features to calculate the successor features for. params['features'] = ... This can be any RatInABox Neurons class (e.g. PlaceCells, BoundaryVectorCells, GridCells etc...or more complex things)."
            )
        self.params["n"] = self.params["features"].n
        super().__init__(
            Agent, self.params
        )  # initialise parent classese, starting with ValueNeuron



    def update_weights(self):
        rewards = self.params["features"].firingrate
        super().update_weights(rewards)
