import ratinabox
from ratinabox import utils
from ratinabox.neuron.Neurons import Neurons

import numpy as np

import copy
import warnings


class FeedForwardLayer(Neurons):
    """The FeedForwardLayer class defines a layer of Neurons() whos firing rates are an activated linear combination of downstream input layers. This class is a subclass of Neurons() and inherits it properties/plotting functions.
    *** Understanding this layer is crucial if you want to build multilayer networks of Neurons with RatInABox ***
    Must be initialised with an Agent and a 'params' dictionary.
    Input params dictionary should contain a list of input_layers which feed into this layer. This list looks like [Input1, Input2,...] where each is a Neurons() class (typically this list will be length 1 but you can have arbitrariy many layers feed into this one if you want). You can also add inputs one-by-one using self.add_input()
    Each layer which feeds into this one is assigned a set of weights
        firingrate = activation_function(sum_over_layers(sum_over_inputs_in_this_layer(w_i I_i)) + bias)
    (you my be interested in accessing these weights in order to write a function which "learns" them, for example). A dictionary stores all the inputs, the key for each input layer is its name (e.g Input1.name = "Input1"), so to get the weights call
        FeedForwardLayer.inputs["Input1"]['w'] --> returns the weight matrix from Input1 to FFL
        FeedForwardLayer.inputs["Input2"]['w'] --> returns the weight matrix from Input1 to FFL
        ...
    One set of biases are stored in self.biases (defaulting to an array of zeros), one for each neuron.
    Currently supported activations include 'sigmoid' (paramterised by max_fr, min_fr, mid_x, width), 'relu' (gain, threshold) and 'linear' specified with the "activation_params" dictionary in the inout params dictionary. See also utils.utils.activate() for full details. It is possible to write your own activatino function (not recommended) under the key {"function" : an_activation_function}, see utils.actvate for how one of these should be written.
    Check that the input layers are all named differently.
    List of functions:
        • get_state()
        • add_input()
    default_params = {
            "n": 10,
            "input_layers": [],  # a list of input layers, or add one by one using self.adD_inout
            "activation_params": {
                "activation": "linear",
            },
            "name": "FeedForwardLayer",
        }
    """

    default_params = {
        "n": 10,
        "input_layers": [],  # a list of input layers, or add one by one using self.add_inout
        "activation_params": {"activation": "linear"},
        "name": "FeedForwardLayer",
        "biases": None,  # an array of biases, one for each neuron
    }

    def __init__(self, Agent, params={}):
        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Agent, self.params)

        assert isinstance(
            self.input_layers, list
        ), "param['input_layers'] must be a list."
        if len(self.input_layers) == 0:
            warnings.warn(
                "No input layers have been provided. Either hand them in in the params dictionary params['input_layers']=[list,of,inputs] or use self.add_input_layer() to add them manually."
            )

        self.inputs = {}
        for input_layer in self.input_layers:
            self.add_input(input_layer)

        if self.biases is None:
            self.biases = np.zeros(self.n)

        self.firingrate_prime = np.zeros_like(
            self.firingrate
        )  # this will hold the firingrate except passed through the derivative of the activation func

        if ratinabox.verbose is True:
            print(
                f"FeedForwardLayer initialised with {len(self.inputs.keys())} layers. To add another layer use FeedForwardLayer.add_input_layer().\nTo set the weights manually edit them by changing self.inputs['layer_name']['w']"
            )

    def add_input(self, input_layer, w=None, w_init_scale=1, **kwargs):
        """Adds an input layer to the class. Each input layer is stored in a dictionary of self.inputs. Each has an associated matrix of weights which are initialised randomly.
        Note the inputs are stored in a dictionary. The keys are taken to be the name of each layer passed (input_layer.name). Make sure you set this correctly (and uniquely).
        Args:
            • input_layer (_type_): the layer intself. Must be a Neurons() class object (e.g. can be PlaceCells(), etc...).
            • w: the weight matrix. If None these will be drawn randomly, see next argument.
            • w_init_scale: initial weights drawn from zero-centred gaussian with std w_init_scale / sqrt(N_in)
            • **kwargs any extra kwargs will get saved into the inputs dictionary in case you need these
        """
        n = input_layer.n
        name = input_layer.name
        if w is None:
            w = np.random.normal(
                loc=0, scale=w_init_scale / np.sqrt(n), size=(self.n, n)
            )
        I = np.zeros(n)
        if name in self.inputs.keys():
            if ratinabox.verbose is True:
                print(
                    f"There already exists a layer called {name}. This may be because you have two input players with the same attribute `InputLayer.name`. Overwriting it now."
                )
        self.inputs[name] = {}
        self.inputs[name]["layer"] = input_layer
        self.inputs[name]["w"] = w
        self.inputs[name]["w_init"] = w.copy()
        self.inputs[name]["I"] = I
        self.inputs[name]["n"] = input_layer.n  # a copy for convenience
        for key, value in kwargs.items():
            self.inputs[name][key] = value
        if ratinabox.verbose is True:
            print(
                f'An input layer called {name} was added. The weights can be accessed with "self.inputs[{name}]["w"]"'
            )

    def get_state(self, evaluate_at="last", **kwargs):
        """Returns the firing rate of the feedforward layer cells. By default this layer uses the last saved firingrate from its input layers. Alternatively evaluate_at and kwargs can be set to be anything else which will just be passed to the input layer for evaluation.
        Once the firing rate of the inout layers is established these are multiplied by the weight matrices and then activated to obtain the firing rate of this FeedForwardLayer.
        Args:
            evaluate_at (str, optional). Defaults to 'last'.
        Returns:
            firingrate: array of firing rates
        """
        if evaluate_at == "last":
            V = np.zeros(self.n)
        elif evaluate_at == "all":
            V = np.zeros(
                (self.n, self.Agent.Environment.flattened_discrete_coords.shape[0])
            )
        else:
            V = np.zeros((self.n, kwargs["pos"].shape[0]))

        for inputlayer in self.inputs.values():
            w = inputlayer["w"]
            if evaluate_at == "last":
                I = inputlayer["layer"].firingrate
            else:  # kick can down the road let input layer decide how to evaluate the firingrate. this is core to feedforward layer as this recursive call will backprop through the upstraem layers until it reaches a "core" (e.g. place cells) layer which will then evaluate the firingrate.
                I = inputlayer["layer"].get_state(evaluate_at, **kwargs)
            inputlayer["I_temp"] = I
            V += np.matmul(w, I)

        biases = self.biases
        if biases.shape != V.shape:
            biases = biases.reshape((-1, 1))
        V += biases

        firingrate = utils.activate(V, other_args=self.activation_params)
        # saves current copy of activation derivative at firing rate (useful for learning rules)
        if (
            evaluate_at == "last"
        ):  # save copy of the firing rate through the dervative of the activation function
            self.firingrate_prime = utils.activate(
                V, other_args=self.activation_params, deriv=True
            )
        return firingrate