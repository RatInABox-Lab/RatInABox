from ratinabox.Environment import Environment
from ratinabox.Neurons import Neurons
import copy
import numpy as np

import torch #pytorch, for the neural network
import torch.nn as nn
import warnings


class NeuralNetworkNeurons(Neurons):
    """The NeuralNetworkNeurons class takes as inputs other RiaB Neurons classes and uses these to compute its own firing rates by concatenating the firing rates of all the inputs and passing them through a neural network model provided by the user.  

    For example a NeuralNetworkNeurons could take the firing rates of a set of place cells and a set of grid cells as inputs and passes these through the deep neural network. This would go as follows: 
        
        Env = Environment()
        Ag = Agent(Env)
        PCs = PlaceCells(Ag)
        GCs = GridCells(Ag)
        NNNs = NeuralNetworkNeurons(Ag, params={"input_layers": [PCs, GCs], "NeuralNetworkModule": <any-user-provided-torch-nn.Module>})

    Training the neural network: In order to maintain compatibility with other RiaB functionalities the .get_state() method returns a numpy array of firing rates detached from the pytorch graph (which cannot then be used for training). However, when called in the standard update()-loop it also saves an attribute called .firingrate_torch which is an attached pytorch tensor and can be used to take gradients for training etc., if this is needed. 
    
    Users either provide a "NeuralNetworkModule" and no "n" (in which case self.n is set to be the output size of the NN) or "n" and no "NeuralNetworkModule" (in which case a default ReLU MLP with two hidden layers is used with self.n_out = self.n). If both are provided then a warning is raised and the user provided "n" is overwritten by the output size of the NN.

    Args:
        input_layers (list): a list of input layers, these are RatInABox.Neurons classes and must all be provided at initialisation.
        
        NeuralNetworkModule (nn.Module): Any nn.Module that has a .forward() method. It is the users job to ensure that the input size of this network matches the total number of neurons in the passed input_layers. The number of neurons in the output of this network will serve as the number of neurons in this layer. The .forward(X) function must accept X as a torch array of shape (n_batch, n_in) and returns a 2D torch array of shape (n_batch, n_out). Note, however, most nn.Modules assume the first dimension is the batch dimension anyway.
    """

    default_params = {
        "n":None, #number of neurons in this layer, must match output size of NeuralNetworkModule. Typically leave this as None and let it be set by the output size of the NeuralNetworkModule
        "input_layers": [],  # a list of input layers, each must be a ratinabox.Neurons class
        "NeuralNetworkModule": None, #Any torch nn.Sequential or nn.Module with a .forward() method
        "name": "NeuralNetworkNeurons",}
    
    def __init__(self,Agent,params={}):
        self.Agent = Agent
        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Agent, self.params)

        assert isinstance(self.input_layers, list), "param['input_layers'] must be a list of Neurons."
        assert len(self.input_layers) > 0, "No input layers have been provided. Hand them in in the params dictionary params['input_layers']=[list,of,inputs]"
        
        self.n_in = sum([layer.n for layer in self.input_layers])

        # Check the number of output of the NeuralNetworkModule and self.n are compatible 
        # in order of preference...
        if (self.n is None) and (self.NeuralNetworkModule is not None): 
            self.n = self.NeuralNetworkModule(torch.zeros(1,self.n_in)).shape[1]
        
        elif (self.n is not None) and (self.NeuralNetworkModule is None):
            self.NeuralNetworkModule = MultiLayerPerceptron(n_in=self.n_in, n_out=self.n, n_hidden=[20,20])
            warnings.warn(f"No NeuralNetworkModule was provided so a default MLP with {self.n_in} inputs, {self.n} outputs and 2 hidden ReLU layers of size 20 will be used. Alternatively provide one in the params['NeuralNetworkModule']=<any-torch-nn.Module-with-a-.forward()-method> and don't set 'n'.")

        elif (self.n is not None) and (self.NeuralNetworkModule is not None):
            raise ValueError(f"You provided both 'n' and `NeuralNetworkModule` as parameters. These are mutually exclusive. Either provide 'NeuralNetworkModule' and no 'n' (the output size of the NeuralNetworkModule will be used as the number of neurons in this layer) or 'n' and no 'NeuralNetworkModule (a default MLP will be initialised)")

        elif (self.n is None) and (self.NeuralNetworkModule is None):
            raise ValueError("You provided neither a 'NeuralNetworkModule' nor 'n' as parameters. Either provide 'NeuralNetworkModule' and no 'n' (the output size of the NeuralNetworkModule will be used as the number of neurons in this layer) or 'n' and no 'NeuralNetworkModule (a default MLP will be initialised)")


        #Finally, check that the NeuralNetworkModule accepts the right sized inputs
        try:
            self.NeuralNetworkModule(torch.zeros(1,self.n_in))
        except: 
            raise ValueError(f"You provided inputs layers with a total of {self.n_in} neurons but the NeuralNetworkModule you provided does not accept inputs of size (1,{self.n_in}) so they are incompatible")
        
        return

    def get_state(self, evaluate_at="last", save_torch=False, **kwargs):
        """Returns the firing rate of the NeuralNetworkNeurons. There are two stages: 
        
        1. First the input layer Neurons are queried for their firing rates. By default, this just takes the last firingrate (i.e. from the last time .update() was called on these Neurons). Alternatively `evaluate_at` and `kwargs` can be set in order to evaluate the inputs in some other way (e.g. at some other location: evaluate_at=None, pos=np.array([x,y])....or an array of all locations evaluate_all='all' etc.). All inputs are concatentated into a single array of shape (n_batch, n_in) where n_in is the total number of input neurons. n_batch is typical 1 (for a single location) but can be more if evaluating at multiple locations (e.g. evaluate_all='all').
        
        2. The concatenated inputs are passed through the NeuralNetworkModule to compute the firing rate of the firing rate of the input layers is established these are concatenated and passeed through the neural network to calculate the output.

        Note the function returns the firingrate as a detached numpy array (i.e. no gradients) but also saves it as an attribute self.firingrate_torch which is a torch tensor and can be used to take gradients.

        Args:
            • evaluate_at (str, optional): If "last" (default) then the firing rate of the input layers is taken as the last firing rate (i.e. from the last time .update() was called on these Neurons). Alternatively, this can be set to None in which case the inputs are evaluated at some other location (e.g. evaluate_at=None, pos=np.array([x,y])....or an array of all locations evaluate_all='all' etc.). Defaults to "last".
            • save_torch (bool, optional): If True then the torch tensor self.firingrate_torch is saved as an attribute and can be used for optimsation of the network. Defaults to False.
            • kwargs: any additional arguments to be passed to the input layer's .get_state() method.
        Returns:
            firingrate: array of firing rates
        """
        #Gets the inputs in the shape (n_batch, n_in)
        if evaluate_at=="last":
            inputs = np.concatenate([layer.firingrate for layer in self.input_layers]).reshape(1,-1)
        else:
            #else kick the can down the road and just run get_state on all inputs and concatenate these
            # note the convention change of (n_batch, )
            inputs = np.concatenate([layer.get_state(evaluate_at, **kwargs) for layer in self.input_layers]).T
        
        inputs_torch = torch.Tensor(inputs.astype(np.float32))
        inputs_torch.requires_grad = True
        firingrate_torch = self.NeuralNetworkModule(inputs_torch) 
        if save_torch: 
            self.firingrate_torch = firingrate_torch # <-- note the shape convention on this in (n_batch, n_neurons, the opposite of RiaB standard
        
        return firingrate_torch.detach().numpy().T


    def update(self):
        """Updates the neuron but sets save_torch arg to True so self.firingrate_torch will be saved giving user access to the computational graph"""
        super().update(save_torch=True)















#An example neural network which will act as the core neural network module at work within for the DeepNeuralNetworkNeurons class
class MultiLayerPerceptron(nn.Module):
    """A generic ReLU neural network class, default used for the core function in NeuralNetworkNeurons. 
    Specify input size, output size and hidden layer sizes (a list). Biases are used by default.

    Args:
        n_in (int, optional): The number of input neurons. Defaults to 20.
        n_out (int, optional): The number of output neurons. Defaults to 1.
        n_hidden (list, optional): A list of integers specifying the number of neurons in each hidden layer. Defaults to [20,20]."""

    def __init__(self, n_in=20, n_out=1, n_hidden=[20,20]):
        nn.Module.__init__(self)
        n = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(len(n)-1):
            layers.append(nn.Linear(n[i],n[i+1]))
            if i < len(n)-2: layers.append(nn.ReLU()) #add a ReLU after each hidden layer (but not the last)
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        """Forward pass, X must be a torch tensor. Returns an (attached) torch tensor through which you can take gradients. """
        return self.net(X)