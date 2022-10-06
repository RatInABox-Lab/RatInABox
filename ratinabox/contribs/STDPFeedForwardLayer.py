import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.utils import *


class STDPFeedForwardLayer(FeedForwardLayer):
    """
    Contributer: Tom George tomgeorge1@btinternet.com
    Date: 05/10/2022

    The STDPFeedForwardLayer class defines a population of FeedForwardLayer neurons which is itself subclass of Neurons() and therefore inherits it properties/plotting functions.  

    Must be initialised with an Agent and a 'params' dictionary. Put in params anything you would also put in your params dictionary for a FeedForwardLayer. 

    STDPFeedForwardLayer defines a set of 'n' feed forward cells. The only difference with the typical feed forward layer class is that the weight matrix is learn via STDP between the spikes in the pre and post lneurons. The pre layer is the layer (or layers) which feed into this layer. The post layer is this layer itself. The full STDP learning rule  is the same as that described in George et al. (2023) "Rapid learning of predictive maps with STDP and theta phase precession".
    
    This requires several new parameters to be provided: 
        • tau_STDP_plus       (default 20e-3     pre trace decay time)
        • tau_STDP_minus      (default 40e-3     post trace decay time)
        • a_STDP              (default -0.4      pre-before-post potentiation factor)
        • eta                 (default 0.05      STDP learning rate)
        • update_inert_copy   (default False     whether to use the learnt weight matrix to determine firing rates online or just use the matrix at initialisation)

    List of functions: 

    """

    def __init__(self, Agent, params={}):
        """Initialise STDPFeedForwardLayer(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}."""

        default_params = {
            "n": 10,
            "input_layers": [],
            "tau_STDP_plus": 20e-3,
            "tau_STDP_minus": 40e-3,
            "a_STDP": -0.4,
            "eta": 0.05,
            "update_inert_copy": False,
            "name": "STDPFeedForwardLayer",
        }

        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        #add in trace variables
        self.post_trace = np.zeros(self.n)
        for input_name in self.inputs.keys():
            self.inputs[input_name]['pre_trace'] = np.zeros(self.inputs[input_name]['layer'].n)
        print(self.inputs)
        return
    
    def update(self):
        super().update() # FeedForwardLayer builtin function. This sums the inputs from the input features over the weight matrix and saves the firingrate.
        for input_name in self.inputs.keys():
            layer = self.inputs[input_name]['layer']
            spiked_cells = layer.history['spikes'][-1]
            if spiked_cells.sum() > 0: 
                spike_times = np.random.uniform(self.Agent.t,self.Agent.t+self.Agent.dt,layer.n)[spiked_cells]
                spike_ids = np.arange(layer.n)[spiked_cells]
                spike_names = [layer.name]*spiked_cells.sum()
                spike_list = np.vstack((spike_name,spike_ids,spike_times))





if __name__ == "__main__":
    """Example of use
    """
    from ratinabox.contribs.STDPFeedForwardLayer import STDPFeedForwardLayer

    Env = Environment()
    Ag = Agent(Env)
    PCs = PlaceCells(Ag)
    STDPPCs = STDPFeedForwardLayer(Ag, params={"input_layers": [PCs],})
