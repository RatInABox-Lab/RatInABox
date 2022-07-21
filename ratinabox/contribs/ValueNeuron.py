class ValueNeuron(FeedForwardLayer):
    """
    Contributer: Tom George
    Date: 21/07/2022

    The ValueNeuron class defines a neuron which learns the "value" of a policy using temporally continuous TD learning . This class is a subclass of FeedForwardLayer() which is a subclass of Neurons() and inherits it properties/plotting functions from both of these.  

    It takes as input a layer of neurons (these are the "features" over which value is calculated). You could pass in any ratinabox Neurons class here (a set of PlaceCells, BoundaryVectorCells, GridCells etc...or more complex things)
    
    It linearly sums these inputs to calculate its firing rate (this summation is all handled by the FeedForwardLayer class). 
    
    Weights are trained using TD learning, self.update_weights() should be called at each update step and passed the current reward density). For more infor see ratinabox/example_scripts/reinforcement_learning_example/  

    Since this is a Neurons subclass, after (or even during) learning you can plot the value function just by querying the ValueNeurons rate map (ValueNeuron.plot_rate_map()), or check the estimate of value at a postion using ValueNeuron.get_state(evaluate_at=None, pos=np.array([[x,y]]))

    To see the weights try ValueNeuron.inputs['name_of_input_layer']['w']
    """    
    def __init__(self, Agent, params={}):
        default_params = {
            "input_layer": None,  # the features it is using as inputs
            "tau":10, #discount time horizon
            "tau_e":1, #eligibility trace timescale
            "eta":0.0001, #learning rate
        }

        default_params.update(params)
        self.params = default_params
        self.params['activation_params']={'activation':'linear'} #we use linear func approx
        self.params['n']=1 #one value neuron 
        self.params['input_layers'] = [self.params['input_layer']]
        super().__init__(Agent, self.params) #initialise parent class

        self.et = np.zeros(params['input_layer'].n) #initialise eligibility trace
        self.firingrate = np.zeros(1) #initialise firing rate
        self.firingrate_deriv = np.zeros(1) #initialise firing rate derivative
        self.max_fr = 1 #will update this with each episode later 


    def update(self):
        """Updates firing rate as weighted linear sum of feature inputs
        """        
        firingrate_last = self.firingrate 
        #update the firing rate
        super.update() #FeedForwardLayer builtin function. this sums the inouts from the input features over the weight matrix and saves the firingrate. 
        #calculate temporal derivative of the firing rate 
        self.firingrate_deriv = (self.firingrate - firingrate_last)/self.Agent.dt
        # update eligibility trace 
        self.et = ((self.Agent.dt/self.tau_e) * self.input_layer.firingrate + 
                   (1 - self.Agent.dt/self.tau_e) * self.et)
        return

    def update_weights(self,reward):
        """Trains the weights by implementing the TD learning rule,
        reward is the current reward density"""
        w = self.inputs[self.input_layer.name]['w'] #weights
        V = self.firingrate #current value estimate
        dVdt = self.firingrate_deriv #currrent value derivative estimate
        td_error = R + self.tau * dVdt - V #this is the continuous analog of the TD error 
        dw = self.Agent.dt*self.eta*(np.outer(td_error,self.et)) - 0.0001*w #note L2 regularisation
        self.inputs[self.input_layer.name]['w'] += dw
        return