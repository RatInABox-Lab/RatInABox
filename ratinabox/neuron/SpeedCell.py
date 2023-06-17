import ratinabox
from ratinabox.neuron.Neurons import Neurons


import copy


class SpeedCell(Neurons):
    """The SpeedCell class defines a single speed cell. This class is a subclass of Neurons() and inherits it properties/plotting functions.
    Must be initialised with an Agent and a 'params' dictionary.
    The firing rate is scaled according to the expected velocity of the agent (max firing rate acheive when velocity = mean + std)
    List of functions:
        • get_state()
    default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "name": "SpeedCell",
        }
    """

    default_params = {
        "min_fr": 0,
        "max_fr": 1,
        "name": "SpeedCell",
    }

    def __init__(self, Agent, params={}):
        """Initialise SpeedCell(), takes as input a parameter dictionary, 'params'. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Agent, self.params)
        self.n = 1
        self.one_sigma_speed = self.Agent.speed_mean + self.Agent.speed_std

        if ratinabox.verbose is True:
            print(
                f"SpeedCell successfully initialised. The speed of the agent is encoded linearly by the firing rate of this cell. Speed = 0 --> min firing rate of of {self.min_fr}. Speed = mean + 1std (here {self.one_sigma_speed} --> max firing rate of {self.max_fr}."
            )

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the speed cell. By default velocity (which determines speed) is taken from the agent but this can also be passed as a kwarg 'vel'
        Args:
            evaluate_at (str, optional): _description_. Defaults to 'agent'.
        Returns:
            firingrate: np.array([firingrate])
        """
        if evaluate_at == "agent":
            vel = self.Agent.history["vel"][-1]
        else:
            vel = np.array(kwargs["vel"])

        speed = np.linalg.norm(vel)
        firingrate = np.array([speed / self.one_sigma_speed])
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate