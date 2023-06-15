import ratinabox

from ratinabox.neuron.HeadDirectionCells import HeadDirectionCells

import copy
import numpy as np


class VelocityCells(HeadDirectionCells):
    """The VelocityCells class defines a population of Velocity cells. This basically takes the output from a population of HeadDirectionCells and scales it proportional to the speed (dependence on speed and direction --> velocity).
    Must be initialised with an Agent and a 'params' dictionary. Initalise tehse cells as if they are HeadDirectionCells
    VelocityCells defines a set of 'dim x 2' velocity cells. Encoding the East, West (and North and South) velocities in 1D (2D). The firing rates are scaled according to the multiple current_speed / expected_speed where expected_speed = Agent.speed_mean + self.Agent.speed_std is just some measure of speed approximately equal to a likely ``rough`` maximum for the Agent.
    List of functions:
        • get_state()
    default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "name": "VelocityCells",
        }
    """

    default_params = {
        "min_fr": 0,
        "max_fr": 1,
        "name": "VelocityCells",
    }

    def __init__(self, Agent, params={}):
        """Initialise VelocityCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""
        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        self.one_sigma_speed = self.Agent.speed_mean + self.Agent.speed_std

        super().__init__(Agent, self.params)

        if ratinabox.verbose is True:
            print(
                f"VelocityCells successfully initialised. Your environment is {self.Agent.Environment.dimensionality} and you have {self.n} velocity cells"
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Takes firing rate of equivalent set of head direction cells and scales by how fast teh speed is realtive to one_sigma_speed (likely rough maximum speed)"""

        HDC_firingrates = super().get_state(evaluate_at, **kwargs)
        speed_scale = np.linalg.norm(self.Agent.velocity) / self.one_sigma_speed
        firingrate = HDC_firingrates * speed_scale
        return firingrate