from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.utils import *

import copy
import numpy as np


class PlaneWaveNeurons(Neurons):
    """
    Contributer: Tom George tomgeorge1@btinternet.com
    Date: 23/07/2022

    The PlaneWaveNeurons class defines a population of PlaneWaveNeurons. This class is a subclass of Neurons() and inherits it properties/plotting functions.

    Must be initialised with an Agent and a 'params' dictionary.

    PlaneWaveNeurons defines a set of 'n' neurons whos firing rate is a plane cells with random orientations, lengthscales (around some mean) and offsets). If you want non random orientations and phase offsets just set self.w and self.phase_offsets and self.wavescales manually.

    List of functions:
        • get_state()
    """

    default_params = {
        "n": 10,
        "wavescale": 0.2,  # metres
        "min_fr": 0,
        "max_fr": 1,
        "name": "PlaneWaveNeurons",
    }

    def __init__(self, Agent, params={}):
        """Initialise PlaneWaveNeurons(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)        
        self.params.update(params)

        super().__init__(Agent, self.params)

        # Initialise  cells
        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "PlaneWaveNeurons only available in 2D"

        if self.Agent.Environment.boundary_conditions == "periodic":
            print(
                "PlaneWaveNeurons not optimized for periodic environments, you may notice some discontinuities"
            )

        self.phase_offsets = np.random.uniform(0, self.wavescale, size=(self.n, 2))
        self.w = np.random.normal(size=(self.n, 2))
        self.w = self.w / np.expand_dims(np.linalg.norm(self.w, axis=1), axis=1)
        self.wavescales = np.random.rayleigh(scale=self.wavescale, size=self.n)

        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the PlaneWaveNeurons .
        By default position is taken from the Agent and used to calculate firing rates. This can also by passed directly (evaluate_at=None, pos=pass_array_of_positions) or you can use all the positions in the environment (evaluate_at="all").

        Returns:
            firingrates: an array of firing rates
        """
        if evaluate_at == "agent":
            pos = self.Agent.pos
        elif evaluate_at == "all":
            pos = self.Agent.Environment.flattened_discrete_coords
        else:
            pos = kwargs["pos"]
        pos = np.array(pos)
        pos = pos.reshape(-1, pos.shape[-1])

        # grid cells are modelled as the thresholded sum of three cosines all at 60 degree offsets
        # vectors to grids cells "centred" at their (random) phase offsets
        vecs = get_vectors_between(self.phase_offsets, pos)  # shape = (N_cells,N_pos,2)
        w = np.tile(np.expand_dims(self.w, axis=1), reps=(1, pos.shape[0], 1))
        wavescales = np.tile(
            np.expand_dims(self.wavescales, axis=1), reps=(1, pos.shape[0])
        )
        phi_1 = ((2 * np.pi) / wavescales) * (vecs * w).sum(axis=-1)
        firingrate = 0.5 * ((np.cos(phi_1)) + 1)
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate


if __name__ == "__main__":
    """Example of use"""
    from ratinabox.contribs.PlaneWaveNeurons import PlaneWaveNeurons

    Env = Environment()
    Ag = Agent(Env)
    PWNs = PlaneWaveNeurons(Ag, params={"wavescale": 0.01})
    fig, ax = PWNs.plot_rate_map()
    plt.show()
