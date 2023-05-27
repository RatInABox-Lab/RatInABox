from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.utils import *

import copy
import numpy as np


class PhasePrecessingPlaceCells(PlaceCells):
    """
    Contributer: Tom George tomgeorge1@btinternet.com
    Date: 05/10/2022

    The PhasePrecessingPlaceCell class is a subclass of PlaceCells() which is a subclass of Neurons() and inherits its properties/plotting functions from these two classes.

    Must be initialised with an Agent and a 'params' dictionary.

    PhasePrecessingPlaceCell defines a set of 'n' place cells who's firing rate is a given by a modulation factor multipled by the normal place cell firing rate. The modulation factor is simply determined by a normalised von Mises in time - as the animal enters the place field the von mises peaks early in the theta cycle, and as the animal leaves the place field, the von mises peaks late in the theta cycle. The full model is the same as that described in George et al. (2023) "Rapid learning of predictive maps with STDP and theta phase precession".

    Note this only works for PlaceCells with well defined widths (i.e. not one_hots). For gaussian place cells, since they dont have well defined boundaries, the boundary is taken at 2 sigma.

    This requires several new parameters to be provided:
        • theta_freq (default 10 Hz)
        • kappa (von Mises spread parameter, default 1)
        • precess_fraction (default 0.5)

    List of functions:
        • get_state()
        • theta_modulation_factors()
    """

    default_params = {
        "n": 10,
        "min_fr": 0,
        "max_fr": 1,
        "theta_freq": 10,
        "kappa": 1,
        "precess_fraction": 0.5,
        "description": "gaussian_threshold",
        "name": "PhasePrecessingPlaceCell",
    }

    def __init__(self, Agent, params={}):
        """Initialise PhasePrecessingPlaceCell(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)        
        self.params.update(params)

        super().__init__(Agent, self.params)
        self.sigma = np.sqrt(1 / self.kappa)

        assert self.description in [
            "gaussian",
            "diff_of_gaussians",
            "gaussian_threshold",
            "top_hat",
        ]
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the PhasePrecessingPlaceCells.
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

        firingrate = super().get_state(evaluate_at, **kwargs)

        if evaluate_at == "agent":
            theta_modulation_factors = self.theta_modulation_factors()
            firingrate *= theta_modulation_factors
        else:
            print(
                "Since you are not evaluating hte firing rate using the current state of the agent no phase precession modulation has been applied (since this requires a velocity). Ignore this if you are plotting receptive field. "
            )

        return firingrate

    def theta_modulation_factors(self):
        """
        Calcualtes from the agents position and direction of motion, as well as the position of all the place cells centres, how much the firing rate of each place cell should be modulated by.
        """
        position = self.Agent.pos
        direction = self.Agent.velocity / (1e-8 + np.linalg.norm(self.Agent.velocity))
        theta_phase = (
            self.theta_freq * (self.Agent.t % (1 / self.theta_freq)) * 2 * np.pi
        )
        sigma = self.place_cell_widths.copy()
        if self.description == "gaussian":
            sigma *= 2  # gaussian place cell boundary taken at 2 sigma

        vectors_to_cells = get_vectors_between(position, self.place_cell_centres)
        sigmas_to_cell_midline = (
            np.dot(vectors_to_cells, direction) / sigma
        )  # as mutiple of sigma
        prefered_theta_phase = (
            np.pi - sigmas_to_cell_midline * self.precess_fraction * np.pi
        )
        phase_diff = prefered_theta_phase - theta_phase
        theta_modulation_factor = (
            von_mises(phase_diff, mu=0, sigma=self.sigma) * 2 * np.pi
        ).T

        return theta_modulation_factor


if __name__ == "__main__":
    """Example of use"""
    from ratinabox.contribs.PhasePrecessingPlaceCells import PhasePrecessingPlaceCells

    Env = Environment()
    Ag = Agent(Env)
    Ag.speed_mean = 0.3
    PPPCs = PhasePrecessingPlaceCells(
        Ag,
        params={
            "widths": 0.3,
            "theta_freq": 5,
            "precess_fraction": 1,
            "kappa": 2,
            "max_fr": 10.0,
            "description": "gaussian",
        },
    )
    while Ag.t < 10:
        Ag.update()
        PPPCs.update()

    Ag.plot_trajectory()
    PPPCs.plot_rate_map()
    fig, ax = PPPCs.plot_rate_timeseries()
    T = 0
    while T < Ag.t:
        ax.axvline(T / 60, linewidth=1, color="grey", linestyle="--")
        T += 1 / PPPCs.theta_freq
    plt.show()
