import ratinabox
from ratinabox import utils
from ratinabox.neuron.Neurons import Neurons


from matplotlib import pyplot as plt
import numpy as np

import copy


class HeadDirectionCells(Neurons):
    """The HeadDirectionCells class defines a population of head direction cells. This class is a subclass of Neurons() and inherits it properties/plotting functions.
    Must be initialised with an Agent and a 'params' dictionary.
    HeadDirectionCells defines a set of 'n' head direction cells. Each cell has a preffered direction/angle (default evenly spaced across unit circle). In 1D there are always only n=2 cells preffering left and right directions. The firing rates are scaled such that when agent travels exactly along the preferred direction the firing rate of that cell is the max_fr. The firing field of a cell is a von mises centred around its preferred direction of default width 30 degrees (can be changed with parameter params["angular_spread_degrees"])
    To print/set preffered direction: self.preferred_angles
    List of functions:
        • get_state()
    default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "n":10,
            "angle_spread_degrees":30,
            "name": "HeadDirectionCells",
        }
    """

    default_params = {
        "min_fr": 0,
        "max_fr": 1,
        "n": 10,
        "angular_spread_degrees": 45,  # width of HDC preference function (degrees)
        "name": "HeadDirectionCells",
    }

    def __init__(self, Agent, params={}):
        """Initialise HeadDirectionCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        if self.Agent.Environment.dimensionality == "2D":
            self.n = self.params["n"]
            self.preferred_angles = np.linspace(0, 2 * np.pi, self.n + 1)[:-1]
            # self.preferred_directions = np.array([np.cos(angles),np.sin(angles)]).T #n HDCs even spaced on unit circle
            self.angular_tunings = np.array(
                [self.params["angular_spread_degrees"] * np.pi / 180] * self.n
            )
        if self.Agent.Environment.dimensionality == "1D":
            self.n = 2  # one left, one right
        self.params["n"] = self.n
        super().__init__(Agent, self.params)
        if ratinabox.verbose is True:
            print(
                f"HeadDirectionCells successfully initialised. Your environment is {self.Agent.Environment.dimensionality}, you have {self.n} head direction cells"
            )

    def get_state(self, evaluate_at="agent", **kwargs):
        """In 2D n head direction cells encode the head direction of the animal. By default velocity (which determines head direction) is taken from the agent but this can also be passed as a kwarg 'vel'"""

        if evaluate_at == "agent":
            vel = self.Agent.history["vel"][-1]
        elif "vel" in kwargs.keys():
            vel = np.array(kwargs["vel"])
        else:
            print("HeadDirection cells need a velocity but not was given, taking...")
            if self.Agent.Environment.dimensionality == "2D":
                vel = np.array([1, 0])
                print("...[1,0] as default")
            if self.Agent.Environment.dimensionality == "1D":
                vel = np.array([1])
                print("...[1] as default")

        if self.Agent.Environment.dimensionality == "1D":
            hdleft_fr = max(0, np.sign(vel[0]))
            hdright_fr = max(0, -np.sign(vel[0]))
            firingrate = np.array([hdleft_fr, hdright_fr])
        if self.Agent.Environment.dimensionality == "2D":
            current_angle = utils.get_angle(vel)
            firingrate = utils.von_mises(
                current_angle, self.preferred_angles, self.angular_tunings, norm=1
            )

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]

        return firingrate

    def plot_HDC_receptive_field(
        self, chosen_neurons="all", fig=None, ax=None, autosave=None
    ):
        """Plots the receptive fields, in polar coordinates, of hte head direction cells. The receptive field is a von mises function centred around the preferred direction of the cell.
        Args:
            chosen_neurons (str, optional): The neurons to plot. Defaults to "all".
            fig, ax (_type_, optional): matplotlib fig, ax objects ot plot onto (optional).
            autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
        Returns:
            fig, ax
        """
        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)
        if fig is None and ax is None:
            fig, ax = plt.subplots(
                1,
                len(chosen_neurons),
                figsize=(2 * len(chosen_neurons), 2),
                subplot_kw={"projection": "polar"},
            )

        for i, n in enumerate(chosen_neurons):
            theta = np.linspace(0, 2 * np.pi, 100)
            pref_theta = self.preferred_angles[n]
            sigma_theta = self.angular_tunings[n]
            fr = utils.von_mises(theta, pref_theta, sigma_theta, norm=1)
            fr = fr * (self.max_fr - self.min_fr) + self.min_fr
            ax[i].plot(theta, fr, linewidth=2, color=self.color, zorder=11)
            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax[i].fill_between(theta, fr, 0, color=self.color, alpha=0.2)
            ax[i].set_ylim([0, self.max_fr])
            ax[i].tick_params(pad=-18)
            ax[i].set_xticklabels(["E", "N", "W", "S"])

        ratinabox.utils.save_figure(fig, self.name + "_ratemaps", save=autosave)

        return fig, ax