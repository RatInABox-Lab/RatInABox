import numpy as np

import ratinabox
from ratinabox.neuron.Neurons import Neurons


import copy


class PlaceCells(Neurons):
    """The PlaceCells class defines a population of PlaceCells. This class is a subclass of Neurons() and inherits it properties/plotting functions.
       Must be initialised with an Agent and a 'params' dictionary.
       PlaceCells defines a set of 'n' place cells scattered across the environment. The firing rate is a functions of the distance from the Agent to the place cell centres. This function (params['description'])can be:
           • gaussian (default)
           • gaussian_threshold
           • diff_of_gaussians
           • top_hat
           • one_hot
    #TO-DO • tanni_harland  https://pubmed.ncbi.nlm.nih.gov/33770492/
       List of functions:
           • get_state()
           • plot_place_cell_locations()
       default_params = {
               "n": 10,
               "name": "PlaceCells",
               "description": "gaussian",
               "widths": 0.20,
               "place_cell_centres": None,  # if given this will overwrite 'n',
               "wall_geometry": "geodesic",
               "min_fr": 0,
               "max_fr": 1,
               "name": "PlaceCells",
           }
    """

    default_params = {
        "n": 10,
        "name": "PlaceCells",
        "description": "gaussian",
        "widths": 0.20,  # the radii
        "place_cell_centres": None,  # if given this will overwrite 'n',
        "wall_geometry": "geodesic",
        "min_fr": 0,
        "max_fr": 1,
        "name": "PlaceCells",
    }

    def __init__(self, Agent, params={}):
        """Initialise PlaceCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}.
        """

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        if self.params["place_cell_centres"] is None:
            self.params["place_cell_centres"] = self.Agent.Environment.sample_positions(
                n=self.params["n"], method="uniform_jitter"
            )
        elif type(self.params["place_cell_centres"]) is str:
            if self.params["place_cell_centres"] in [
                "random",
                "uniform",
                "uniform_jitter",
            ]:
                self.params[
                    "place_cell_centres"
                ] = self.Agent.Environment.sample_positions(
                    n=self.params["n"], method=self.params["place_cell_centres"]
                )
            else:
                raise ValueError(
                    "self.params['place_cell_centres'] must be None, an array of locations or one of the instructions ['random', 'uniform', 'uniform_jitter']"
                )
        else:
            self.params["n"] = self.params["place_cell_centres"].shape[0]
        self.place_cell_widths = self.params["widths"] * np.ones(self.params["n"])

        super().__init__(Agent, self.params)

        # Assertions (some combinations of boundary condition and wall geometries aren't allowed)
        if self.Agent.Environment.dimensionality == "2D":
            if all(
                [
                    (
                        (self.wall_geometry == "line_of_sight")
                        or ((self.wall_geometry == "geodesic"))
                    ),
                    (self.Agent.Environment.boundary_conditions == "periodic"),
                    (self.Agent.Environment.dimensionality == "2D"),
                ]
            ):
                print(
                    f"{self.wall_geometry} wall geometry only possible in 2D when the boundary conditions are solid. Using 'euclidean' instead."
                )
                self.wall_geometry = "euclidean"
            if (self.wall_geometry == "geodesic") and (
                len(self.Agent.Environment.walls) > 5
            ):
                print(
                    "'geodesic' wall geometry only supported for enivironments with 1 additional wall (4 bounding walls + 1 additional). Sorry. Using 'line_of_sight' instead."
                )
                self.wall_geometry = "line_of_sight"

        if ratinabox.verbose is True:
            print(
                "PlaceCells successfully initialised. You can see where they are centred at using PlaceCells.plot_place_cell_locations()"
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the place cells.
        By default position is taken from the Agent and used to calculate firinf rates. This can also by passed directly (evaluate_at=None, pos=pass_array_of_positions) or ou can use all the positions in the environment (evaluate_at="all").
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

        # place cell fr's depend only on how far the agent is from cell centres (and their widths)
        dist = (
            self.Agent.Environment.get_distances_between___accounting_for_environment(
                self.place_cell_centres, pos, wall_geometry=self.wall_geometry
            )
        )  # distances to place cell centres
        widths = np.expand_dims(self.place_cell_widths, axis=-1)

        if self.description == "gaussian":
            firingrate = np.exp(-(dist**2) / (2 * (widths**2)))
        if self.description == "gaussian_threshold":
            firingrate = np.maximum(
                np.exp(-(dist**2) / (2 * (widths**2))) - np.exp(-1 / 2),
                0,
            ) / (1 - np.exp(-1 / 2))
        if self.description == "diff_of_gaussians":
            ratio = 1.5
            firingrate = np.exp(-(dist**2) / (2 * (widths**2))) - (
                1 / ratio**2
            ) * np.exp(-(dist**2) / (2 * ((ratio * widths) ** 2)))
            firingrate *= ratio**2 / (ratio**2 - 1)
        if self.description == "one_hot":
            closest_centres = np.argmin(np.abs(dist), axis=0)
            firingrate = np.eye(self.n)[closest_centres].T
        if self.description == "top_hat":
            firingrate = 1 * (dist < self.widths)

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def plot_place_cell_locations(
        self,
        fig=None,
        ax=None,
        autosave=None,
    ):
        """Scatter plots where the centre of the place cells are
        Args:
            fig, ax: if provided, will plot fig and ax onto these instead of making new.
            autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`.Defaults to None in which case looks for global constant ratinabox.autosave_plots
        Returns:
            _type_: _description_
        """
        if fig is None and ax is None:
            fig, ax = self.Agent.Environment.plot_environment(autosave=False)
        else:
            _, _ = self.Agent.Environment.plot_environment(
                fig=fig, ax=ax, autosave=False
            )
        place_cell_centres = self.place_cell_centres

        x = place_cell_centres[:, 0]
        if self.Agent.Environment.dimensionality == "1D":
            y = np.zeros_like(x)
        elif self.Agent.Environment.dimensionality == "2D":
            y = place_cell_centres[:, 1]

        ax.scatter(
            x,
            y,
            c="C1",
            marker="x",
            s=15,
            zorder=2,
        )
        ratinabox.utils.save_figure(fig, "place_cell_locations", save=autosave)

        return fig, ax

    def remap(self):
        """Resets the place cell centres to a new random distribution. These will be uniformly randomly distributed in the environment (i.e. they will still approximately span the space)"""
        self.place_cell_centres = self.Agent.Environment.sample_positions(
            n=self.n, method="uniform_jitter"
        )
        np.random.shuffle(self.place_cell_centres)
        return