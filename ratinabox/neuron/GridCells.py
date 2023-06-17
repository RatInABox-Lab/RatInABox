import numpy as np

import ratinabox
from ratinabox import utils
from ratinabox.neuron.Neurons import Neurons


import copy
import warnings


class GridCells(Neurons):
    """The GridCells class defines a population of GridCells. This class is a subclass of Neurons() and inherits it properties/plotting functions.
    Must be initialised with an Agent and a 'params' dictionary.
    GridCells defines a set of 'n' grid cells with random orientations, grid scales and offsets (these can be set non-randomly of course). Grids are modelled as the rectified sum of three cosine waves at 60 degrees to each other.
    To initialise grid cells you specify three things: (i) params['gridscale'], (ii) params['orientation'] and (iii) params['phase_offset']. These are all sampled from a distribution (specified as, e.g. params['phase_offset_distribution']) and then used to calculate the firing rate of each grid cell. For each of these there quantities the value you specify parameterises the distribution from which it is sampled. For example params['gridscale':0.5,'gridscale_distribution':'uniform'] will pull gridscales from a uniform distribution between 0.5*gridscale (=0.25m) and 1.5*gridscale (=0.75m) The 'delta' distribution means a constant will be taken. For all three of these you can optionally just pass an array of length GridCells.n (in which case the corresponding distribution parameter is ignored). This array is set a the value for each grid cell.
    params['description'] gives the place cells model being used. Currently either rectified sum of three cosines "three_rectified_cosines" or a shifted sum of three cosines "three_shifted_cosines" (which is similar, just a little softer at the edges, see Solstad et al. 2006)
    List of functions:
        • get_state()
        • set_phase_offsets()
    default_params = {
            "n": 10,
            "gridscale": 0.5,
            "gridscale_distribution": "uniform",
            "orientation": None,
            "orientation_distribution": "uniform",
            "phase_offset": True,
            "phase_offset_distribution": "uniform",
            "description":"three_rectified_cosines",
            "min_fr": 0,
            "max_fr": 1,
            "name": "GridCells",
        }
    """

    default_params = {
        "n": 10,
        "gridscale": 0.50,
        "gridscale_distribution": "rayleigh",
        "orientation": None,
        "orientation_distribution": "uniform",
        "phase_offset": None,
        "phase_offset_distribution": "uniform",
        "description": "three_rectified_cosines",  # can also be "three_shifted_cosines" as in Solstad 2006 Eq. (2)
        "min_fr": 0,
        "max_fr": 1,
        "name": "GridCells",
    }

    def __init__(self, Agent, params={}):
        """Initialise GridCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        # deprecation warnings
        if (
            ("random_gridscales" in self.params)
            or ("random_orientations" in self.params)
            or ("random_phase_offsets" in self.params)
        ):
            warnings.warn(
                "the GridCell API has changed slightly, 'random_gridscales', 'random_orientations' and 'random_phase_offsets' are no longer accepted as parameters. Please use 'gridscale','gridscale_distribution','orientation','orientation_distribution','phase_offset' and 'phase_offset_distribution' instead. See docstring or 1.7.0 release notes for instructions."
            )

        # Initialise the gridscales
        if hasattr(self.params["gridscale"], "__len__"):
            self.gridscales = np.array(self.params["gridscale"])
            self.params["n"] = len(self.gridscales)
        else:
            if self.params["gridscale_distribution"] == "uniform":
                self.gridscales = np.random.uniform(
                    0.5 * self.params["gridscale"],
                    1.5 * self.params["gridscale"],
                    self.params["n"],
                )
            elif self.params["gridscale_distribution"] == "rayleigh":
                self.gridscales = np.random.rayleigh(
                    scale=self.params["gridscale"], size=self.params["n"]
                )
            elif self.params["gridscale_distribution"] == "logarithmic":
                [low, high] = list(self.params["gridscale"])
                self.gridscales = np.logspace(
                    np.log10(low), np.log10(high), num=self.params["n"], base=10
                )
            elif self.params["gridscale_distribution"] == "delta":
                self.gridscales = np.ones(self.params["n"]) * self.params["gridscale"]
            else:
                raise ValueError("gridscale distribution not recognised")

        # Initialise Neurons parent class
        super().__init__(Agent, self.params)

        # Initialise phase offsets for each grid cell
        if (
            hasattr(self.params["phase_offset"], "__len__")
            and len(np.array(self.params["phase_offset"]).shape) == 2
        ):
            self.phase_offsets = np.array(self.params["phase_offset"])
            assert (
                len(self.phase_offsets) == self.params["n"]
            ), "number of phase offsets supplied incompatible with number of neurons"
        else:
            if self.params["phase_offset_distribution"] == "uniform":
                self.phase_offsets = np.random.uniform(
                    0, 2 * np.pi, size=(self.params["n"], 2)
                )
            elif self.params["phase_offset_distribution"] == "delta":
                phase_offset = self.params["phase_offset"] or np.array([0, 0])
                self.phase_offsets = np.ones((self.params["n"], 2)) * np.array(
                    phase_offset
                )
            elif self.params["phase_offset_distribution"] == "grid":
                self.phase_offsets = self.set_phase_offsets_on_grid()
            else:
                raise ValueError("phase offset distribution not recognised")

        # Initialise orientations for each grid cell
        if hasattr(self.params["orientation"], "__len__"):
            self.orientations = np.array(self.params["orientation"])
            assert (
                len(self.orientations) == self.params["n"]
            ), "number of orientations supplied incompatible with number of neurons"
        else:
            if self.params["orientation_distribution"] == "uniform":
                self.orientations = np.random.uniform(
                    0, 2 * np.pi, size=(self.params["n"],)
                )
            elif self.params["orientation_distribution"] == "delta":
                orientation = self.params["orientation"] or 0
                self.orientations = np.ones(self.params["n"]) * orientation
            else:
                raise ValueError("orientation distribution not recognised")

        # Initialise grid cells
        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "grid cells only available in 2D"

        w = []
        for i in range(self.n):
            w1 = np.array([1, 0])
            w1 = utils.rotate(w1, self.orientations[i])
            w2 = utils.rotate(w1, np.pi / 3)
            w3 = utils.rotate(w1, 2 * np.pi / 3)
            w.append(np.array([w1, w2, w3]))
        self.w = np.array(w)

        if ratinabox.verbose is True:
            print(
                "GridCells successfully initialised. You can also manually set their gridscale (GridCells.gridscales), offsets (GridCells.phase_offsets) and orientations (GridCells.w1, GridCells.w2,GridCells.w3 give the cosine vectors)"
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the grid cells.
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
        origin = self.gridscales.reshape(-1, 1) * self.phase_offsets / (2 * np.pi)
        vecs = utils.get_vectors_between(origin, pos)  # shape = (N_cells,N_pos,2)
        w1 = np.tile(np.expand_dims(self.w[:, 0, :], axis=1), reps=(1, pos.shape[0], 1))
        w2 = np.tile(np.expand_dims(self.w[:, 1, :], axis=1), reps=(1, pos.shape[0], 1))
        w3 = np.tile(np.expand_dims(self.w[:, 2, :], axis=1), reps=(1, pos.shape[0], 1))
        gridscales = np.tile(
            np.expand_dims(self.gridscales, axis=1), reps=(1, pos.shape[0])
        )
        phi_1 = ((2 * np.pi) / gridscales) * (vecs * w1).sum(axis=-1)
        phi_2 = ((2 * np.pi) / gridscales) * (vecs * w2).sum(axis=-1)
        phi_3 = ((2 * np.pi) / gridscales) * (vecs * w3).sum(axis=-1)

        if self.description == "three_rectified_cosines":
            firingrate = (1 / 3) * ((np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3)))
            firingrate[firingrate < 0] = 0
        elif self.description == "three_shifted_cosines":
            firingrate = (2 / 3) * (
                (1 / 3) * (np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3)) + (1 / 2)
            )

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def set_phase_offsets_on_grid(self):
        """Set non-random phase_offsets. Most offsets (n_on_grid, the largest square numer before self.n) will tile a grid of 0 to 2pi in x and 0 to 2pi in y, while the remainings (cell number: n - n_on_grid) are random."""
        n_x = int(np.sqrt(self.n))
        n_y = self.n // n_x
        n_remaining = self.n - n_x * n_y

        dx = 2 * np.pi / n_x
        dy = 2 * np.pi / n_y

        grid = np.mgrid[
            (0 + dx / 2) : (2 * np.pi - dx / 2) : (n_x * 1j),
            (0 + dy / 2) : (2 * np.pi - dy / 2) : (n_y * 1j),
        ]
        grid = grid.reshape(2, -1).T
        remaining = np.random.uniform(0, 2 * np.pi, size=(n_remaining, 2))

        all_offsets = np.vstack([grid, remaining])

        return all_offsets