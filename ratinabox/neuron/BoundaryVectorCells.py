
import ratinabox
from ratinabox import utils
from ratinabox.neuron.Neurons import Neurons


from matplotlib import pyplot as plt
import numpy as np
import scipy

import copy


class BoundaryVectorCells(Neurons):
    """The BoundaryVectorCells class defines a population of Boundary Vector Cells. This class is a subclass of Neurons() and inherits it properties/plotting functions.
    Must be initialised with an Agent and a 'params' dictionary.
    BoundaryVectorCells defines a set of 'n' BVCs cells with random orientations preferences, distance preferences  (these can be set non-randomly of course). We use the model described firstly by Hartley et al. (2000) and more recently de Cothi and Barry (2000).
    Distance preferences of each BVC are drawn fro ma random distribution which can be one of "uniform" (default), "rayleigh", "normal", and "delta" and parameterised by "wall_pref_dist".
    BVCs can have allocentric (mec,subiculum) OR egocentric (ppc, retrosplenial cortex) reference frames.
    List of functions:
        • get_state()
        • boundary_vector_preference_function()
    default_params = {
            "n": 10,
            "reference_frame": "allocentric",
            "pref_wall_dist": 0.15,
            "pref_wall_dist_distribution": "uniform",
            "angle_spread_degrees": 11.25,
            "xi": 0.08,  # as in de cothi and barry 2020
            "beta": 12,
            "dtheta":2, #angular resolution in degrees
            "min_fr": 0,
            "max_fr": 1,
            "name": "BoundaryVectorCells",
        }
    """

    default_params = {
        "n": 10,
        "reference_frame": "allocentric",
        "pref_wall_dist": 0.25,
        "pref_wall_dist_distribution": "uniform",
        "angle_spread_degrees": 11.25,
        "xi": 0.08,
        "beta": 12,
        "dtheta": 2,
        "min_fr": 0,
        "max_fr": 1,
        "name": "BoundaryVectorCells",
    }

    def __init__(self, Agent, params={}):
        """Initialise BoundaryVectorCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Agent, self.params)

        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "boundary cells only possible in 2D"
        assert (
            self.Agent.Environment.boundary_conditions == "solid"
        ), "boundary cells only possible with solid boundary conditions"
        xi = self.xi
        beta = self.beta
        test_direction = np.array([1, 0])
        test_directions = [test_direction]
        test_angles = [0]
        # numerically discretise over 360 degrees
        self.n_test_angles = int(360 / self.dtheta)
        for i in range(self.n_test_angles - 1):
            test_direction_ = utils.rotate(
                test_direction, 2 * np.pi * i * self.dtheta / 360
            )
            test_directions.append(test_direction_)
            test_angles.append(2 * np.pi * i * self.dtheta / 360)
        self.test_directions = np.array(test_directions)
        self.test_angles = np.array(test_angles)
        self.sigma_angles = np.array(
            [(self.angle_spread_degrees / 360) * 2 * np.pi] * self.n
        )
        self.tuning_angles = np.random.uniform(0, 2 * np.pi, size=self.n)

        # define tuning distances from specific distribution in params dict
        if self.pref_wall_dist_distribution == "rayleigh":
            self.tuning_distances = np.random.rayleigh(
                scale=self.pref_wall_dist, size=self.n
            )
        elif self.pref_wall_dist_distribution == "uniform":
            self.tuning_distances = np.random.uniform(
                low=0, high=self.pref_wall_dist * 2, size=self.n
            )
        elif self.pref_wall_dist_distribution == "normal":
            lower, upper = 0, self.Agent.Environment.scale
            mu, sigma = self.pref_wall_dist, self.pref_wall_dist / 2
            self.tuning_distances = scipy.stats.truncnorm.rvs(
                (lower - mu) / sigma,
                (upper - mu) / sigma,
                scale=sigma,
                loc=mu,
                size=self.n,
            )
        elif self.pref_wall_dist_distribution == "delta":
            self.tuning_distances = self.pref_wall_dist * np.ones(self.n)

        self.sigma_distances = self.tuning_distances / beta + xi

        # calculate normalising constants for BVS firing rates in the current environment. Any extra walls you add from here onwards you add will likely push the firingrate up further.
        locs = self.Agent.Environment.discretise_environment(dx=0.04)
        locs = locs.reshape(-1, locs.shape[-1])
        self.cell_fr_norm = np.ones(self.n)
        self.cell_fr_norm = np.max(self.get_state(evaluate_at=None, pos=locs), axis=1)

        if ratinabox.verbose is True:
            print(
                "BoundaryVectorCells (BVCs) successfully initialised. You can also manually set their orientation preferences (BVCs.tuning_angles, BVCs.sigma_angles), distance preferences (BVCs.tuning_distances, BVCs.sigma_distances)."
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Here we implement the same type if boundary vector cells as de Cothi et al. (2020), who follow Barry & Burgess, (2007). See equations there.
        The way we do this is a little complex. We will describe how it works from a single position (but remember this can be called in a vectorised manner from an array of positons in parallel)
            1. An array of normalised "test vectors" span, in all directions at small increments, from the current position
            2. These define an array of line segments stretching from [pos, pos+test vector]
            3. Where these line segments collide with all walls in the environment is established, this uses the function "utils.vector_intercepts()"
            4. This pays attention to only consider the first (closest) wall forawrd along a line segment. Walls behind other walls are "shaded" by closer walls. Its a little complex to do this and requires the function "boundary_vector_preference_function()"
            5. Now that, for every test direction, the closest wall is established it is simple a process of finding the response of the neuron to that wall segment at that angle (multiple of two gaussians, see de Cothi (2020)) and then summing over all wall segments for all test angles.
        We also apply a check in the middle to utils.rotate the reference frame into that of the head direction of the agent iff self.reference_frame='egocentric'.
        By default position is taken from the Agent and used to calculate firing rates. This can also by passed directly (evaluate_at=None, pos=pass_array_of_positions) or you can use all the positions in the environment (evaluate_at="all").
        """
        if evaluate_at == "agent":
            pos = self.Agent.pos
        elif evaluate_at == "all":
            pos = self.Agent.Environment.flattened_discrete_coords
        else:
            pos = kwargs["pos"]
        pos = np.array(pos)

        N_cells = self.n
        pos = pos.reshape(-1, pos.shape[-1])  # (N_pos,2)
        N_pos = pos.shape[0]
        N_test = self.test_angles.shape[0]
        pos_line_segments = np.tile(
            np.expand_dims(np.expand_dims(pos, axis=1), axis=1), reps=(1, N_test, 2, 1)
        )  # (N_pos,N_test,2,2)
        test_directions_tiled = np.tile(
            np.expand_dims(self.test_directions, axis=0), reps=(N_pos, 1, 1)
        )  # (N_pos,N_test,2)
        pos_line_segments[:, :, 1, :] += test_directions_tiled  # (N_pos,N_test,2,2)
        pos_line_segments = pos_line_segments.reshape(-1, 2, 2)  # (N_pos x N_test,2,2)
        walls = self.Agent.Environment.walls  # (N_walls,2,2)
        N_walls = walls.shape[0]
        pos_lineseg_wall_intercepts = utils.vector_intercepts(
            pos_line_segments, walls
        )  # (N_pos x N_test,N_walls,2)
        pos_lineseg_wall_intercepts = pos_lineseg_wall_intercepts.reshape(
            (N_pos, N_test, N_walls, 2)
        )  # (N_pos,N_test,N_walls,2)
        dist_to_walls = pos_lineseg_wall_intercepts[
            :, :, :, 0
        ]  # (N_pos,N_test,N_walls)
        first_wall_for_each_direction = self.boundary_vector_preference_function(
            pos_lineseg_wall_intercepts
        )  # (N_pos,N_test,N_walls)
        first_wall_for_each_direction_id = np.expand_dims(
            np.argmax(first_wall_for_each_direction, axis=-1), axis=-1
        )  # (N_pos,N_test,1)
        dist_to_first_wall = np.take_along_axis(
            dist_to_walls, first_wall_for_each_direction_id, axis=-1
        ).reshape(
            (N_pos, N_test)
        )  # (N_pos,N_test)
        # reshape everything to have shape (N_cell,N_pos,N_test)

        test_angles = np.tile(
            np.expand_dims(np.expand_dims(self.test_angles, axis=0), axis=0),
            reps=(N_cells, N_pos, 1),
        )  # (N_cell,N_pos,N_test)

        # if egocentric references frame shift angle into coordinate from of heading direction of agent
        if self.reference_frame == "egocentric":
            if evaluate_at == "agent":
                vel = self.Agent.velocity
            elif "vel" in kwargs.keys():
                vel = kwargs["vel"]
            else:
                vel = np.array([1, 0])
            vel = np.array(vel)
            head_direction_angle = utils.get_angle(vel)
            test_angles = test_angles - head_direction_angle

        tuning_angles = np.tile(
            np.expand_dims(np.expand_dims(self.tuning_angles, axis=-1), axis=-1),
            reps=(1, N_pos, N_test),
        )  # (N_cell,N_pos,N_test)
        sigma_angles = np.tile(
            np.expand_dims(
                np.expand_dims(np.array(self.sigma_angles), axis=-1),
                axis=-1,
            ),
            reps=(1, N_pos, N_test),
        )  # (N_cell,N_pos,N_test)
        tuning_distances = np.tile(
            np.expand_dims(np.expand_dims(self.tuning_distances, axis=-1), axis=-1),
            reps=(1, N_pos, N_test),
        )  # (N_cell,N_pos,N_test)
        sigma_distances = np.tile(
            np.expand_dims(np.expand_dims(self.sigma_distances, axis=-1), axis=-1),
            reps=(1, N_pos, N_test),
        )  # (N_cell,N_pos,N_test)
        dist_to_first_wall = np.tile(
            np.expand_dims(dist_to_first_wall, axis=0), reps=(N_cells, 1, 1)
        )  # (N_cell,N_pos,N_test)

        g = utils.gaussian(
            dist_to_first_wall, tuning_distances, sigma_distances, norm=1
        ) * utils.von_mises(
            test_angles, tuning_angles, sigma_angles, norm=1
        )  # (N_cell,N_pos,N_test)

        firingrate = g.sum(axis=-1)  # (N_cell,N_pos)
        firingrate = firingrate / np.expand_dims(self.cell_fr_norm, axis=-1)
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def boundary_vector_preference_function(self, x):
        """This is a random function needed to efficiently produce boundary vector cells. x is any array of final dimension shape shape[-1]=2. As I use it here x has the form of the output of utils.vector_intercepts. I.e. each point gives shape[-1]=2 lambda values (lam1,lam2) for where a pair of line segments intercept. This function gives a preference for each pair. Preference is -1 if lam1<0 (the collision occurs behind the first point) and if lam2>1 or lam2<0 (the collision occurs ahead of the first point but not on the second line segment). If neither of these are true it's 1/x (i.e. it prefers collisions which are closest).
        Args:
            x (array): shape=(any_shape...,2)
        Returns:
            the preferece values: shape=(any_shape)
        """
        assert x.shape[-1] == 2
        pref = np.piecewise(
            x=x,
            condlist=(
                x[..., 0] > 0,
                x[..., 0] < 0,
                x[..., 1] < 0,
                x[..., 1] > 1,
            ),
            funclist=(
                1 / x[x[..., 0] > 0],
                -1,
                -1,
                -1,
            ),
        )
        return pref[..., 0]

    def plot_BVC_receptive_field(
        self,
        chosen_neurons="all",
        fig=None,
        ax=None,
        autosave=None,
    ):
        """Plots the receptive field (in polar corrdinates) of the BVC cells. For allocentric BVCs "up" in this plot == "North", for egocentric BVCs, up == the head direction of the animals
        Args:
            chosen_neurons: Which neurons to plot. Can be int, list, array or "all". Defaults to "all".
            fig, ax: the figure/ax object to plot onto (optional)
            autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
        Returns:
            fig, ax
        """

        if chosen_neurons == "all":
            chosen_neurons = np.arange(self.n)
        if type(chosen_neurons) is str:
            if chosen_neurons.isdigit():
                chosen_neurons = np.linspace(0, self.n - 1, int(chosen_neurons)).astype(
                    int
                )
        if fig is None and ax is None:
            fig, ax = plt.subplots(
                1,
                len(chosen_neurons),
                figsize=(3 * len(chosen_neurons), 3 * 1),
                subplot_kw={"projection": "polar"},
            )
        ax = np.array([ax]).reshape(-1)

        r = np.linspace(0, self.Agent.Environment.scale, 20)
        theta = np.linspace(0, 2 * np.pi, int(360 / 5))
        [theta_meshgrid, r_meshgrid] = np.meshgrid(theta, r)

        def bvc_rf(theta, r, mu_r=0.5, sigma_r=0.2, mu_theta=0.5, sigma_theta=0.1):
            theta = utils.pi_domain(theta)
            return utils.gaussian(r, mu_r, sigma_r) * utils.von_mises(
                theta, mu_theta, sigma_theta
            )

        for i, n in enumerate(chosen_neurons):
            mu_r = self.tuning_distances[n]
            sigma_r = self.sigma_angles[n]
            mu_theta = self.tuning_angles[n]
            sigma_theta = self.sigma_angles[n]
            receptive_field = bvc_rf(
                theta_meshgrid, r_meshgrid, mu_r, sigma_r, mu_theta, sigma_theta
            )
            ax[i].grid(False)
            ax[i].pcolormesh(
                theta, r, receptive_field, edgecolors="face", shading="nearest"
            )
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        ratinabox.utils.save_figure(fig, "BVC_receptive_fields", save=autosave)

        return fig, ax