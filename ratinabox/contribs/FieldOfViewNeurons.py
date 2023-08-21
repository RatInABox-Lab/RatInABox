from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.utils import *


import matplotlib
from matplotlib.collections import EllipseCollection

import copy
import numpy as np
import math
import types
from warnings import warn


class FieldOfViewNeurons(Neurons):
    """FieldOfViewNeurons are collection of boundary vector cells or object vector cells organised so as to represent the local field of view i.e. what walls or objects the agent can "see" in the local vicinity. They work as follow:

    A "manifold" of boundary vector cells (BVCs) or object vector cells (OVCs) tiling the agents FoV is initialised. Users define the radius and angular extent of this manifold which determine the distances and angles available in the FoV. These default to a 180 degree FoV from distance of 0 to 20cm. Each point on this manifold is therefore defined  by tuple (angle and radius), or (θ,r). Egocentric cells are initialised which tile this manifold (uniformly or growing with radius), the cell at position (θ,r) will fire with a preference for boundaries (in the case of BVCs) or objects (for OVCs) a distance r from the agent at an angle θ relative to the current heading direction (they are "egocentric"). Thus only if the part of the manifold where the cell sits crosses a wall or object, will that cell fire. Thus only the cells on the part of a manifold touching or close to a wall/object will fire.

    By default these are BVCs (the agent can only "see" walls), but if cell_type == 'OVC' then for each unique object type present in the Environment a manifold of these cells is created. So if there are two "types " of objects (red objects and blue objects) then one manifold will reveal the presence of red objects in the Agents FoV and the other blue objects. See Neurons.ObjectVectorCells for more information about how OVCs work.

    The tiling resolution (the spatial and angular tuning sizes of the smallest cells receptive fields) is given by the `spatial_resolution` param in the input dictionary.

    In order to visuale the manifold, we created a plotting function. First make a trajectory figure, then pass this into the plotting func:
        >>> fig, ax = Ag.plot_trajectory()
        >>> fig, ax = my_FoVNeurons.display_manifold(fig, ax)
    or animate it with
        >>> fig, ax = Ag.animate_trajectory(additional_plot_func=my_FoVNeurons.display_manifold)

    """

    default_params = {
        "FoV_distance": [0.0, 0.2],  # min and max distances the agent can "see"
        "FoV_angles": [
            0,
            90,
        ],  # angluar FoV in degrees (will be symmetric on both sides, so give range in 0 (forwards) to 180 (backwards)
        "spatial_resolution": 0.04,  # resolution of each BVC tiling FoV
        "manifold_function": "uniform",  # whether all cells have "uniform" receptive field sizes or they grow ("hartley") with radius.
        "cell_type": "BVC",  # FoV neurons can either respond to local boundaries ("BVC") or objects ("OVC")
    }

    def __init__(self, Agent, params={}):

        """This throws a deprecation warning on initialization."""
        warn(f'{self.__class__.__name__} will be deprecated. Please use the new class that can be imported using: from ratinabox.Neurons import FieldOfViewOVCs, FieldOfViewBVCs', DeprecationWarning, stacklevel=2)
        

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        # initialise the manifold of BVCs/OVCs tiling the Agents FoV

        if (
            type(self.params["manifold_function"]) is types.FunctionType
        ):  # allow users pass their own function which generates the manifold dictionary
            manifold = self.params["manifold_function"]()
        else:
            manifold = self.get_manifold(
                FoV_distance=self.params["FoV_distance"],
                FoV_angles=self.params["FoV_angles"],
                spatial_resolution=self.params["spatial_resolution"],
                manifold_function=self.params["manifold_function"],
            )
        self.manifold = manifold

        self.params["reference_frame"] = "egocentric"

        # initialise BVCs and set their distance/angular preferences and widths
        if self.params["cell_type"] == "BVC":
            self.params["n"] = manifold["n"]  # one manifold,
            super().__init__(Agent, self.params)
            self.super = BoundaryVectorCells(Agent, self.params)
            self.super.tuning_distances = manifold["mu_d"]
            self.super.sigma_distances = manifold["sigma_d"]
            self.super.tuning_angles = manifold["mu_theta"]
            self.super.sigma_angles = manifold["sigma_theta"]

            # normalise the firing rates of the cells so their maximum fr is approximately given by max_fr
            locs = self.Agent.Environment.discretise_environment(dx=0.04)
            locs = locs.reshape(-1, locs.shape[-1])
            self.super.cell_fr_norm = np.ones(self.n)
            self.super.cell_fr_norm = np.max(
                self.get_state(evaluate_at=None, pos=locs), axis=1
            )

        elif self.params["cell_type"] == "OVC":
            unique_objects = np.unique(Agent.Environment.objects["object_types"])
            self.params["n"] = manifold["n"] * len(
                unique_objects
            )  # one manifold for each unique object type
            super().__init__(Agent, self.params)
            self.super = ObjectVectorCells(Agent, self.params)
            for i, obj_id in enumerate(unique_objects):
                # for each unique object type, make a manifold of neurons which respond to it at different angles/distances tiling the FOV
                sid, eid = i * manifold["n"], (i + 1) * manifold["n"]
                self.super.tuning_types[sid:eid] = obj_id * np.ones(manifold["n"])
                self.super.tuning_distances[sid:eid] = manifold["mu_d"]
                self.super.sigma_distances[sid:eid] = manifold["sigma_d"]
                self.super.tuning_angles[sid:eid] = manifold["mu_theta"]
                self.super.sigma_angles[sid:eid] = manifold["sigma_theta"]

    def get_state(self, evaluate_at="agent", **kwargs):
        return self.super.get_state(evaluate_at, **kwargs)

    def get_manifold(
        self,
        FoV_distance=[0.0, 0.2],
        FoV_angles=[0, 90],
        spatial_resolution=0.04,
        manifold_function="uniform",
    ):
        """Returns the manifold of parameters for the B/OVCs tiling the Agents FoV.

         Args:
            • FoV_distance (list): [min,max] distance from the agent to tile the manifold
            • FoV_angles (list): [min,max] angular extent of the manifold in degrees
            • spatial_resolution (float): size of each receptive field ("uniform") or the smallest receptive field ("hartley")
            • manifold_function (str): "uniform" (all receptive fields the same size or "hartley" (further away receptive fields are larger in line with Hartley et al. 2000 eqn (1))

        Returns:
            • manifold (dict): a dictionary containing all details of the manifold of cells including
                • "mu_d" : distance preferences
                • "mu_theta" : angular preferences (radians)
                • "sigma_d" : distance tunings
                • "sigma_theta" : angular tunings (radians)
                • "manifold_coords_polar" : coordinates in egocentric polar coords [d, theta]
                • "manifold_coords_euclid" : coordinates in egocentric euclidean coords [x,y]
                • "n" : number of cells on the manifold

        """
        FoV_angles_radians = [a * np.pi / 180 for a in FoV_angles]
        (mu_d, mu_theta, sigma_d, sigma_theta) = ([], [], [], [])

        if manifold_function == "uniform":
            dx = spatial_resolution
            radii = np.arange(max(0.01, FoV_distance[0]), FoV_distance[1], dx)
            for radius in radii:
                dtheta = dx / radius
                right_thetas = np.arange(
                    FoV_angles_radians[0] + dtheta / 2,
                    FoV_angles_radians[1],
                    dtheta,
                )
                left_thetas = -right_thetas[::-1]
                thetas = np.concatenate((left_thetas, right_thetas))
                for theta in thetas:
                    mu_d.append(radius)
                    mu_theta.append(theta)
                    sigma_d.append(spatial_resolution)
                    sigma_theta.append(spatial_resolution / radius)

        elif manifold_function == "hartley":
            # Hartley model says that the spatial tuning width of a B/OVC should be proportional to the distance from the agent according to
            # sigma_d = mu_d / beta + xi where beta := 12 and xi := 0.08 m  are constants.
            # In this case, however, we want to force the smallest cells to have sigma_d = spatial_resolution setting the constraint that xi = spatial_resolution - min_radius / beta
            radius = max(0.01, FoV_distance[0])
            beta = 5  # smaller means larger rate of increase of cell size with radius
            xi = spatial_resolution - radius / beta
            while radius < FoV_distance[1]:
                resolution = xi + radius / beta  # spatial resolution of this row
                dtheta = resolution / radius
                if dtheta / 2 > FoV_angles_radians[1]:
                    right_thetas = np.array(
                        [FoV_angles_radians[0] + dtheta / 2]
                    )  # at least one cell in this row
                else:
                    right_thetas = np.arange(
                        FoV_angles_radians[0] + dtheta / 2,
                        FoV_angles_radians[1],
                        dtheta,
                    )
                left_thetas = -right_thetas[::-1]
                thetas = np.concatenate((left_thetas, right_thetas))
                for theta in thetas:
                    mu_d.append(radius)
                    mu_theta.append(theta)
                    sigma_d.append(resolution)
                    sigma_theta.append(resolution / radius)
                # the radius of the next row of cells is found by solving the simultaneous equations:
                # • r2 = r1 + sigma1/2 + sigma2/2 (the radius, i.e. sigma/2, of the second cell just touches the first cell)
                # • sigma2 = r2/beta + xi (Hartleys rule)
                # the solution is: r2 = (2*r1 + sigma1 +xi) / (2 - 1/beta)
                radius = (2 * radius + resolution + xi) / (2 - 1 / beta)
        mu_d, mu_theta, sigma_d, sigma_theta = (
            np.array(mu_d),
            np.array(mu_theta),
            np.array(sigma_d),
            np.array(sigma_theta),
        )
        manifold_coords_polar = np.array([mu_d, mu_theta]).T
        manifold_coords_euclid = np.array(
            [mu_d * np.sin(mu_theta), mu_d * np.cos(mu_theta)]
        ).T
        n = len(mu_d)

        manifold = {
            "mu_d": mu_d,
            "mu_theta": mu_theta,
            "sigma_d": sigma_d,
            "sigma_theta": sigma_theta,
            "manifold_coords_polar": manifold_coords_polar,
            "manifold_coords_euclid": manifold_coords_euclid,
            "n": n,
        }

        return manifold

    def display_manifold(self, fig=None, ax=None, t=None, object_type=0, **kwargs):
        """Visualises the current firing rate of these cells relative to the Agent. Essentially this plots the "manifold" ontop of the Agent. Each cell is plotted as an ellipse where the alpha-value of its facecolor reflects the current firing rate (normalised against the approximate maximum firing rate for all cells, but, take this just as a visualisation)

        Args:
        • fig, ax: the matplotlib fig, ax objects to plot on (if any), otherwise will plot the Environment
        • t (float): time to plot at
        • object_type (int): if self.cell_type=="OVC", which object type to plot

        Returns:
            fig, ax: with the
        """
        if t is None:
            t = self.Agent.history["t"][-1]
        t_id = np.argmin(np.abs(np.array(self.Agent.history["t"]) - t))

        if fig is None and ax is None:
            fig, ax = self.Agent.plot_trajectory(t_start=t - 10, t_end=t, **kwargs)

        pos = self.Agent.history["pos"][t_id]
        head_direction = self.Agent.history["head_direction"][t_id]
        head_direction_angle = (180 / np.pi) * (
            ratinabox.utils.get_angle(head_direction) - np.pi / 2
        )  # head direction angle (CCW from true North)
        fr = self.history["firingrate"][t_id]
        ego_y = head_direction / np.linalg.norm(
            head_direction
        )  # this is the "y" direction" in egocentric coords
        ego_x = utils.rotate(ego_y, -np.pi / 2)
        facecolor = self.color or "C1"

        if self.cell_type == "OVC":
            sid, eid = int(object_type * self.manifold["n"]), int(
                (object_type + 1) * self.manifold["n"]
            )
            fr = fr[sid:eid]
            facecolor = matplotlib.cm.get_cmap(self.Agent.Environment.object_colormap)
            facecolor = facecolor(
                object_type / (self.Agent.Environment.n_object_types - 1 + 1e-8)
            )

        # TODO redo this with ellipse collections
        for i, coord in enumerate(self.manifold["manifold_coords_euclid"]):
            [x, y] = coord
            pos_of_cell = pos - x * ego_x + y * ego_y
            facecolor = list(matplotlib.colors.to_rgba(facecolor))
            facecolor[-1] = max(
                0, min(1, fr[i] / (0.5 * self.super.max_fr))
            )  # set the alpha value
            # circ = matplotlib.patches.Circle(
            #     (pos_of_cell[0], pos_of_cell[1]),
            #     radius=0.5 * self.spatial_resolution,
            #     linewidth=0.5,
            #     edgecolor="dimgrey",
            #     facecolor=facecolor,
            #     zorder=2.1,
            # )
            ellipse = matplotlib.patches.Ellipse(
                (pos_of_cell[0], pos_of_cell[1]),
                width=self.manifold["sigma_theta"][i] * self.manifold["mu_d"][i],
                height=self.manifold["sigma_d"][i],
                angle=1.0 * head_direction_angle
                + self.manifold["mu_theta"][i] * 180 / np.pi,
                linewidth=0.5,
                edgecolor="dimgrey",
                facecolor=facecolor,
                zorder=2.1,
            )
            ax.add_patch(ellipse)

        return fig, ax
