from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.utils import *

import matplotlib

import copy
import numpy as np
import math


class FieldOfViewNeurons(Neurons):
    """FieldOfViewNeurons are collection of boundary vector cells or object vector cells organised so as to represent the local field of view i.e. what walls or objects the agent can "see" in the local vicinity. They work as follow:

    A "manifold" of boundary vector cells (BVCs) or object vector cells (OVCs) tiling the agents FoV is initialised. Users define the radius and angular extent of this manifold which determine the distances and angles available in the FoV. These default to a 180 degree FoV from distance of 0 to 20cm. Each point on this manifold is therefore defined  by tuple (angle and radius), or (θ,r). Egocentric cells are initialised which tile this manifold uniformly, the cell at position (θ,r) will fire with a preference for boundaries (in the case of BVCs) or objects (for OVCs) a distance r from the agent at an angle θ relative to the current heading direction (they are "egocentric"). The position on the manifold determines the angular and distance prefences of the BVC in egocentric coordinates. Thus only if the part of the manifold where the cell sits crosses a wall or object, will that cell fire. Thus only the cells on the part of a manifold touching or close to a wall/object will fire.

    By default these are BVCs (the agent can only "see" walls"), but if cell_type == 'OVC' then for each unique object type present in the Environment a manifold of these cells is created. So if there are two "types " of objects (red objects and blue objects) then ne manifold will reveal the presence of red objects in the Agents FoV and the other blue objects. See Neurons.ObjectVectorCells for more information about how OVCs work.

    The tiling resolution is given by the `spatial_resolution` param in the input dictionary.

    In order to visuale the manifold, we created a plotting function. First make a trajectory figure, then pass this into the plotting func:
        >>> fig, ax = Ag.plot_trajectory()
        >>> fig, ax = my_FoVNeurons.display_manifold(fig, ax)
    or animate it with
        >>> fig, ax = Ag.animate_trajectory(additional_plot_func=my_FoVNeurons.display_manifold)


    Args:
        Neurons (_type_): _description_
    """

    default_params = {
        "FoV_distance": [0.0, 0.2],  # min and max distances the agent can "see"
        "FoV_angles": [
            0,
            90,
        ],  # angluar FoV in degrees (will be symmetric on both sides, so give range in 0 (forwards) to 180 (backwards)
        "spatial_resolution": 0.04,  # resolution of each BVC tiling FoV
        "cell_type": "BVC",  # FoV neurons can either respond to local boundaries ("BVC") or objects ("OVC")
    }

    def __init__(self, Agent, params={}):

        self.params = copy.deepcopy(__class__.default_params)        
        self.params.update(params)

        # tile FoV within angular and distance specifications given as params
        self.FoV_angles_radians = [a * np.pi / 180 for a in self.params["FoV_angles"]]
        dx = self.params["spatial_resolution"]
        radii = np.arange(
            max(0.01, self.params["FoV_distance"][0]),
            self.params["FoV_distance"][1],
            dx,
        )
        manifold_coords_polar = (
            []
        )  # list of [r,theta] coordinate for distance and ang preferences of BVCs
        for radius in radii:
            dtheta = dx / radius
            right_thetas = np.arange(
                self.FoV_angles_radians[0] + dtheta / 2,
                self.FoV_angles_radians[1],
                dtheta,
            )
            left_thetas = -right_thetas[::-1]
            thetas = np.concatenate((left_thetas, right_thetas))
            for theta in thetas:
                manifold_coords_polar.append([radius, theta])
        self.manifold_coords_polar = np.array(manifold_coords_polar)
        # convert these to x,y coords, so we can plot them relative to the Agent
        self.manifold_coords_euclid = np.array(
            [
                self.manifold_coords_polar[:, 0]
                * np.sin(self.manifold_coords_polar[:, 1]),
                self.manifold_coords_polar[:, 0]
                * np.cos(self.manifold_coords_polar[:, 1]),
            ]
        ).T
        self.n_manifold = len(self.manifold_coords_polar)

        self.params["reference_frame"] = "egocentric"

        # initialise BVCs and set their distance/angular preferences and widths
        if self.params["cell_type"] == "BVC":
            self.params["n"] = self.n_manifold
            super().__init__(Agent, self.params)
            self.super = BoundaryVectorCells(Agent, self.params)
            self.super.tuning_distances = self.manifold_coords_polar[:, 0]
            self.super.sigma_distances = max(
                0.01, self.spatial_resolution / 2
            ) * np.ones(self.n)
            self.super.tuning_angles = self.manifold_coords_polar[:, 1]
            self.super.sigma_angles = self.super.sigma_distances / (
                self.manifold_coords_polar[:, 0]
            )

        elif self.params["cell_type"] == "OVC":
            unique_objects = np.unique(Agent.Environment.objects["object_types"])
            self.params["n"] = len(self.manifold_coords_polar) * len(
                unique_objects
            )  # one manifold for each unique object type
            super().__init__(Agent, self.params)
            self.super = ObjectVectorCells(Agent, self.params)
            for (i, obj_id) in enumerate(unique_objects):
                # for each unique object type, make a manifold of neurons which respond to it at differnet angles/distances tiling the FOV
                sid, eid = i * self.n_manifold, (i + 1) * self.n_manifold
                self.super.tuning_types[sid:eid] = obj_id * np.ones(self.n_manifold)
                self.super.tuning_distances[sid:eid] = self.manifold_coords_polar[:, 0]
                self.super.sigma_distances[sid:eid] = max(
                    0.01, self.spatial_resolution / 2
                ) * np.ones(self.n_manifold)
                self.super.tuning_angles[sid:eid] = self.manifold_coords_polar[:, 1]
                self.super.sigma_angles[sid:eid] = self.super.sigma_distances[
                    sid:eid
                ] / (self.manifold_coords_polar[:, 0])

    def get_state(self, evaluate_at="agent", **kwargs):
        return self.super.get_state(evaluate_at, **kwargs)

    def display_manifold(self, fig=None, ax=None, t=None, object_type=0, **kwargs):
        """Visualises the current firing rate of these cells relative to the Agent. Essentially this plots the "manifold" ontop of the Agent.

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
        vel = self.Agent.history["vel"][t_id]
        fr = self.history["firingrate"][t_id]
        ego_y = vel / np.linalg.norm(
            vel
        )  # this is the "y" direction" in egocentric coords
        ego_x = utils.rotate(ego_y, -np.pi / 2)
        facecolor = "C1"

        if self.cell_type == "OVC":
            sid, eid = int(object_type * self.n_manifold), int(
                (object_type + 1) * self.n_manifold
            )
            fr = fr[sid:eid]
            facecolor = matplotlib.cm.get_cmap(self.Agent.Environment.object_colormap)
            facecolor = facecolor(
                object_type / (self.Agent.Environment.n_object_types - 1 + 1e-8)
            )

        for (i, coord) in enumerate(self.manifold_coords_euclid):
            [x, y] = coord
            pos_of_cell = pos - x * ego_x + y * ego_y
            facecolor = list(matplotlib.colors.to_rgba(facecolor))
            facecolor[-1] = max(0, min(1, fr[i] / (0.1 * self.super.max_fr)))
            circ = matplotlib.patches.Circle(
                (pos_of_cell[0], pos_of_cell[1]),
                radius=0.5 * self.spatial_resolution,
                linewidth=0.5,
                edgecolor="dimgrey",
                facecolor=facecolor,
                zorder=2.1,
            )
            ax.add_patch(circ)

        return fig, ax
