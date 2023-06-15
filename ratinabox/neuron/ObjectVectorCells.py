import ratinabox
from ratinabox import utils
from ratinabox.neuron.Neurons import Neurons

import numpy as np

import copy


class ObjectVectorCells(Neurons):
    """Initialises ObjectVectorCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
    ObjectVectorCells respond to Objects inside the Environment (2D only). Add objects to the environment using the `Env.add_object()` method. Each OVC has a prefrerred type (which "type" of object it responds to), tuning angle, and tuning distance (direction and distance from object cell will preferentially fire at).
    Reference frame can be allocentric or egocentric. In the latter case the tuning angle is relative to the heading direction of the agent.
    default_params = {
        "n": 10, #each will be randomly assigned an object type, tuning angle and tuning distance
        "name": "ObjectVectorCell",
        "walls_occlude":True, #whether walls occlude OVC firing
        "reference_frame":"allocentric", #"or "egocentric" (equivalent to field of view neurons)
        "angle_spread_degrees":15, #spread of von Mises angular preferrence functinon for each OVC, you can also set this array manually after initialisation
        "pref_object_dist": 0.25, # distance preference drawn from a Rayleigh with this sigma. How far away from object the OVC fires. Can set this array manually after initialisation.
        "xi": 0.08, #parameters determining the distance preferrence function std given the preferred distance. See BoundaryVectorCells or de cothi and barry 2020
        "beta": 12,
        "max_fr":1, # likely max firing rate of an OVC
        "min_fr":0, # likely min firing rate
    }
    """

    default_params = {
        "n": 10,
        "name": "ObjectVectorCell",
        "walls_occlude": True,
        "reference_frame": "allocentric",
        "angle_spread_degrees": 15,
        "pref_object_dist": 0.25,
        "xi": 0.08,
        "beta": 12,
        "max_fr": 1,
        "min_fr": 0,
    }

    def __init__(self, Agent, params={}):
        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "object vector cells only possible in 2D"

        super().__init__(Agent, self.params)

        self.object_locations = self.Agent.Environment.objects["objects"]
        assert len(self.object_locations) > 0, print(
            "No objects in Environments, add objects using `Env.add_object(object_position=[x,y]) method"
        )

        # preferred object types, distance and angle to objects and their tuning widths (set these yourself if needed)
        self.object_types = self.Agent.Environment.objects["object_types"]
        self.tuning_types = np.random.choice(
            np.unique(self.object_types), replace=True, size=(self.n,)
        )
        self.tuning_angles = np.random.uniform(0, 2 * np.pi, size=self.n)
        self.tuning_distances = np.random.rayleigh(
            scale=self.pref_object_dist, size=self.n
        )
        self.sigma_distances = self.tuning_distances / self.beta + self.xi
        self.sigma_angles = np.array(
            [(self.angle_spread_degrees / 360) * 2 * np.pi] * self.n
        )

        if self.walls_occlude == True:
            self.wall_geometry = "line_of_sight"
        else:
            self.wall_geometry = "euclidean"

        # normalises activity over the environment
        locs = self.Agent.Environment.discretise_environment(dx=0.04)
        locs = locs.reshape(-1, locs.shape[-1])

        if ratinabox.verbose is True:
            print(
                "ObjectVectorCells (OVCs) successfully initialised. You can also manually set their orientation preferences (OVCs.tuning_angles, OVCs.sigma_angles), distance preferences (OVCs.tuning_distances, OVCs.sigma_distances)."
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the ObjectVectorCells.
        The way we do this is a little complex. We will describe how it works from a single position to a single OVC (but remember this can be called in a vectorised manner from an array of positons in parallel and there are in principle multiple OVCs)
            1. A vector from the position to the OVC is calculated.
            2. The bearing of this vector is calculated and its length. Note if self.field_of_view == True then the bearing is relative to the heading direction of the agent (along its current velocity), not true-north.
            3. Since the distance to the OVC is calculated taking the environment into account if there is a wall occluding the agent from the obvject this object will not fire.
            4. It is now simple to calculate the firing rate of the cell. Each OVC has a preferred distance and angle away from it which cause it to fire. Its a multiple of a gaussian (distance) and von mises (for angle) which creates teh eventual firing rate.
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
        pos = pos.reshape(-1, pos.shape[-1])  # (N_pos, 2)
        N_pos = pos.shape[0]
        N_cells = self.n
        N_objects = len(self.object_locations)

        (
            distances_to_objects,
            vectors_to_objects,
        ) = self.Agent.Environment.get_distances_between___accounting_for_environment(
            pos,
            self.object_locations,
            return_vectors=True,
            wall_geometry=self.wall_geometry,
        )  # (N_pos,N_objects) (N_pos,N_objects,2)
        flattened_vectors_to_objects = vectors_to_objects.reshape(
            -1, 2
        )  # (N_pos x N_objects, 2)
        bearings_to_objects = (
            utils.get_angle(flattened_vectors_to_objects, is_array=True).reshape(
                N_pos, N_objects
            )
            - np.pi
        )  # (N_pos,N_objects) #vectors go from pos2 to pos1 so must do subtract pi from bearing
        if self.reference_frame == "egocentric":
            if evaluate_at == "agent":
                vel = self.Agent.velocity
            elif "vel" in kwargs.keys():
                vel = kwargs["vel"]
            else:
                vel = np.array([1, 0])
                print(
                    "Field of view OVCs require a velocity vector but none was passed. Using [1,0]"
                )
            head_bearing = utils.get_angle(vel)
            bearings_to_objects -= head_bearing  # account for head direction

        tuning_distances = np.tile(
            np.expand_dims(np.expand_dims(self.tuning_distances, axis=0), axis=0),
            reps=(N_pos, N_objects, 1),
        )  # (N_pos,N_objects,N_cell)
        sigma_distances = np.tile(
            np.expand_dims(np.expand_dims(self.sigma_distances, axis=0), axis=0),
            reps=(N_pos, N_objects, 1),
        )  # (N_pos,N_objects,N_cell)
        tuning_angles = np.tile(
            np.expand_dims(np.expand_dims(self.tuning_angles, axis=0), axis=0),
            reps=(N_pos, N_objects, 1),
        )  # (N_pos,N_objects,N_cell)
        sigma_angles = np.tile(
            np.expand_dims(np.expand_dims(self.sigma_angles, axis=0), axis=0),
            reps=(N_pos, N_objects, 1),
        )  # (N_pos,N_objects,N_cell)
        tuning_types = np.tile(
            np.expand_dims(self.tuning_types, axis=0), reps=(N_objects, 1)
        )
        object_types = np.tile(
            np.expand_dims(self.object_types, axis=-1), reps=(1, N_cells)
        )

        distances_to_objects = np.tile(
            np.expand_dims(distances_to_objects, axis=-1), reps=(1, 1, N_cells)
        )  # (N_pos,N_objects,N_cells)
        bearings_to_objects = np.tile(
            np.expand_dims(bearings_to_objects, axis=-1), reps=(1, 1, N_cells)
        )  # (N_pos,N_objects,N_cells)

        firingrate = utils.gaussian(
            distances_to_objects, tuning_distances, sigma_distances, norm=1
        ) * utils.von_mises(
            bearings_to_objects, tuning_angles, sigma_angles, norm=1
        )  # (N_pos,N_objects,N_cell)

        tuning_mask = np.expand_dims(
            np.array(object_types == tuning_types, int), axis=0
        )  # (1,N_objects,N_cells)
        firingrate *= tuning_mask
        firingrate = np.sum(
            firingrate, axis=1
        ).T  # (N_cell,N_pos), sum over objects which this cell is selective to
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate