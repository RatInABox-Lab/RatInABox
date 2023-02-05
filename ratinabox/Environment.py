import ratinabox 

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from ratinabox import utils

"""ENVIRONMENT"""


class Environment:
    """Environment class: defines the Environment in which the Agent lives.
    This class needs no other classes to initialise it. It exists independently of any Agents or Neurons.

    A default parameters dictionary (with descriptions) can be found in __init__()

    List of functions...
        ...that you might use:
            • add_wall()
            • plot_environment()
        ...that you probably won't directly use:
            • sample_positions()
            • discretise_environment()
            • get_vectors_between___accounting_for_environment()
            • get_distances_between___accounting_for_environment()
            • check_if_posision_is_in_environment()
            • check_wall_collisions()
            • vectors_from_walls()
            • apply_boundary_conditions()

    The default_params are
    default_params = {
            "dimensionality": "2D",
            "boundary_conditions": "solid",
            "scale": 1,
            "aspect": 1,
            "dx": 0.01,
        }
    """

    def __init__(self, params={}):
        """Initialise Environment, takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary.

        Args:
            params (dict, optional). Defaults to {}.
        """
        default_params = {
            "dimensionality": "2D",  # 1D or 2D environment
            "boundary_conditions": "solid",  # solid vs periodic
            "scale": 1,  # scale of environment (in metres)
            "aspect": 1,  # x/y aspect ratio for the (rectangular) 2D environment
            "dx": 0.01,  # discretises the environment (for plotting purposes only)
        }

        default_params.update(params)
        self.params = default_params
        utils.update_class_params(self, self.params)

        if self.dimensionality == "1D":
            self.extent = np.array([0, self.scale])
            self.centre = np.array([self.scale / 2, self.scale / 2])

        self.walls = np.array([])
        if self.dimensionality == "2D":
            if self.boundary_conditions != "periodic":
                self.walls = np.array(
                    [
                        [[0, 0], [0, self.scale]],
                        [[0, self.scale], [self.aspect * self.scale, self.scale]],
                        [
                            [self.aspect * self.scale, self.scale],
                            [self.aspect * self.scale, 0],
                        ],
                        [[self.aspect * self.scale, 0], [0, 0]],
                    ]
                )
            self.centre = np.array([self.aspect * self.scale / 2, self.scale / 2])
            self.extent = np.array([0, self.aspect * self.scale, 0, self.scale])
        self.params["extent"] = self.extent
        self.params["centre"] = self.centre

        # save some prediscretised coords (useful for plotting rate maps later)
        self.discrete_coords = self.discretise_environment(dx=self.dx)
        self.flattened_discrete_coords = self.discrete_coords.reshape(
            -1, self.discrete_coords.shape[-1]
        )

        if ratinabox.verbose is True:
            print(
                f"\nAn Environment has been initialised with parameters: {self.params}. Use Env.add_wall() to add a wall into the Environment. Plot Environment using Env.plot_environment()."
            )

        return

    def add_wall(self, wall):
        """Add a wall to the (2D) environment.
        Extends self.walls array to include one new wall.
        Args:
            wall (np.array): 2x2 array [[x1,y1],[x2,y2]]
        """
        assert self.dimensionality == "2D", "can only add walls into a 2D environment"
        wall = np.expand_dims(np.array(wall), axis=0)
        if len(self.walls) == 0:
            self.walls = wall
        else:
            self.walls = np.concatenate((self.walls, wall), axis=0)
        return

    def plot_environment(self, fig=None, ax=None, height=1):
        """Plots the environment on the x axis, dark grey lines show the walls
        Args:
            fig,ax: the fig and ax to plot on (can be None)
            height: if 1D, how many line plots will be stacked (5.5mm per line)
        Returns:
            fig, ax: the environment figures, can be used for further downstream plotting.
        """

        if self.dimensionality == "1D":
            extent = self.extent
            fig, ax = plt.subplots(
                figsize=(2 * (extent[1] - extent[0]), height * (5.5 / 25))
            )
            ax.set_xlim(left=extent[0], right=extent[1])
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.spines["bottom"].set_position("zero")
            ax.spines["top"].set_color("none")
            ax.set_yticks([])
            ax.set_xticks([extent[0], extent[1]])
            ax.set_xlabel("Position / m")

        if self.dimensionality == "2D":
            extent, walls = self.extent, self.walls
            if fig is None and ax is None:
                fig, ax = plt.subplots(
                    figsize=(3 * (extent[1] - extent[0]), 3 * (extent[3] - extent[2]))
                )
            background = matplotlib.patches.Rectangle(
                (extent[0], extent[2]),
                extent[1],
                extent[3],
                facecolor="lightgrey",
                zorder=-1,
            )
            setattr(background, 'name', 'background')
            ax.add_patch(background)
            for wall in walls:
                ax.plot(
                    [wall[0][0], wall[1][0]],
                    [wall[0][1], wall[1][1]],
                    color="grey",
                    linewidth=4,
                    solid_capstyle='round',
                    zorder=1.1,
                )
            ax.set_aspect("equal")
            ax.grid(False)
            ax.axis("off")
            ax.set_xlim(left=extent[0] - 0.03, right=extent[1] + 0.03)
            ax.set_ylim(bottom=extent[2] - 0.03, top=extent[3] + 0.03)
        return fig, ax

    def sample_positions(self, n=10, method="uniform_jitter"):
        """Scatters 'n' locations across the environment which can act as, for example, the centres of gaussian place fields, or as a random starting position.
        If method == "uniform" an evenly spaced grid of locations is returned.  If method == "uniform_jitter" these locations are jittered slightly (i.e. random but span the space). Note; if n doesn't uniformly divide the size (i.e. n is not a square number in a square environment) then the largest number that can be scattered uniformly are found, the remaining are randomly placed.
        Args:
            n (int): number of features
            method: "uniform", "uniform_jittered" or "random" for how points are distributed
            true_random: if True, just randomly scatters point
        Returns:
            array: (n x dimensionality) of positions
        """
        if self.dimensionality == "1D":
            if method == "random":
                positions = np.random.uniform(
                    self.extent[0], self.extent[1], size=(n, 1)
                )
            elif method[:7] == "uniform":
                dx = self.scale / n
                positions = np.arange(0 + dx / 2, self.scale, dx).reshape(-1, 1)
                if method[7:] == "_jitter":
                    positions += np.random.uniform(
                        -0.45 * dx, 0.45 * dx, positions.shape
                    )
            return positions

        elif self.dimensionality == "2D":
            if method == "random":
                positions = np.random.uniform(size=(n, 2))
                positions[:, 0] *= self.extent[1] - self.extent[0]
                positions[:, 1] *= self.extent[3] - self.extent[2]
            elif method[:7] == "uniform":
                ex = self.extent
                area = (ex[1] - ex[0]) * (ex[3] - ex[2])
                delta = np.sqrt(area / n)
                x = np.arange(ex[0] + delta / 2, ex[1] - delta / 2 + 1e-6, delta)
                y = np.arange(ex[2] + delta / 2, ex[3] - delta / 2 + 1e-6, delta)
                positions = np.array(np.meshgrid(x, y)).reshape(2, -1).T
                n_uniformly_distributed = positions.shape[0]
                if method[7:] == "_jitter":
                    positions += np.random.uniform(
                        -0.45 * delta, 0.45 * delta, positions.shape
                    )
                n_remaining = n - n_uniformly_distributed
                if n_remaining > 0:
                    positions_remaining = self.sample_positions(
                        n=n_remaining, method="random"
                    )
                    positions = np.vstack((positions, positions_remaining))
            return positions

    def discretise_environment(self, dx=None):
        """Discretises the environment, for plotting purposes.
        Returns an array of positions spanning the environment
        Important: this discretisation is not used for geometry or firing rate calculations which are precise and fundamentally continuous. Its typically used if you want to, say, display the receptive field of a neuron so you want to calculate its firing rate at all points across the environment and plot those.
        Args:
            dx (float): discretisation distance
        Returns:
            array: an (Ny x Mx x 2) array of position coordinates or (Nx x 1) for 1D
        """  # returns a 2D array of locations discretised by dx
        if dx is None:
            dx = self.dx
        [minx, maxx] = list(self.extent[:2])
        self.x_array = np.arange(minx + dx / 2, maxx, dx)
        discrete_coords = self.x_array.reshape(-1, 1)
        if self.dimensionality == "2D":
            [miny, maxy] = list(self.extent[2:])
            self.y_array = np.arange(miny + dx / 2, maxy, dx)[::-1]
            x_mesh, y_mesh = np.meshgrid(self.x_array, self.y_array)
            coordinate_mesh = np.array([x_mesh, y_mesh])
            discrete_coords = np.swapaxes(np.swapaxes(coordinate_mesh, 0, 1), 1, 2)
        return discrete_coords

    def get_vectors_between___accounting_for_environment(
        self, pos1=None, pos2=None, line_segments=None
    ):
        """Takes two position arrays and returns an array of pair-wise vectors from pos2's to pos1's, taking into account boundary conditions. Unlike the global function "utils.get_vectors_between()' (which this calls) this additionally accounts for environment boundary conditions such that if two positions fall on either sides of the boundary AND boundary cons are periodic then the returned shortest-vector actually goes around the loop, not across the environment)...
            pos1 (array): N x dimensionality array of poisitions
            pos2 (array): M x dimensionality array of positions
            line_segments: if you have already calculated line segments from pos1 to pos2 pass this here for quicker evaluation
        Returns:
            N x M x dimensionality array of pairwise vectors
        """
        vectors = utils.get_vectors_between(pos1=pos1, pos2=pos2, line_segments=line_segments)
        if self.boundary_conditions == "periodic":
            flip = np.abs(vectors) > (self.scale / 2)
            vectors[flip] = -np.sign(vectors[flip]) * (
                self.scale - np.abs(vectors[flip])
            )
        return vectors

    def get_distances_between___accounting_for_environment(
        self, pos1, pos2, wall_geometry="euclidean", return_vectors=False
    ):
        """Takes two position arrays and returns the array of pair-wise distances between points, taking into account walls and boundary conditions. Unlike the global function utils.get_distances_between() (which this one, at times, calls) this additionally accounts for the boundaries AND walls in the environment.

        For example, geodesic geometry estimates distance by shortest walk...line_of_sight geometry distance is euclidean but if there is a wall in between two positions (i.e. no line of sight) then the returned distance is "very high"...if boundary conditions are periodic distance is via the shortest possible route, which may or may not go around the back. euclidean geometry essentially ignores walls when calculating distances between two points.
        Allowed geometries, typically passed from the neurons class, are "euclidean", "geodesic" or "line_of_sight"
        Args:
            pos1 (array): N x dimensionality array of poisitions
            pos2 (array): M x dimensionality array of positions
            wall_geometry: how the distance calculation handles walls in the env (can be "euclidean", "line_of_sight" or "geodesic")
            return_vectors (False): If True, returns the distances and the vectors as a tuple
        Returns:
            N x M array of pairwise distances
        """

        line_segments = utils.get_line_segments_between(pos1=pos1, pos2=pos2)
        vectors = self.get_vectors_between___accounting_for_environment(
            pos1=None, pos2=None, line_segments=line_segments
        )

        # shorthand
        dimensionality = self.dimensionality
        boundary_conditions = self.boundary_conditions

        if dimensionality == "1D":
            distances = utils.get_distances_between(vectors=vectors)

        if dimensionality == "2D":
            walls = self.walls
            if wall_geometry == "euclidean":
                distances = utils.get_distances_between(vectors=vectors)

            if wall_geometry == "line_of_sight":
                assert (
                    boundary_conditions == "solid"
                ), "line of sight geometry not available for periodic boundary conditions"
                # if a wall obstructs line-of-sight between two positions, distance is set to 1000
                internal_walls = walls[
                    4:
                ]  # only the internal walls (not room walls) are worth checking
                line_segments_ = line_segments.reshape(-1, *line_segments.shape[-2:])
                wall_obstructs_view_of_cell = utils.vector_intercepts(
                    line_segments_, internal_walls, return_collisions=True
                )
                wall_obstructs_view_of_cell = wall_obstructs_view_of_cell.sum(
                    axis=-1
                )  # sum over walls axis as we don't care which wall it collided with
                wall_obstructs_view_of_cell = wall_obstructs_view_of_cell != 0
                wall_obstructs_view_of_cell = wall_obstructs_view_of_cell.reshape(
                    line_segments.shape[:2]
                )
                distances = utils.get_distances_between(vectors=vectors)
                distances[wall_obstructs_view_of_cell == True] = 1000

            if wall_geometry == "geodesic":
                assert (boundary_conditions == "solid"), "geodesic geometry is not available for periodic boundary conditions"
                assert (len(walls) <= 5), """unfortunately geodesic geomtry is only defined in closed rooms with one additional wall
                (efficient geometry calculations with more than 1 wall are super hard I have discovered!)"""
                distances = utils.get_distances_between(vectors=vectors)
                if len(walls) == 4:
                    pass
                else:
                    wall = walls[4]
                    via_wall_distances = []
                    for part_of_wall in wall:
                        wall_edge = np.expand_dims(part_of_wall, axis=0)
                        if self.check_if_position_is_in_environment(part_of_wall):
                            distances_via_part_of_wall = utils.get_distances_between(
                                pos1, wall_edge
                            ) + utils.get_distances_between(wall_edge, pos2)
                            via_wall_distances.append(distances_via_part_of_wall)
                    via_wall_distances = np.array(via_wall_distances)
                    line_segments_ = line_segments.reshape(
                        -1, *line_segments.shape[-2:]
                    )
                    wall_obstructs_view_of_cell = utils.vector_intercepts(
                        line_segments_,
                        np.expand_dims(wall, axis=0),
                        return_collisions=True,
                    )
                    wall_obstructs_view_of_cell = wall_obstructs_view_of_cell.reshape(
                        line_segments.shape[:2]
                    )
                    flattened_distances = distances.reshape(-1)
                    flattened_wall_obstructs_view_of_cell = wall_obstructs_view_of_cell.reshape(
                        -1
                    )
                    flattened_distances[
                        flattened_wall_obstructs_view_of_cell
                    ] = np.amin(via_wall_distances, axis=0).reshape(-1)[
                        flattened_wall_obstructs_view_of_cell
                    ]
                    distances = flattened_distances.reshape(distances.shape)

        if return_vectors:
            return (distances, vectors)
        else:
            return distances

    def check_if_position_is_in_environment(self, pos):
        """Returns True if pos is INside the environment
        Points EXACTLY on the edge of the environment are NOT classed as being inside the environment. This is relevant in geodesic geometry calculations since routes past the edge of a wall connection with the edge of an environmnet are not valid routes.
        Args:
            pos (array): np.array([x,y])
        Returns:
            bool: True if pos is inside environment.
        """
        pos = np.array(pos).reshape(-1)
        if self.dimensionality == "2D":
            if all([
                (pos[0] > self.extent[0]),
                (pos[0] < self.extent[1]),
                (pos[1] > self.extent[2]),
                (pos[1] < self.extent[3]),
            ]):
                return True
            else:
                return False
        elif self.dimensionality == "1D":
            if (pos[0] > self.extent[0]) and (pos[0] < self.extent[1]):
                return True
            else:
                return False

    def check_wall_collisions(self, proposed_step):
        """Given proposed step [current_pos, next_pos] it returns two lists
        1. a list of all the walls in the environment #shape=(N_walls,2,2)
        2. a boolean list of whether the step directly crosses (collides with) any of these walls  #shape=(N_walls,)
        Args:
            proposed_step (array): The proposed step. np.array( [ [x_current, y_current] , [x_next, y_next] ] )
        Returns:
            tuple: (1,2)
        """
        if self.dimensionality == "1D":
            # no walls in 1D to collide with
            return (None, None)
        elif self.dimensionality == "2D":
            if (self.walls is None) or (len(self.walls) == 0):
                # no walls to collide with
                return (None, None)
            elif self.walls is not None:
                walls = self.walls
                wall_collisions = utils.vector_intercepts(
                    walls, proposed_step, return_collisions=True
                ).reshape(-1)
                return (walls, wall_collisions)

    def vectors_from_walls(self, pos):
        """Given a position, pos, it returns a list of the vectors of shortest distance from all the walls to current_pos #shape=(N_walls,2)
        Args:
            proposed_step (array): The position np.array([x,y])
        Returns:
            vector array: np.array(shape=(N_walls,2))
        """
        walls_to_pos_vectors = utils.shortest_vectors_from_points_to_lines(pos, self.walls)[0]
        return walls_to_pos_vectors

    def apply_boundary_conditions(self, pos):
        """Performs a boundary condition check. If pos is OUTside the environment and the boundary conditions are solid then a different position, safely located 1cm within the environmnt, is returne3d. If pos is OUTside the environment but boundary conditions are periodic its position is looped to the other side of the environment appropriately.
        Args:
            pos (np.array): 1 or 2 dimensional position
        returns new_pos
        """
        if self.check_if_position_is_in_environment(pos) is False:
            if self.dimensionality == "1D":
                if self.boundary_conditions == "periodic":
                    pos = pos % self.extent[1]
                if self.boundary_conditions == "solid":
                    pos = min(max(pos, self.extent[0] + 0.01), self.extent[1] - 0.01)
                    pos = np.reshape(pos, (-1))
            if self.dimensionality == "2D":
                if self.boundary_conditions == "periodic":
                    pos[0] = pos[0] % self.extent[1]
                    pos[1] = pos[1] % self.extent[3]
                if self.boundary_conditions == "solid":
                    # in theory this wont be used as wall bouncing catches it earlier on
                    pos[0] = min(
                        max(pos[0], self.extent[0] + 0.01), self.extent[1] - 0.01
                    )
                    pos[1] = min(
                        max(pos[1], self.extent[2] + 0.01), self.extent[3] - 0.01
                    )
        return pos
