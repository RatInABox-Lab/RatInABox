import scipy
from scipy import signal
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import pandas as pd
from tqdm.notebook import tqdm
import time

import tomplotlib.tomplotlib as tpl

tpl.figureDirectory = "./Figures/"
tpl.setColorscheme(colorscheme=2)


class Environment:
    """Environment class: defines the environment and geometry of the world the agent lives in. 
    This class needs no other classes to initialise it. It exists independently of 
    Key functions: 
        • get_distance: geodesic distance between two points (accounting for walls and boundary conditions)
        • check_wall_intercepts: when the agent takes a "step" this function checks the agent deosn;t collide with a wall, if it does it adjusts the step accordingly 
    """

    def __init__(self, params={}):
        """Initialise environment, takes as input a parameter dictionary who's values supercede a default dictionary.

        Args:
            params (dict, optional). Defaults to {}.
        """
        default_params = {
            "maze_type": "one_room",
            "scale": 1,
            "aspect": 2,
            "dx": 0.02,  # superficially discretises the environment for plotting purposes
        }
        update_params(self, default_params)
        update_params(self, params)
        # defines walls of an environment
        self.walls = {}
        if self.maze_type == "one_room":
            self.dimensionality = "2D"
            self.walls["room1"] = np.array(
                [
                    [[0, 0], [0, self.scale]],
                    [[0, self.scale], [self.scale, self.scale]],
                    [[self.scale, self.scale], [self.scale, 0]],
                    [[self.scale, 0], [0, 0]],
                ]
            )
            self.extent = np.array([0, self.scale, 0, self.scale])
            self.centre = np.array([self.scale / 2, self.scale / 2])
        if self.maze_type == "one_room_one_wall":
            self.dimensionality = "2D"
            wall_pos = 0.75
            self.walls["room1"] = np.array(
                [
                    [[0, 0], [0, self.scale]],
                    [[0, self.scale], [self.scale, self.scale]],
                    [[self.scale, self.scale], [self.scale, 0]],
                    [[self.scale, 0], [0, 0]],
                ]
            )
            self.walls["wall"] = np.array(
                [
                    [
                        [wall_pos * self.scale, 0],
                        [wall_pos * self.scale, wall_pos * self.scale],
                    ]
                ]
            )

            self.walls["wall_dontplot"] = np.array(
                [
                    [
                        [wall_pos * self.scale, -3 * self.scale],
                        [wall_pos * self.scale, wall_pos * self.scale],
                    ]
                ]
            )
            self.centre = np.array([self.scale / 2, self.scale / 2])
            self.extent = np.array([0, self.scale, 0, self.scale])
        if self.maze_type == "one_room_rectangle":
            self.dimensionality = "2D"
            self.walls["room1"] = np.array(
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
            self.extent = np.array([0, self.aspect * self.scale, 0, self.scale])
            self.centre = np.array([self.aspect * self.scale / 2, self.scale / 2])

        if self.maze_type == "one_room_periodic":
            self.dimensionality = "2D"
            self.extent = np.array([0, self.scale, 0, self.scale])
            self.centre = np.array([self.scale / 2, self.scale / 2])

        if self.maze_type == "loop":
            self.dimensionality = "1D"
            self.radius = self.scale / (2 * np.pi)
            self.extent = np.array([0, self.scale])

        # save some prediscretised coords
        self.discrete_coords = self.discretise_space(dx=self.dx)
        self.flattened_discrete_coords = self.discrete_coords.reshape(
            -1, self.discrete_coords.shape[-1]
        )

    def discretise_space(self, dx=0.02):
        """Discretises the environment 
        Args:
            dx (float): discretisation distance
        Returns:
            array: and Ny x Mx x 2 array of position coordinates or Nx x 1 for 1D

        """  # returns a 2D array of locations discretised by dx
        if self.dimensionality == "2D":
            [minx, maxx, miny, maxy] = list(self.extent)
            self.x_array = np.arange(minx + dx / 2, maxx, dx)
            self.y_array = np.arange(miny + dx / 2, maxy, dx)[::-1]
            x_mesh, y_mesh = np.meshgrid(self.x_array, self.y_array)
            coordinate_mesh = np.array([x_mesh, y_mesh])
            discrete_coords = np.swapaxes(np.swapaxes(coordinate_mesh, 0, 1), 1, 2)
            return discrete_coords

        elif self.dimensionality == "1D":
            if self.maze_type == "loop":
                dtheta = 2 * np.pi / (int((2 * np.pi * self.radius) / dx))
                self.theta_array = np.arange(0 + dtheta / 2, 2 * np.pi, dtheta)
                discrete_coords = self.radius * self.theta_array.reshape(-1, 1)
                return discrete_coords

    def get_distance(self, pos1, pos2, return_vector_aswell=False):
        """Takes two position arrays and returns the array of pair-wise distances between points in
        Eventually this will take into account walls, locomotion costs etc. 
        Args:
            pos1 (array): N x dimensionality array of poisitions
            pos2 (array): M x dimensionality array of positions
        Returns:
            N x M array of pairwise distances
        """
        pos1_ = pos1.reshape(-1, 1, pos1.shape[-1])
        pos2_ = pos2.reshape(1, -1, pos2.shape[-1])
        pos1 = np.repeat(pos1_, pos2_.shape[1], axis=1)
        pos2 = np.repeat(pos2_, pos1_.shape[0], axis=0)
        lines = np.stack((pos1, pos2), axis=-2)
        # print(lines)
        vectors = lines[..., 0, :] - lines[..., 1, :]
        if self.dimensionality == "2D":
            if self.maze_type == "one_room_one_wall":
                wall = self.walls["wall_dontplot"][0]
                wall_ = wall
                distances = np.linalg.norm(vectors, axis=-1).reshape(-1)
                wall = wall_.reshape((1, 1,) + wall.shape)
                wall = np.repeat(wall, lines.shape[0], axis=0)
                wall = np.repeat(wall, lines.shape[1], axis=1)
                line_and_walls = np.stack((lines, wall), axis=-3)
                line_and_walls = line_and_walls.reshape(
                    (-1,) + line_and_walls.shape[-3:]
                )
                line_and_walls = line_and_walls + np.random.normal(
                    scale=1e-4, size=line_and_walls.shape
                )
                intersect_ = intersect(line_and_walls)
                to_top_of_wall = np.linalg.norm(pos1_ - wall_[0], axis=-1).reshape(
                    -1, 1
                )
                from_top_of_wall = np.linalg.norm(wall_[0] - pos2_, axis=-1).reshape(
                    1, -1
                )
                to_bottom_of_wall = np.linalg.norm(pos1_ - wall_[1], axis=-1).reshape(
                    -1, 1
                )
                from_bottom_of_wall = np.linalg.norm(wall_[1] - pos2_, axis=-1).reshape(
                    1, -1
                )
                top_route = (to_top_of_wall + from_top_of_wall).reshape(-1)
                bottom_route = (to_bottom_of_wall + from_bottom_of_wall).reshape(-1)
                new_quickest_route = np.minimum(top_route, bottom_route)
                distances[intersect_] = new_quickest_route[intersect_]
                distances = distances.reshape(pos1_.shape[0], pos2_.shape[1])
            elif self.maze_type == "one_room_periodic":
                flip = np.abs(vectors) > (self.scale / 2)
                vectors[flip] = -np.sign(vectors[flip]) * (
                    self.scale - np.abs(vectors[flip])
                )
                distances = np.linalg.norm(vectors, axis=-1)
            else:
                distances = np.linalg.norm(vectors, axis=-1)  # shape = N x M

        if self.dimensionality == "1D":
            if self.maze_type == "loop":
                distances = np.linalg.norm(vectors, axis=-1)  # shape = N x M
                overhalfways = vectors[distances > np.pi * self.radius]
                vectors[distances > np.pi * self.radius] = -np.sign(overhalfways) * (
                    2 * np.pi * self.radius - np.abs(overhalfways)
                )
                distances = np.linalg.norm(vectors, axis=-1)

        if return_vector_aswell == True:
            return distances, vectors
        else:
            return distances

    def sample_pos(self):
        """Returns a random position in the environment
        """
        ex = self.extent
        if self.dimensionality == "2D":
            position = np.array(
                [np.random.uniform(ex[0], ex[1]), np.random.uniform(ex[2], ex[3])]
            )
        if self.dimensionality == "1D":
            position = np.array([np.random.uniform(0, 2 * np.pi)])
        return position

    def scatter_feature_centres(self, n=10, jitter=False, random=False):
        """Evenly (if jitter=False) scatters n locations across the environment which can act as, for example, the centres of gaussian place fields. 
        Args:
            n (int): noumber of features 
            jitter (bool, optional): jitter the feature locations a bit. Defaults to False.

        Returns:
            _type_: _description_
        """
        if self.dimensionality == "2D":
            if random == True:
                centres = np.random.uniform(size=(n, 2))
                centres[:, 0] *= self.extent[1] - self.extent[0]
                centres[:, 1] *= self.extent[3] - self.extent[2]
            else:
                ex = self.extent
                area = (ex[1] - ex[0]) * (ex[3] - ex[2])
                delta = np.sqrt(area / n)
                x = np.arange(ex[0] + delta / 2, ex[1] - delta / 2 + 1e-6, delta)
                y = np.arange(ex[2] + delta / 2, ex[3] - delta / 2 + 1e-6, delta)
                centres = np.array(np.meshgrid(x, y)).reshape(2, -1).T
                if jitter == True:
                    centres += np.random.uniform(
                        -0.45 * delta, 0.45 * delta, centres.shape
                    )
            return centres

        elif self.dimensionality == "1D":
            if self.maze_type == "loop":
                dtheta = 2 * np.pi / n
                centres = self.radius * np.arange(
                    0 + dtheta / 2, 2 * np.pi, dtheta
                ).reshape(-1, 1)
                if jitter == True:
                    centres += np.random.uniform(
                        -0.45 * dtheta, 0.45 * dtheta, centres.shape
                    )
                return centres

    def check_wall_intercepts(self, proposed_step):
        """Given the curent proposed step [currentPos, nextPos] it calculates the perpendicular distance [#1,float] to the nearest wall [#2,array] (from next pos). It also checks if there is an immediate collison [#3,bool] with any wall [#4,array] on the current step (though this should be uneccesary if the handles te warning appropriately)
        Args:
            proposed_step (array): The proposed step. np.array( [ [x_current, y_current] , [x_next, y_next] ] )

        Returns:
            tuple: (#1,#2,#3,#4)
        """
        s1, s2 = np.array(proposed_step[0]), np.array(proposed_step[1])
        pos = s1
        ds = s2 - s1
        stepLength = np.linalg.norm(ds)
        ds_perp = get_perpendicular(ds)
        collisionList = [[], []]
        futureCollisionList = [[], []]

        closest_wall_distance = None
        closest_wall = None
        collision_now = False
        collision_now_wall = None

        # check if the current step results in a collision
        walls = self.walls  # current wall state
        for room in walls.keys():
            for wall in walls[room]:
                w1, w2 = np.array(wall[0]), np.array(wall[1])
                dw = w2 - w1
                dw_perp = get_perpendicular(dw)
                # calculates point of intercept between the line passing along the current step direction and the lines passing along the walls,
                # if this intercept lies on the current step and on the current wall (0 < lam_s < 1, 0 < lam_w < 1) this implies a "collision"
                # if it lies ahead of the current step and on the current wall (lam_s > 1, 0 < lam_w < 1) then we should "veer" away from this wall
                # this occurs iff the solution to s1 + lam_s*(s2-s1) = w1 + lam_w*(w2 - w1) satisfies 0 <= lam_s & lam_w <= 1
                with np.errstate(divide="ignore"):
                    lam_s = (np.dot(w1, dw_perp) - np.dot(s1, dw_perp)) / (
                        np.dot(ds, dw_perp)
                    )
                    lam_w = (np.dot(s1, ds_perp) - np.dot(w1, ds_perp)) / (
                        np.dot(dw, ds_perp)
                    )
                # there are two situations we need to worry about:
                # • 0 < lam_s < 1 and 0 < lam_w < 1: the collision is ON the current proposed step . Do something immediately.
                # • lam_s > 1     and 0 < lam_w < 1: the collision is on the current trajectory, some time in the future. Maybe do something.
                if (0 <= lam_s <= 1) and (0 <= lam_w <= 1):
                    collision_now = True
                    collision_now_wall = wall

                # calculate the shortest distance from the current point (s1) to the wall segment.
                # This is done by solving w1+lam.(w2-w1)+gam(dw_perp)=s1.
                # If 0<lam<1 then the shortest distance is gam.||dw_perp||
                # Else, distance is min(||s1-w1||,||s1-w2||)
                lam = np.dot((s2 - w1), dw) / np.dot(dw, dw)
                gam = np.dot((s2 - w1), dw_perp) / np.dot(dw_perp, dw_perp)
                if 0 < lam < 1:
                    D = np.abs(gam * np.dot(dw_perp, dw_perp))
                else:
                    D = min(np.linalg.norm(s2 - w1), np.linalg.norm(s2 - w2))
                if (closest_wall_distance is None) or (D < closest_wall_distance):
                    closest_wall_distance = D
                    closest_wall = wall
        return (closest_wall_distance, closest_wall, collision_now, collision_now_wall)


class Agent:
    """Agent defines an agent moving around the environment. Specifically this class handles the movement policy, and communicates with the environment class to ensure the agent obeys boundaries and walls etc. Initialises with param dictionary which must contain the Environment (class) in which the agent exists. Only has one function update(dt) which moves the agent along in time by dt.
    """

    def __init__(self, params={}, load_from_file=None):
        """Sets the parameters of the maze anad agent (using default if not provided) 
        and initialises everything. This includes: 
        Args:
            params (dict, optional): A dictionary of parameters which you want to differ from the default. Defaults to {}.
        """
        default_params = {
            "Environment": None,
            "policy": "raudies",
            "velocity_decay_time": 3.0,
            "drift_velocity": 0.0,
            "velocity_noise_scale": 0.1,
            "wall_follow_distance": 0.02,
        }
        if load_from_file is not None:
            self.loadFromFile(name=load_from_file)
        else:
            update_params(self, default_params)
            update_params(self, params)
            # initialise history dataframes
            self.history = {}
            self.history["t"] = []
            self.history["pos"] = []

            # time and runID
            self.t = 0

            # set pos/vel
            if self.Environment.dimensionality == "2D":
                ex = self.Environment.extent
                self.pos = np.array(
                    [np.random.uniform(ex[0], ex[1]), np.random.uniform(ex[2], ex[3])]
                )
                direction = np.random.uniform(0, 2 * np.pi)
                self.velocity = self.velocity_noise_scale * np.array(
                    [np.cos(direction), np.sin(direction)]
                )
            if self.Environment.dimensionality == "1D":
                if self.Environment.maze_type == "loop":
                    # self.pos = np.array([np.random.uniform(0,2*np.pi)])*self.Environment.radius
                    self.pos = np.array([0])
                    self.velocity = np.array([self.velocity_noise_scale])
            # handle None params
        return

    def update_state(self, dt):
        """Movement policy update. 
            In principle this does a very simple thing: 
            • updates time by dt
            • updates position along the velocity direction 
            • updates velocity (speed and direction) according to a movement policy
            In reality it's a complex function as the policy requires checking for immediate or upcoming collisions with all walls at each step.
            This is done by function self.check_wall_intercepts()
            What it does with this info (bounce off wall, turn to follow wall, etc.) depends on policy. 
        """
        self.dt = dt
        self.t += dt
        extent = self.Environment.extent

        if self.Environment.dimensionality == "2D":
            proposed_new_pos = self.pos + self.velocity * dt
            proposed_step = np.array([self.pos, proposed_new_pos])
            # if np.linalg.norm(self.last_safe_position-self.pos) >= self.next_check_distance:
            # it's been long since you knew you were far all walls. Check again
            drift_velocity = 0
            if len(self.Environment.walls) > 0:  # check you don't collide with a wall
                (collision_now, i) = (True, 1)
                while (
                    collision_now == True and i <= 2
                ):  # 2 should catch if it collides with a corner
                    (
                        _,
                        _,
                        collision_now,
                        collision_now_wall,
                    ) = self.Environment.check_wall_intercepts(proposed_step)
                    if collision_now == True:
                        self.velocity = wall_bounce_or_follow(
                            self.velocity, collision_now_wall, "bounce"
                        )
                        proposed_new_pos = self.pos + self.velocity * dt
                        proposed_step = np.array([self.pos, proposed_new_pos])
                    i += 1

            if self.Environment.maze_type[-8:] != "periodic":
                # catch instances when agent leaves area (put back inside)
                flip_velocity = False
                if self.pos[0] < extent[0]:
                    self.pos[0] = extent[0] + 0.02
                    flip_velocity = True
                if self.pos[0] > extent[1]:
                    self.pos[0] = extent[1] - 0.02
                    flip_velocity = True
                if self.pos[1] < extent[2]:
                    self.pos[1] = extent[2] + 0.02
                    flip_velocity = True
                if self.pos[1] > extent[3]:
                    self.pos[1] = extent[3] - 0.02
                    flip_velocity = True
                if flip_velocity == True:
                    self.velocity = -self.velocity

            elif self.Environment.maze_type[-8:] == "periodic":
                # catch instances where agent leaves area, bring to other side
                if self.pos[0] < extent[0]:
                    self.pos[0] += extent[1] - extent[0]
                if self.pos[0] > extent[1]:
                    self.pos[0] -= extent[1] - extent[0]
                if self.pos[1] < extent[2]:
                    self.pos[1] += extent[3] - extent[2]
                if self.pos[1] > extent[3]:
                    self.pos[1] -= extent[3] - extent[2]

            # UPDATE POSITION
            self.pos += self.velocity * dt

            # UPDATE VELOCITY
            drift_velocity = self.drift_velocity
            velocity_noise_scale = self.velocity_noise_scale
            velocity_decay_time = self.velocity_decay_time

            self.velocity += ornstein_uhlenbeck(
                dt=dt,
                v=self.velocity,
                drift=drift_velocity,
                noise_scale=velocity_noise_scale,
                decay_time=velocity_decay_time,
            )

        elif self.Environment.dimensionality == "1D":
            if self.Environment.maze_type == "loop":
                radius = self.Environment.radius
                self.pos = np.mod(self.pos + dt * self.velocity, 2 * np.pi * radius)
                self.velocity += ornstein_uhlenbeck(
                    dt=dt,
                    v=self.velocity,
                    drift=self.drift_velocity,
                    noise_scale=self.velocity_noise_scale,
                    decay_time=self.velocity_decay_time,
                )

        # write to history
        self.history["t"].append(self.t)
        self.history["pos"].append(list(self.pos))

        return self.pos


# NEURON CLASSES
"""The below classes can be combined to make complex multilayer heirarchies of multi compartment neurons. 
Crucially, each classes ALWAYS has two functions: 
• get_state(): returns the firing rate or membrane voltage of the neuron according to the input layers to this neuron. If pos variable is supplied, this recursibvely calls get_state() in all upstream layers until StateNeurons eventually supply the actual state at the desired position. 
• update_state(): basically calles get_state() but (1) does so with pos=None, this means it calculates the state from the lastrecorded state of the inout layers, rather that recursively calculating from the bottom up. (2) saves these in place.

Just, always, update_state() is what you want to use during a training loop where the neurons are "live" and reflect the current state of the system. get_state() can be used later during analysis to poll the receptive fields etc of the neurons."""


class StateNeurons:
    """State neurons are special in that they (may) explicitly have access to the state of the system (position, velocity etc.) and interact with the environment. They do not take their input from any other neurons, rather they are "fundamental", representing sensory observations. 
    The input params dictionary must contain the Agent class (who's position determines how these neurons fire and contains the Environment - who's geometry may be important for determining firing rate). The other key param is "type" which specifis type of state neuron (e.g. gaussian, gaussian_threshold, one_hot, velocity, head_direction). "color" is jsut used for later plotting. 
    """

    def __init__(self, params={}):
        default_params = {
            "Agent": None,
            "name": "Features",
            "type": "gaussian_threshold",
            "locations": None,
            "widths": 0.15,
            "color": None,
            "top_layer": False,
            "min_fr": 0,
            "max_fr": 1,
        }
        update_params(self, default_params)
        update_params(self, params)

        self.n = len(self.locations)
        if isinstance(self.widths, (float, int)):
            self.widths = self.widths * np.ones(self.n)
        self.Environment = self.Agent.Environment

    def get_state(self, pos=None, route=None):
        """
        Returns the firing rate of the neuron. 
        pos can be

            • np.array(): an array of locations 
            • 'from_agent': uses Agent.pos 
            • 'all' : array of locations over entire environment (for rate map plotting) 
        
        Here route can be ignored (no need to decided whether state neurons read from basal or apical dendrite, they always read out directly from environment. It will never be used but exists as this function can be called at the end of a long list of recursive get_state() calls, in other (downstream) neurons. 
        """

        if pos == None:  # get stored state (presumably recently updates)
            state = self.phi_U

        else:  # actually calculate state
            if pos == "from_agent":
                pos = self.Agent.pos
            elif pos == "all":
                pos = self.Environment.flattened_discrete_coords

            distance_to_centres = self.Environment.get_distance(self.locations, pos)
            widths = self.widths.reshape(-1, 1)

            if self.Environment.dimensionality == "2D":
                if self.type == "gaussian_threshold":
                    state = np.maximum(
                        np.exp(-(distance_to_centres ** 2) / (2 * (widths ** 2)))
                        - np.exp(-1 / 2),
                        0,
                    ) / (1 - np.exp(-1 / 2))
                if self.type == "gaussian":
                    state = np.exp(-(distance_to_centres ** 2) / (2 * (widths ** 2)))
                if self.type == "one_hot":
                    closest_centres = np.argmin(np.abs(distance_to_centres), axis=0)
                    state = np.eye(self.n)[closest_centres].T
                if self.type == "diff_of_gaussians":
                    ratio = 1.5
                    state = np.exp(
                        -(distance_to_centres ** 2) / (2 * (widths ** 2))
                    ) - (1 / ratio ** 2) * np.exp(
                        -(distance_to_centres ** 2) / (2 * ((ratio * widths) ** 2))
                    )
                    state *= ratio ** 2 / (ratio ** 2 - 1)
                if self.type == "top_hat":
                    closest_centres = np.argmin(np.abs(distance_to_centres), axis=0)
                    state = np.eye(self.n)[closest_centres].T

            if self.Environment.dimensionality == "1D":
                if self.type == "von_mises":
                    kappa = 1 / widths ** 2
                    state = (
                        np.e
                        ** (
                            kappa
                            * np.cos(distance_to_centres / self.Environment.radius)
                        )
                    ) / (np.e ** kappa)
                if self.type == "gaussian_threshold":
                    state = np.maximum(
                        np.exp(-(distance_to_centres ** 2) / (2 * (widths ** 2)))
                        - np.exp(-1 / 2),
                        0,
                    ) / (1 - np.exp(-1 / 2))
                if self.type == "gaussian":
                    state = np.exp(-(distance_to_centres ** 2) / (2 * (widths ** 2)))
                if self.type == "diff_of_gaussians":
                    ratio = 1.5
                    state = np.exp(
                        -(distance_to_centres ** 2) / (2 * (widths ** 2))
                    ) - (1 / ratio) * np.exp(
                        -(distance_to_centres ** 2) / (2 * ((ratio * widths) ** 2))
                    )
                    state *= ratio / (ratio - 1)
                if self.type == "one_hot":
                    closest_centres = np.argmin(np.abs(distance_to_centres), axis=0)
                    state = np.eye(self.n)[closest_centres].T

            state = state * (self.max_fr - self.min_fr) + self.min_fr

        return state

    def update_state(self):
        """Updates the state of these state neurons using the Agents current position. Stores the current firing rate in self.phi_U
        """
        state = self.get_state(pos="from_agent")
        self.phi_U = state.reshape(-1)


class Neurons:
    """PyramidalNeurons is, arguably, the most important class. It manages the sets of neurons i'm interested in studying and which exist in a multilayer network of other neurons and have input dendritic weights which can be trained. It works like this:
    All pyramidal neurons have somatic compartments. This defines the firing rate of the neurons, essentially their output. Dendritic compartments can be added using self.add_comartment(). Depending on the conditions (specifically the current theta phase) the soma's membrane voltage is determined from the membrane voltages of it's input dendritic compartments. The dendritc compartments (defined as a class below) themselves have input layers but will only comunicate with the soma - that is they don't spike and so cannot be read by other neurons. 

    """

    def __init__(self, params: dict):
        default_params = {
            "name": "Neurons",
            "n": 10,
            "Agent": None,
            "top_layer": False,
            "error_timescale": 5,
            "noise_level": 0.01,
            "noise_timescale": 100e-3,
            # actiavtion params
            "activation": "sigmoid",
            "max_fr": 1,
            "min_fr": -1,
            "mid_x": 0,
            "width_x": 2,
            # theta params
            "theta_model": "square",
            "theta_freq": 10,
            "theta_phase": 0,  # in radians
            "theta_bias": 0,  # -1=all basal/bottom-up, +1=all apical/top_down
        }
        update_params(self, default_params)
        update_params(self, params)

        self.phi_U = np.zeros(self.n)
        self.noise = np.zeros_like(self.phi_U)
        self.Compartments = {"basal": None, "apical": None}

        self.history = {"time": [], "loss": []}
        self.t = 0
        self.error = 0  # error trace

        self.activation_params = {
            "activation": self.activation,
            "max_fr": self.max_fr,
            "min_fr": self.min_fr,
            "mid_x": self.mid_x,
            "width_x": self.width_x,
        }

        self.theta_params = {
            "theta_model": self.theta_model,
            "theta_freq": self.theta_freq,
            "theta_phase": self.theta_phase,  # in radians
            "theta_bias": self.theta_bias,
            "theta_T": 1 / self.theta_freq,
        }

    def add_compartment(self, params={}):
        """Adds a dendritic compartment to the neuron. Params dictionary must contain a 'name' ('basal' or 'apical') and a 'layers' (a list of input layers , ie. neuron classes which feed into it) amongst other (see class listing for details)
        Args:
            params (dict)
        """
        params["n"] = self.n
        params["ParentLayer"] = self
        self.Compartments[params["name"]] = DendriticCompartment(params)

    def update_dendrite_states(self):
        """Updtes the firing rates of the dendritic compartments.
        """
        for compartment_name, compartment in self.Compartments.items():
            compartment.update_state()
        return

    def update_soma_states(self):
        """Updates firing rate of the somatic compartment.
        """
        self.get_state(pos=None, save_in_place=True, add_noise=True)

    def update_weights(self):
        """Dendritc prediction of somatic activity"""
        dt = self.Agent.dt
        for (compartment_name, compartment) in self.Compartments.items():
            if compartment.eta != 0:
                compartment.update_weights(target=self.phi_U, dt=dt)

        try:
            phi_Vb, phi_Va = (
                self.Compartments["basal"].phi_V,
                self.Compartments["apical"].phi_V,
            )
            error = np.mean(np.abs(phi_Vb - phi_Va))
            if len(self.history["loss"]) == 0:
                self.error = error
            self.error = (dt / self.error_timescale) * error + (
                1 - dt / self.error_timescale
            ) * self.error
            self.history["time"].append(self.t)
            self.history["loss"].append(self.error)
        except KeyError:
            pass

        return

    def get_state(self, pos=None, route="basal", save_in_place=False, add_noise=False):
        """
        Returns the firing rate of the neurons. 
        Either:
            • pos = None: simply gets the last records state of the neurons dendritic compartments and uses these to estimate the firing rate (for the current time/theta phase). 
        or 
            • pos = np.array(): an array of locations 
            • pos = 'from_agent': uses Agent.pos 
            • pos = 'all' : array of locations over entire environment (for rate map plotting) 
        in which case this function recursively calls get_state from it's input layer (which calls get_state ...) until this terminates at features. 
        
        route, in the recursive case, determines if the information route to the feature neurons initially goes through. 
            • 'basal':  basal dendrites i.e.   bottom-up inference. information flow example in third layer of four layer network  1 --> 2 --> 3
            • 'apical': apical dendrite i.e. top-down generation. information flow example in third layer of four layer network  1 --> 2 --> 3 --> 4 --> 4 --> 3
        returns the state/an array of states evaluated at the position/s 

        save_in_place whether to actually save in place (should only be done in training loop)

        """
        if pos == None:

            basal = self.Compartments["basal"]
            apical = self.Compartments["apical"]
            self.t = self.Agent.t
            theta = get_theta(self.t, self.theta_params)
            U = (
                theta * basal.V + (1 - theta) * apical.V
            )  # this needs to be more general, i.e. an input somehow specifies this function
            phi_U = activate(U, other_args=self.activation_params)

        else:
            compartment = self.Compartments[route]
            next_route = route
            if self.top_layer == True:
                next_route = "basal"
            U = compartment.get_state(pos=pos, route=next_route)
            phi_U = activate(U, other_args=self.activation_params)

        if add_noise == True:
            dt = self.Agent.dt
            self.noise = self.noise + ornstein_uhlenbeck(
                dt,
                self.noise,
                drift=0,
                noise_scale=self.noise_level,
                decay_time=self.noise_timescale,
            )
            noise = self.noise
            phi_U += noise

        if save_in_place == True:
            self.U = U
            self.phi_U = phi_U

        return phi_U

    def plot_activation(self):
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        x = np.arange(-5, 5, 0.1)
        y = activate(x, other_args=self.activation_params)
        y_prime = activate(x, other_args=self.activation_params, deriv=True)
        ax.plot(x, y, label="f(x)")
        ax.plot(x, y_prime, label="df/dx")
        tpl.xyAxes(ax)
        ax.set_xlabel("x")
        ax.legend(loc="lower right")
        ax.set_xticks([self.mid_x])
        ax.set_yticks([self.max_fr, self.min_fr])
        return fig, ax





class Visualiser:
    def __init__(
        self, Environment=None, Agent=None, Neurons=None,
    ):
        self.Agent = Agent
        self.Environment = Environment
        self.Neurons = Neurons  # list
        layer_dict = {}
        for layer in self.Neurons:
            layer_dict[layer.name] = layer
        self.Neurons = layer_dict

    def plot_maze_structure(self, fig=None, ax=None, save=False):
        extent, walls = self.Environment.extent, self.Environment.walls
        if fig == None and ax == None:
            fig, ax = plt.subplots(
                figsize=((extent[1] - extent[0]), (extent[3] - extent[2]))
            )
        for wall_object in walls.keys():
            for wall in walls[wall_object]:
                if wall_object[-8:] != "dontplot":
                    ax.plot(
                        [wall[0][0], wall[1][0]],
                        [wall[0][1], wall[1][1]],
                        color="darkgrey",
                        linewidth=4,
                    )
        ax.set_aspect("equal")
        ax.grid(False)
        ax.axis("off")
        if save == True:
            tpl.saveFigure(fig, "mazeStructure")
        return fig, ax

    def plot_feature_locations(self, layer_name="Features"):
        fig, ax = self.plot_maze_structure()
        feature_locations = self.Neurons[layer_name].locations
        ax.scatter(
            feature_locations[:, 0], feature_locations[:, 1], c="C1", marker="x", s=15
        )
        return fig, ax

    def plot_trajectory(self, timerange="all", fig=None, ax=None):
        t = np.array(self.Agent.history["t"])
        dt = self.Agent.dt
        t, pos = np.array(self.Agent.history["t"]), np.array(self.Agent.history["pos"])
        if timerange == "all":
            startid, endid = 0, -1
        if type(timerange) in (float, int):
            if timerange < 0:
                startid = np.argmin(np.abs(t - (t[-1] + timerange * 60)))
                endid = -1
            if timerange >= 0:
                startid = 0
                endid = np.argmin(np.abs(t - timerange * 60))
        if type(timerange) is list:
            startid = np.argmin(np.abs(t - timerange[0] * 60))
            endid = np.argmin(np.abs(t - timerange[1] * 60))
        if self.Environment.dimensionality == "2D":
            print(dt)
            skiprate = max(1, int(0.01 / (0.1 * dt)))
            trajectory = pos[startid:endid, :][::skiprate]
            if fig is None and ax is None:
                fig, ax = self.plot_maze_structure()
            ax.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                s=10,
                alpha=0.7,
                zorder=2,
                c="C0",
                linewidth=0,
            )
        if self.Environment.dimensionality == "1D":
            skiprate = max(1, int(0.1 / dt))
            if fig is None and ax is None:
                fig, ax = plt.subplots(figsize=(4, 2))
            ax.scatter(t[startid:endid][::skiprate], pos[startid:endid][::skiprate])
            tpl.xyAxes(ax)
            ax.spines["left"].set_position(("data", t[startid]))
        tpl.saveFigure(fig, "trajectory")
        return fig, ax

    def plot_heatmap(self, granularity=0.05):
        pos = np.array(self.Agent.history["pos"])
        ex = self.Environment.extent
        pos = np.vstack(
            (
                pos,
                np.array(
                    [[ex[0], ex[2]], [ex[1], ex[2]], [ex[0], ex[3]], [ex[1], ex[3]]]
                ),
            )
        )
        bins = [int((ex[1] - ex[0]) / granularity), int((ex[3] - ex[2]) / granularity)]
        heatmap, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig, ax = self.plot_maze_structure()
        ax.imshow(heatmap.T[::-1, :], extent=extent)
        return fig, ax

    def plot_rate_map(
        self, layer_name="Features", route="basal", neuron_id="all", cbar=False
    ):
        layer = self.Neurons[layer_name]
        color = layer.color
        rate_maps = layer.get_state(pos="all", route=route)

        if self.Environment.dimensionality == "2D":
            if neuron_id is None or neuron_id == "all":
                neuron_id = [np.random.randint(0, rate_maps.shape[0])]
            if type(neuron_id) in (list, np.ndarray):
                neuron_id = np.array(neuron_id)
                neuron_id = [
                    np.argmin(np.linalg.norm(layer.locations - neuron_id, axis=1))
                ]
            if type(neuron_id) is str:
                if neuron_id.isdigit():
                    neuron_id = np.linspace(
                        0, rate_maps.shape[0] - 1e-6, int(neuron_id)
                    ).astype(int)
                    pass
            else:
                neuron_id = [neuron_id]

            rate_maps = rate_maps[neuron_id, :]

            rows = 1
            fig, ax = plt.subplots(
                rows,
                len(neuron_id),
                figsize=(0.5 * len(neuron_id), 0.5 * 1),
                facecolor=layer.color,
            )
            if not hasattr(ax, "__len__"):
                ax = [ax]
            for (i, ax_) in enumerate(ax):
                self.plot_maze_structure(fig, ax_)

                rate_map = rate_maps[i].reshape(
                    self.Environment.discrete_coords.shape[:2]
                )

                rate_map[0, 0] = 0
                rate_map[-1, -1] = 1
                # vmin = layer.min_fr
                # vmax = layer.max_fr
                extent = self.Environment.extent
                im = ax_.imshow(rate_map, extent=extent)
                # im = ax_.imshow(rate_map, extent=extent, vmin=vmin, vmax=vmax)
                if (i == len(ax) - 1) and (cbar is True):
                    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax_)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cb = fig.colorbar(im, cax=cax, ticks=[0, np.max(rate_map)])
                    cb.outline.set_visible(False)
            return fig, ax

        if self.Environment.dimensionality == "1D":
            if type(neuron_id) is str:
                if neuron_id == "random":
                    neuron_id = np.random.randint(0, rate_maps.shape[0])
                    rate_maps = rate_maps[neuron_id, :].reshape(1, -1)
                if neuron_id == "all":
                    pass
                if neuron_id.isdigit():
                    neuron_ids = np.linspace(
                        0, rate_maps.shape[0] - 1e-6, int(neuron_id)
                    ).astype(int)
                    rate_maps = rate_maps[list(neuron_ids), :]
            if type(neuron_id) is int:
                rate_maps = rate_maps[neuron_id, :].reshape(1, -1)
            fig, ax = self.mountain_plot(
                self.Environment.flattened_discrete_coords[:, 0], rate_maps, color=color
            )

        return fig, ax

    def plot_training_curves(self, layer_names=["Layer1", "Layer2"], crop_last=0):
        fig, ax = plt.subplots(figsize=(4, 2))
        for layer_name in layer_names:
            layer = self.Neurons[layer_name]
            t = np.array(layer.history["time"]) / 60
            endid = np.argmin(np.abs(t - (t[-1] - 60 * crop_last)))
            W_hist = np.array(layer.history["W"])
            W_last = W_hist[-1, :, :]
            W_loss = np.linalg.norm(W_hist - W_last, axis=(1, 2))
            ax.plot(t[:endid], W_loss[:endid] / max(W_loss))
        tpl.xyAxes(ax)
        ax.set_title("Loss")
        return

    def mountain_plot(
        self, X, NbyX, gap_mm=3.2, overlap=0.75, color="C0", yaxis=False, scale=False
    ):
        shiftby = np.max(np.abs(NbyX)) / overlap
        shiftby = 1 / overlap
        NbyX = NbyX / shiftby
        a = 1
        if NbyX.shape[0] == 1:
            a = 6
        fig, ax = plt.subplots(figsize=(2, a * len(NbyX) * gap_mm / 18))
        for i in range(len(NbyX)):
            ax.plot(X, NbyX[i] + i + 1, c=color, linestyle="solid", alpha=1, zorder=100)
            tpl.xyAxes(ax)
            ax.fill_between(X, NbyX[i] + i + 1, i + 1, facecolor=color, alpha=0.3)
        ax.spines["left"].set_bounds(1, len(NbyX))
        ax.spines["bottom"].set_position(("outward", 1))
        ax.spines["left"].set_position(("outward", 1))
        ax.set_yticks([1, len(NbyX)])
        ax.set_ylim(
            1 - 0.5,
            0.5 + np.max(NbyX + np.arange(1, len(NbyX) + 1).reshape(len(NbyX), -1)),
        )
        ax.set_xticks(np.arange(max(X + 0.1)))
        if yaxis == False:
            ax.spines["left"].set_color(None)
            ax.set_yticks([])
        if scale == True:
            ax.plot(
                [0.8 * np.max(X), 0.8 * np.max(X)],
                [0.8 * len(NbyX), 0.8 * len(NbyX) + overlap],
                alpha=0.5,
                color=color,
                zorder=111,
            )
            ax.fill_between(
                [0.78 * np.max(X), 0.92 * np.max(X)],
                [0.82 * len(NbyX) + overlap, 0.82 * len(NbyX) + overlap],
                [0.78 * len(NbyX), 0.78 * len(NbyX)],
                facecolor="w",
                zorder=110,
                alpha=0.8,
            )
            ax.text(
                0.82 * np.max(X),
                0.8 * len(NbyX) + 0.25 * overlap,
                f"$=$ {overlap:.2f}",
                {"fontsize": 4},
                zorder=111,
            )
        return fig, ax

    def plot_loss(self, layers=["Layer1"]):
        fig, ax = plt.subplots(figsize=(2, 2))
        maximum = 0
        for layer_name in layers:
            layer = self.Neurons[layer_name]
            t = np.array(layer.history["time"]) / 60
            loss = layer.history["loss"]
            start_loss = loss[0]
            maximum = max(maximum, np.max(loss))
            ax.plot(t, loss, color=layer.color, label=layer_name)
            ax.set_ylim(bottom=0, top=maximum)
            ax.legend()
            tpl.xyAxes(ax)
            ax.set_xlabel("Training time / min")
            ax.set_ylabel("Loss")
        return fig, ax


"""
OTHER USEFUL FUNCTIONS
"""


def get_theta(time, other_args={}):
    """Returns the theta value \in [0,1]

    Args:
        time (float): current time
        other_args : dict of additional parameters (theta_freq, theta_phase etc...). Must contain 
    """
    try:
        theta_model = other_args["theta_model"]
    except KeyError:
        theta_model = "square"

    if theta_model == "square":
        try:
            theta_freq = other_args["theta_freq"]
        except KeyError:
            theta_freq = 10
        try:
            theta_phase = other_args["theta_phase"]
        except KeyError:
            theta_phase = 0
        try:
            theta_bias = other_args["theta_bias"]
        except KeyError:
            theta_bias = 0
        T_theta = 1 / theta_freq
        theta = 1 * (
            (((time - theta_phase * T_theta / (2 * np.pi)) % T_theta) / T_theta)
            < (0.5 * (theta_bias + 1))
        )

    return theta


def activate(x, activation="sigmoid", deriv=False, other_args={}):
    """Activation function function

    Args:
        x (the input (vector))
        activation: which type of fucntion to use (this is overwritten by 'activation' key in other_args)
        deriv (bool, optional): Whether it return f(x) or df(x)/dx. Defaults to False.
        other_args: Dictionary of parameters including other_args["activation"] = str for what type of activation (sigmoid, linear) and other params e.g. sigmoid midpoi n, max firing rate... 

    Returns:
        f(x) or df(x)/dx : array same as x
    """
    try:
        name = other_args["activation"]
    except KeyError:
        name = activation
    if name == "linear":
        if deriv == False:
            return x
        elif deriv == True:
            return np.ones(x.shape)

    if name == "sigmoid":
        # default sigmoid parameters set so that
        # max_f = max firing rate = 1
        # mid_x = middle point on domain = 1
        # width_x = distance from 5percent_x to 95percent_x = 1
        try:
            max_fr = other_args["max_fr"]
        except KeyError:
            max_fr = 1
        try:
            min_fr = other_args["min_fr"]
        except KeyError:
            min_fr = 0
        try:
            mid_x = other_args["mid_x"]  # neuronal input for half-max output
        except KeyError:
            mid_x = 1
        try:
            width_x = other_args["width_x"]
        except KeyError:
            width_x = 1
        beta = np.log((1 - 0.05) / 0.05) / (0.5 * width_x)  # sigmoid width
        if deriv == False:
            exp = np.exp(-beta * (x - mid_x))
            return ((max_fr - min_fr) / (1 + np.exp(-beta * (x - mid_x)))) + min_fr
        elif deriv == True:
            f = activate(x, deriv=False, other_args=other_args)
            return beta * (f - min_fr) * (1 - (f - min_fr) / (max_fr - min_fr))

    # if name == "threshold_linear":
    #     try:
    #         max_fr = other_args["max_fr"]
    #     except KeyError:
    #         max_fr = 1
    #     try:
    #         mid_x = other_args["mid_x"]  # neuronal input for half-max output
    #     except KeyError:
    #         mid_x = 1
    #     try:
    #         width_x = other_args["width_x"]
    #     except KeyError:
    #         width_x = 1
    #     lin_con = (x < (mid_x + width_x / 2)) * (
    #         x > (mid_x - width_x / 2)
    #     )  # condition for middle linear component
    #     if deriv == False:
    #         f = np.zeros_like(x)
    #         f[x < (mid_x - width_x / 2)] = 0
    #         f[x > (mid_x + width_x / 2)] = max_fr
    #         f[lin_con] = (max_fr / (width_x)) * (x[lin_con] - (mid_x - width_x / 2))
    #         return f
    #     elif deriv == True:
    #         f = np.zeros_like(x)
    #         f[lin_con] = max_fr / (width_x)
    #         return f

    # if name == "tanh":
    #     if deriv == False:
    #         return np.tanh(x)
    #     elif deriv == True:
    #         return 1 - np.tanh(x) ** 2


def ornstein_uhlenbeck(dt, v, drift=0.0, noise_scale=0.16, decay_time=5.0):
    """An ornstein uhlenbeck process in v.
    v can be one or two dimensional 

    Args:
        dt
        v (): [description]
        drift_velocity (float, or same type as v, optional): [description]. Defaults to 0.
        velocity_noise_scale (float, or same type as v, optional): Magnitude of velocity deviations from drift velocity. Defaults to 0.16 (16 cm s^-1).
        velocity_decay_time (float, optional): Effectively over what time scale you expect veloctiy to change directions. Defaults to 5.

    Returns:
        dv (same type as v); the required update ot the velocity
    """
    v = np.array(v)
    drift = drift * np.ones_like(v)
    noise_scale = noise_scale * np.ones_like(v)
    decay_time = decay_time * np.ones_like(v)
    sigma = np.sqrt((2 * noise_scale ** 2) / (decay_time * dt))
    theta = 1 / decay_time
    dv = theta * (drift - v) * dt + sigma * np.random.normal(size=v.shape, scale=dt)
    return dv

    # if type(v) is not np.ndarray:

    #     sigma = np.sqrt((2 * velocity_noise_scale ** 2) / (velocity_decay_time * dt))
    #     theta = 1 / velocity_decay_time
    #     dv = theta * (drift_velocity - v) * dt + sigma * np.random.normal(scale=dt)
    #     return dv

    # if type(v) is np.ndarray:
    #     if type(drift_velocity) in (float, int):
    #         drift_velocity = drift_velocity * np.ones(v.shape)
    #     if type(velocity_noise_scale) in (float, int):
    #         velocity_noise_scale = velocity_noise_scale * np.ones(v.shape)
    #     if type(velocity_decay_time) in (float, int):
    #         velocity_decay_time = velocity_decay_time * np.ones(v.shape)
    #     sigma = np.diag(
    #         np.sqrt((2 * velocity_noise_scale ** 2) / (velocity_decay_time * dt))
    #     )
    #     theta = np.diag(1 / velocity_decay_time)
    #     dv = np.matmul(theta, (drift_velocity - v)) * dt + np.matmul(
    #         sigma, np.random.normal(size=v.shape, scale=dt)
    #     )

    #     return dv


def get_angle(segment):
    """Given a 'segment' (either 2x2 start and end positions or 2x1 direction bearing) 
         returns the 'angle' of this segment modulo 2pi
    Args:
        segment (array): The segment, (2,2) or (2,) array 
    Returns:
        float: angle of segment
    """
    eps = 1e-6
    if segment.shape == (2,):
        return np.mod(np.arctan2(segment[1], (segment[0] + eps)), 2 * np.pi)
    elif segment.shape == (2, 2):
        return np.mod(
            np.arctan2(
                (segment[1][1] - segment[0][1]), (segment[1][0] - segment[0][0] + eps)
            ),
            2 * np.pi,
        )


def turn(current_direction, turn_angle):
    """Turns the current direction by an amount turn_angle, modulus 2pi
    Args:
        current_direction (array): current direction 2-vector
        turn_angle (float): angle ot turn in radians
    Returns:
        array: new direction
    """
    angle_ = get_angle(current_direction)
    angle_ += turn_angle
    angle_ = np.mod(angle_, 2 * np.pi)
    new_direction = np.array([np.cos(angle_), np.sin(angle_)])
    return new_direction


def wall_bounce_or_follow(current_velocity, wall, what_to_do="bounce"):
    """Given current direction, and wall and an instruction returns a new direction which is the result of implementing that instruction on the current direction
        wrt the wall. e.g. 'bounce' returns direction after elastic bounce off wall. 'follow' returns direction parallel to wall (closest to current heading)
    Args:
        current_direction (array): the current direction vector
        wall (array): start and end coordinates of the wall
        what_to_do (str, optional): 'bounce' or 'follow'. Defaults to 'bounce'.
    Returns:
        array: new direction
    """
    if what_to_do == "bounce":
        wall_perp = get_perpendicular(wall[1] - wall[0])
        if np.dot(wall_perp, current_velocity) <= 0:
            wall_perp = (
                -wall_perp
            )  # it is now the get_perpendicular with smallest angle to dir
        wall_par = wall[1] - wall[0]
        if np.dot(wall_par, current_velocity) <= 0:
            wall_par = -wall_par  # it is now the parallel with smallest angle to dir
        wall_par, wall_perp = (
            wall_par / np.linalg.norm(wall_par),
            wall_perp / np.linalg.norm(wall_perp),
        )  # normalise
        new_velocity = wall_par * np.dot(
            current_velocity, wall_par
        ) - wall_perp * np.dot(current_velocity, wall_perp)
    elif what_to_do == "follow":
        wall_par = wall[1] - wall[0]
        if np.dot(wall_par, current_velocity) <= 0:
            wall_par = -wall_par  # it is now the parallel with smallest angle to dir
        wall_par = wall_par / np.linalg.norm(wall_par)
        new_velocity = wall_par * np.dot(current_velocity, wall_par)

    return new_velocity


def get_perpendicular(a=None):
    """Given 2-vector, a, returns its perpendicular
    Args:
        a (array, optional): 2-vector direction. Defaults to None.
    Returns:
        array: perpendicular to a
    """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def intersect(a):
    # a numpy array with dimension [n, 2, 2, 2]
    # axis 0: line-pair, axis 1: two lines, axis 2: line delimiters axis 3: x and y coords
    # for each of the n line pairs a boolean is returned stating of the two lines intersect
    # Note: the edge case of a vertical line is not handled.
    m = (a[:, :, 1, 1] - a[:, :, 0, 1]) / (a[:, :, 1, 0] - a[:, :, 0, 0] + 1e-8)
    t = a[:, :, 0, 1] - m[:, :] * a[:, :, 0, 0]
    x = (t[:, 0] - t[:, 1]) / (m[:, 1] - m[:, 0] + 1e-8)
    y = m[:, 0] * x + t[:, 0]
    r = a.min(axis=2).max(axis=1), a.max(axis=2).min(axis=1)
    return (x >= r[0][:, 0]) & (x <= r[1][:, 0]) & (y >= r[0][:, 1]) & (y <= r[1][:, 1])


def saveToFile(Class, name, directory="../savedObjects/"):
    """Saves all attributes from a Class to file
    Args:
        Class: The class to save
        name (string): Name to save as
        directory (str, optional): [description]. Defaults to "../savedObjects/".
    """
    np.savez(directory + name + ".npz", Class.__dict__)
    return


def loadFromFile(Class, name, directory="../savedObjects/"):
    attributes_dictionary = np.load(directory + name + ".npz", allow_pickle=True)[
        "arr_0"
    ].item()
    print("Loading attributes...", end="")
    for key, value in attributes_dictionary.items():
        setattr(Class, key, value)
    print("done. use 'Class.__dict__.keys()'  to see avaiable attributes")


def update_params(Class, params: dict):
    """Updates parameters from a dictionary. 
    All parameters found in params will be updated to new value
    Args:
        params (dict): dictionary of parameters to change
        initialise (bool, optional): [description]. Defaults to False.
    """
    for key, value in params.items():
        setattr(Class, key, value)


def gridscore(grid):
    """Takes an array (the same shape as the environment) or list of arrays and 

    Args:
        arr (np.array): The "grid" firing field
    """
    ac = scipy.signal.correlate2d(grid, grid)
    ac_0 = ac.reshape(-1)
    ac_30 = scipy.ndimage.rotate(ac, 30, reshape=False).reshape(-1)
    ac_60 = scipy.ndimage.rotate(ac, 60, reshape=False).reshape(-1)
    ac_90 = scipy.ndimage.rotate(ac, 90, reshape=False).reshape(-1)
    ac_120 = scipy.ndimage.rotate(ac, 120, reshape=False).reshape(-1)
    ac_150 = scipy.ndimage.rotate(ac, 150, reshape=False).reshape(-1)

    C_30 = scipy.stats.pearsonr(ac_0, ac_30)[0]
    C_60 = scipy.stats.pearsonr(ac_0, ac_60)[0]
    C_90 = scipy.stats.pearsonr(ac_0, ac_90)[0]
    C_120 = scipy.stats.pearsonr(ac_0, ac_120)[0]
    C_150 = scipy.stats.pearsonr(ac_0, ac_150)[0]

    GS = (1 / 2) * (C_60 + C_120) - (1 / 3) * (C_30 + C_90 + C_150)

    return GS


if __name__ == "__main__":
    Env = Environment({"maze_type": "loop", "scale": 2, "dx": 0.015})

    Ag = Agent(
        {"Environment": Env, "drift_velocity": 0.2, "velocity_noise_scale": 0.0,}
    )

    Features = StateNeurons(
        {
            "Environment": Env,
            "Agent": Ag,
            "locations": Env.scatter_feature_centres(n=200, jitter=False),
            "widths": 0.1,
            "type": "gaussian",
            "name": "Features",
        }
    )

    HPC = PyramidalNeurons(
        {
            "Environment": Env,
            "Agent": Ag,
            "name": "HPC",
            "n": Features.n,
            "color": "C4",
        }
    )

    MEC = PyramidalNeurons(
        {"Environment": Env, "Agent": Ag, "name": "MEC", "n": 200, "color": "C5",}
    )

    HPC.add_compartment(
        params={
            "name": "basal",
            "input_layers": [Features],
            "eta": 0,
            "activation": "linear",
            "use_bias": False,
        }
    )
    HPC.add_compartment(
        params={
            "name": "apical",
            "input_layers": [MEC],
            # 'use_bias':True,
            # 'activation':'linear'
        }
    )

    MEC.add_compartment(
        params={
            "name": "basal",
            "input_layers": [HPC],
            # 'use_bias':True,
            # 'activation':'linear'
        }
    )

    MEC.add_compartment(
        params={
            "name": "apical",
            "input_layers": [MEC],
            # 'use_bias':True,
            # 'activation':'linear'
        }
    )

    MEC.top_layer = True
    HPC.Compartments["basal"].W = np.identity(HPC.n)

    Plotter = Visualiser(Agent=Ag, Environment=Env, Neurons=[Features, HPC, MEC])

    N_plot = "10"
    Plotter.plot_rate_map(
        layer_name="Features", neuron_id=N_plot,
    )
    Plotter.plot_rate_map(layer_name="HPC", route="basal", neuron_id=N_plot)
    Plotter.plot_rate_map(layer_name="HPC", route="apical", neuron_id=N_plot)
    Plotter.plot_rate_map(layer_name="MEC", route="basal", neuron_id=N_plot)
    Plotter.plot_rate_map(layer_name="MEC", route="apical", neuron_id=N_plot)

    Tmax_min = 1
    dt = (HPC.theta_params["theta_freq"] ** (-1)) / 10
    steps = int(Tmax_min * 60 / dt)
    for i in tqdm(range(steps)):
        Ag.update_state(dt)
        Features.update_state()
        HPC.update_dendrite_states()
        MEC.update_dendrite_states()
        HPC.update_soma_states()
        MEC.update_soma_states()
        HPC.update_weights()
        MEC.update_weights()
