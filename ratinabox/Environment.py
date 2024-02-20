import ratinabox

import copy
import pprint
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import shapely


import warnings
from typing import Union

from ratinabox import utils
from ratinabox.Agent import Agent

"""ENVIRONMENT"""


class Environment:
    """Environment class: defines the Environment in which the Agent lives.
    This class needs no other classes to initialise it. It exists independently of any Agents or Neurons.

    2D Environments (the default) can contain
        • a boundary  ------- if non-default (square), this must be given at initialisation
        • walls ------------- can be given at initialisation or added after
        • holes ------------- can be given at initialisation or added after
        • objects ----------- can only be given after initialisation using Env.add_object
        (beware, some Neurons classes depend on walls and holes being define before they are initialised, see their documentation - this means it is safest to define walls and holes at initialisation)

    A default parameters dictionary (with descriptions) can be found in __init__()

    List of functions...
        ...that you might use:
            • add_wall()
            • add_hole()
            • add_object()
            • plot_environment()
        ...that you probably won't directly use/use very often:
            • sample_positions()
            • discretise_environment()
            • get_vectors_between___accounting_for_environment()
            • get_distances_between___accounting_for_environment()
            • check_if_posision_is_in_environment()
            • check_wall_collisions()
            • vectors_from_walls()
            • apply_boundary_conditions()
            • add_object()


    The default_params are
    default_params = {
            "dimensionality": "2D",
            "boundary_conditions": "solid",
            "scale": 1,
            "aspect": 1,
            "dx": 0.01,
            "walls":[],#defaults to no walls
            "boundary":None #defaults to square
            "holes":[],#defaults to no holes
            "objects":[],#defaults to no objects
        }
    """

    default_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "scale": 1,  # scale of environment (in metres)
        "aspect": 1,  # x/y aspect ratio for the (rectangular) 2D environment (how wide this is relative to tall). Only applies if you are not passing a boundary
        "dx": 0.01,  # discretises the environment (for plotting purposes only)
        "boundary": None,  # coordinates [[x0,y0],[x1,y1],...] of the corners of a 2D polygon bounding the Env (if None, Env defaults to rectangular). Corners must be ordered clockwise or anticlockwise, and the polygon must be a 'simple polygon' (no holes, doesn't self-intersect).
        "walls": [],  # a list of loose walls within the environment. Each wall in the list can be defined by it's start and end coords [[x0,y0],[x1,y1]]. You can also manually add walls after init using Env.add_wall() (preferred).
        "holes": [],  # coordinates [[[x0,y0],[x1,y1],...],...] of corners of any holes inside the Env. These must be entirely inside the environment and not intersect one another. Corners must be ordered clockwise or anticlockwise. holes has 1-dimension more than boundary since there can be multiple holes
        "objects": [], # a list of objects within the environment. Each object is defined by its position [[x0,y0],[x1,y1],...] for 2D environments and [[x0],[x1],...] for 1D environments. By default all objects are type 0, alternatively you can manually add objects after init using Env.add_object(object, type) (preferred).
    }

    def __init__(self, params={}):
        """Initialise Environment, takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary.

        Args:
            params (dict, optional). Defaults to {}.
        """

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        utils.update_class_params(self, self.params, get_all_defaults=True)
        utils.check_params(self, params.keys())

        self.Agents : list[Agent] = []  # each new Agent will append itself to this list
        self.agents_dict = {} # this is a dictionary which allows you to lookup a agent by name

        if self.dimensionality == "1D":
            self.D = 1
            self.extent = np.array([0, self.scale])
            self.centre = np.array([self.scale / 2, self.scale / 2])
            if self.boundary is not None:
                warnings.warn(
                    "You have passed a boundary into a 1D environment. This is ignored."
                )
                self.boundary = None

            for feature_for_2D_only in ["holes", "walls"]:
                if len(getattr(self, feature_for_2D_only)) > 0:
                    warnings.warn(
                        f"You have passed {feature_for_2D_only} into a 1D environment. "
                        "This is ignored."
                    )
                    setattr(self, feature_for_2D_only, list())

        elif self.dimensionality == "2D":
            self.D = 2
            self.is_rectangular = False
            if (
                self.boundary is None
            ):  # Not passing coordinates of a boundary, fall back to default rectangular env
                self.is_rectangular = True
                self.boundary = [
                    [0, 0],
                    [self.aspect * self.scale, 0],
                    [self.aspect * self.scale, self.scale],
                    [0, self.scale],
                ]
            else:  # self.boundary coordinates passed in the input params
                self.is_rectangular = False
            b = self.boundary

            # make the arena walls
            self.walls = np.array(self.walls).reshape(-1, 2, 2)
            if (self.boundary_conditions == "periodic") and (
                self.is_rectangular == False
            ):
                warnings.warn(
                    "Periodic boundary conditions are only allowed in rectangual environments. Changing boundary conditions to 'solid'."
                )
                self.params["boundary_conditions"] = "solid"
            elif self.boundary_conditions == "solid":
                boundary_walls = np.array(
                    [
                        [b[(i + 1) if (i + 1) < len(b) else 0], b[i]]
                        for i in range(len(b))
                    ]
                )  # constructs walls from points on polygon
                self.walls = np.vstack((boundary_walls, self.walls))
            # make the hole walls (if there are any)
            self.holes_polygons = []
            self.has_holes = False
            if len(self.holes) > 0:
                assert (
                    np.array(self.holes).ndim == 3
                ), "Incorrect dimensionality for holes list. It must be a list of lists of coordinates"

                self.has_holes = True
                for h in self.holes:
                    hole_walls = np.array(
                        [
                            [h[(i + 1) if (i + 1) < len(h) else 0], h[i]]
                            for i in range(len(h))
                        ]
                    )
                    self.walls = np.vstack((self.walls, hole_walls))
                    self.holes_polygons.append(shapely.Polygon(h))
            self.boundary_polygon = shapely.Polygon(self.boundary)

            # make some other attributes
            left = min([c[0] for c in b])
            right = max([c[0] for c in b])
            bottom = min([c[1] for c in b])
            top = max([c[1] for c in b])
            self.centre = np.array([(left + right) / 2, (top + bottom) / 2])
            self.extent = np.array(
                [left, right, bottom, top]
            )  # [left,right,bottom,top] ]the "extent" which will be plotted, always a rectilinear rectangle which will be the extent of all matplotlib plots

        # make list of "objects" within the Env
        self.passed_in_objects = copy.deepcopy(self.objects)
        self.objects = {
            "objects": np.empty((0, self.D)),
            "object_types": np.empty(0, int),
        }
        self.n_object_types = 0
        self.object_colormap = "rainbow_r"
        if len(self.passed_in_objects) > 0:
            for o in self.passed_in_objects:
                self.add_object(o, type=0)

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

    @classmethod
    def get_all_default_params(cls, verbose=False):
        """Returns a dictionary of all the default parameters of the class, including those inherited from its parents."""
        all_default_params = utils.collect_all_params(cls, dict_name="default_params")
        if verbose:
            pprint.pprint(all_default_params)
        return all_default_params

    
    def agent_lookup(self, agent_names:Union[str, list[str]]  = None) -> list[Agent]:
        '''
        This function will lookup a agent by name and return it. This assumes that the agent has been 
        added to the Environment.agents list and that each agent object has a unique name associated with it.


        Args:
            agent_names (str, list[str]): the name of the agent you want to lookup. 
        
        Returns:
            agents (list[Agent]): a list of agents that match the agent_names. If agent_names is a string, then a list of length 1 is returned. If agent_names is None, then None is returned

        '''

        if agent_names is None:
            return None
        
        if isinstance(agent_names, str):
            agent_names = [agent_names]

        agents: list[Agent] = []

        for agent_name in agent_names:
            agent = self._agent_lookup(agent_name)
            agents.append(agent)

        return agents
    
    def _agent_lookup(self, agent_name: str) -> Agent:

        """
        Helper function for agent lookup. 

        The procedure will work as follows:-
        1. If agent_name is None, the function will return None
        2. if the agent_name is found in self.agents_dict, then   self.agents_dict[agent_name] is returned
        3. Else
            - a loop over self.agents list is performed and each agent.name is checked against agent_name
            - if found the agent is added to the self.agents_dict and returned
            - if not found None is returned

        Args:
            agent_name: the name of the agent you want to lookup
        """

        if agent_name is None:
            return None
        
        if agent_name in self.agents_dict:
            return self.agents_dict[agent_name]
        else:
            for agent in self.Agents:
                if agent.name == agent_name:
                    self.agents_dict[agent_name] = agent
                    return agent
        
        raise ValueError('Agent name not found in Environment.agents list. Make sure the there no typos. agent name is case sensitive')
    
    def add_agent(self, agent: Agent = None):
        """
        This function adds a agent to the Envirnoment.Agents list and also adds it to the Agent.agents_dict dictionary
        which allows you to lookup a agent by name.

        This also ensures that the agent is associated with this Agent and has a unique name. 
        Otherwise an index is appended to the name to make it unique and a warning is raised.

        Args:
            agent: the agent object you want to add to the Agent.Agent list

        """
        assert agent is not None and isinstance(agent, Agent), TypeError("agent must be a ratinabox Agent type" )

        #check if a agent with this name already exists
        if agent.name in self.agents_dict:
            
            # we try with the name of the agent + a number

            idx = len(self.Agents)
            name = f"agent_{idx}"

            if name in self.agents_dict:
                raise ValueError(f"A agent with the name {agent.name}  and  {name} already exists. Please choose a unique name for each agent.\n\
                            This can cause trouble with lookups")
            else:
                agent.name = name 
                warnings.warn(f"A agent with the name {agent.name} already exists. Renaming to {name}")
        
        self.Agents.append(agent)
        self.agents_dict[agent.name] = agent


    def remove_agent(self, agent: Union[str, Agent]  = None):
        """
        A function to remove a agent from the Environment.Agents list and the Environment.agents_dict dictionary

        Args:
            agent (str|Agent): the name of the agent you want to remove or the agent object itself
        """

        if isinstance(agent, str):
            agent = self._agent_lookup(agent)
        
        if agent is None:
            return None

        self.Agents.remove(agent)
        self.agents_dict.pop(agent.name)
        
    

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

    def add_hole(self, hole):
        """Add a hole to the (2D) environment.
        Extends self.holes array to include one new hole.
        Args:
            hole (np.array): n_corners x 2 array [[x1,y1],[x2,y2]] where n_corners is the number of corners of the hole (so must be >= 3, holes can't be lines)
        """
        assert self.dimensionality == "2D", "can only add holes into a 2D environment"
        assert len(hole) >= 3, "holes must have at least 3 corners"

        self.holes.append(hole)
        self.has_holes = True
        hole_walls = np.array(
            [
                [hole[(i + 1) if (i + 1) < len(hole) else 0], hole[i]]
                for i in range(len(hole))
            ]
        )

        self.walls = np.vstack((self.walls, hole_walls.reshape(-1, 2, 2)))
        self.holes_polygons.append(shapely.Polygon(hole))
        return

    def add_object(self, object, type="new"):
        """Adds an object to the environment. Objects can be seen by object vector cells but otherwise do very little. Objects have "types". By default when adding a new object a new type is created (n objects n types) but you can specify a type (n objects <n types). Boundary vector cells may be selective to one type.

        Args:
            object (array): The location of the object, 2D list or array
            type (_type_): The "type" of the object, any integer. By default ("new") a new type is made s.t. the first object is type 0, 2nd type 1... n'th object will be type n-1, etc.... If type == "same" then the added object has the same type as the last

        """
        object = np.array(object).reshape(1, -1)
        assert object.shape[1] == self.D

        if type == "new":
            type = self.n_object_types
        elif type == "same":
            if len(self.objects["object_types"]) == 0:
                type = 0
            else:
                type = self.objects["object_types"][-1]
        else:
            assert type <= self.n_object_types, print(
                f"Newly added object must be one of the existing types (currently {np.unique(self.objects['object_types'])}) or the next one along ({self.n_object_types}), not {type}"
            )
        type = np.array([type], int)

        self.objects["objects"] = np.append(self.objects["objects"], object, axis=0)
        self.objects["object_types"] = np.append(
            self.objects["object_types"], type, axis=0
        )
        self.n_object_types = len(np.unique(self.objects["object_types"]))
        return

    def plot_environment(self, 
                         fig=None, 
                         ax=None, 
                         gridlines=False,
                         plot_objects=True,
                         autosave=None,
                         **kwargs,):
        """Plots the environment on the x axis, dark grey lines show the walls
        Args:
            fig,ax: the fig and ax to plot on (can be None)
            gridlines: if True, plots gridlines
            plot_objects: if True, plots objects
            autosave: if True, will try to save the figure to the figure directory `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
        Returns:
            fig, ax: the environment figures, can be used for further downstream plotting.
        """
        if self.dimensionality == "1D":
            extent = self.extent
            if fig is None and ax is None:
                fig, ax = plt.subplots(
                    figsize=(
                        ratinabox.MOUNTAIN_PLOT_WIDTH_MM / 25 * (extent[1] - extent[0]),
                        1,
                    )
                )
            ax.set_xlim(left=extent[0], right=extent[1])
            ax.spines["left"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.spines["bottom"].set_position("zero")
            ax.spines["top"].set_color("none")
            ax.set_yticks([])
            ax.set_xticks([extent[0], extent[1]])
            ax.set_xlabel("Position / m")

            # plot objects, if applicable
            if plot_objects:
                object_cmap = matplotlib.colormaps[self.object_colormap]
                for i, object in enumerate(self.objects["objects"]):
                    object_color = object_cmap(
                        self.objects["object_types"][i]
                        / (self.n_object_types - 1 + 1e-8)
                    )
                    ax.scatter(
                        object[0],
                        0,
                        facecolor=[0, 0, 0, 0],
                        edgecolors=object_color,
                        s=10,
                        zorder=2,
                        marker="o",
                    )

        if self.dimensionality == "2D":
            extent, walls = self.extent, self.walls
            if fig is None and ax is None:
                fig, ax = plt.subplots(
                    figsize=(ratinabox.FIGURE_INCH_PER_ENVIRONMENT_METRE * (extent[1] - extent[0]), ratinabox.FIGURE_INCH_PER_ENVIRONMENT_METRE * (extent[3] - extent[2]))
                )
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) #remove border
            # plot background/arena
            background = matplotlib.patches.Polygon(
                xy=np.array(self.boundary), facecolor=ratinabox.LIGHTGREY, zorder=-1
            )
            setattr(background, "name", "background")
            ax.add_patch(background)

            # plot holes
            for hole in self.holes:
                hole_ = matplotlib.patches.Polygon(
                    xy=np.array(hole),
                    facecolor="white",
                    linewidth=1.0,
                    edgecolor="white",
                    zorder=1,
                )
                setattr(background, "name", "hole")
                ax.add_patch(hole_)

            # plot anti-arena (difference between area and the full extent shown)
            if self.is_rectangular is False:
                # size = self.extent[1]-self.extent[0]
                extent_corners = np.array(
                    [
                        [self.extent[0], self.extent[2]],
                        [self.extent[1], self.extent[2]],
                        [self.extent[1], self.extent[3]],
                        [self.extent[0], self.extent[3]],
                    ]
                )
                extent_poly = shapely.Polygon(extent_corners)
                arena_poly = shapely.Polygon(np.array(self.boundary))
                anti_arena_poly = extent_poly.difference(arena_poly)
                if type(anti_arena_poly) == shapely.Polygon:
                    polys = [anti_arena_poly]
                elif type(anti_arena_poly) == shapely.MultiPolygon:
                    polys = anti_arena_poly.geoms
                for poly in polys:
                    (x, y) = poly.exterior.coords.xy
                    coords = np.stack((list(x), list(y)), axis=1)
                    anti_arena_segment = matplotlib.patches.Polygon(
                        xy=np.array(coords),
                        facecolor="white",
                        linewidth=1.0,
                        edgecolor="white",
                        zorder=1,
                    )
                    setattr(background, "name", "hole")
                    ax.add_patch(anti_arena_segment)

            # plot walls
            for wall in walls:
                ax.plot(
                    [wall[0][0], wall[1][0]],
                    [wall[0][1], wall[1][1]],
                    color=ratinabox.GREY,
                    linewidth=4.0,
                    solid_capstyle="round",
                    zorder=2,
                )

            # plot objects, if applicable
            if plot_objects: 
                object_cmap = matplotlib.colormaps[self.object_colormap]
                for i, object in enumerate(self.objects["objects"]):
                    object_color = object_cmap(
                        self.objects["object_types"][i]
                        / (self.n_object_types - 1 + 1e-8)
                    )
                    ax.scatter(
                        object[0],
                        object[1],
                        facecolor=[0, 0, 0, 0],
                        edgecolors=object_color,
                        s=10,
                        zorder=2,
                        marker="o",
                    )

            #plot grid lines
            ax.set_aspect("equal")
            if gridlines == True:
                ax.grid(True, color=ratinabox.DARKGREY, linewidth=0.5, linestyle="--")
                #turn off the grid lines on the edges
                ax.spines["left"].set_color("none")
                ax.spines["right"].set_color("none")
                ax.spines["bottom"].set_color("none")
                ax.spines["top"].set_color("none")
                ax.tick_params(length=0)

            else: 
                ax.grid(False)
                ax.axis("off")
            ax.set_xlim(left=extent[0] - 0.02, right=extent[1] + 0.02)
            ax.set_ylim(bottom=extent[2] - 0.02, top=extent[3] + 0.02)
            # ax.set_xlim(left=extent[0], right=extent[1])
            # ax.set_ylim(bottom=extent[2], top=extent[3])

        ratinabox.utils.save_figure(fig, "Environment", save=autosave)

        return fig, ax

    def sample_positions(self, n=10, method="uniform_jitter"):
        """Scatters 'n' locations across the environment which can act as, for example, the centres of gaussian place fields, or as a random starting position.
        If method == "uniform" an evenly spaced grid of locations is returned.  If method == "uniform_jitter" these locations are jittered slightly (i.e. random but span the space). Note; if n doesn't uniformly divide the size (i.e. n is not a square number in a square environment) then the largest number that can be scattered uniformly are found, the remaining are randomly placed.
        Args:
            n (int): number of features
            method: "uniform", "uniform_jittered" or "random" for how points are distributed
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
                # random scatter positions and check they aren't in holes etc. 
                positions = np.zeros((n, 2))
                positions[:, 0] = np.random.uniform(
                    self.extent[0], self.extent[1], size=n
                )
                positions[:, 1] = np.random.uniform(
                    self.extent[2], self.extent[3], size=n
                )
                if (self.is_rectangular is False) or (self.has_holes is True):
                # in this case, the positions you have sampled within the extent of the environment may not actually fall within it's legal area (i.e. they could be outside the polygon boundary or inside a hole). Brute force this by randomly resampling these points until all fall within the env.
                    for i, pos in enumerate(positions):
                        if self.check_if_position_is_in_environment(pos) == False:
                            pos = self.sample_positions(n=1, method="random").reshape(
                                -1
                            )  # this recursive call must pass eventually, assuming the env is sufficiently large. this is why we don't need a while loop
                            positions[i] = pos
            elif method[:7] == "uniform":
                # uniformly scatter positions on a square grid and check they aren't in holes etc.
                ex = self.extent
                area = (ex[1] - ex[0]) * (ex[3] - ex[2])
                if (self.has_holes is True): 
                    area -= sum(shapely.geometry.Polygon(hole).area for hole in self.holes)
                delta = np.sqrt(area / n)
                x = np.linspace(ex[0] + delta /2, ex[1] - delta /2, int((ex[1] - ex[0])/delta))
                y = np.linspace(ex[2] + delta /2, ex[3] - delta /2, int((ex[3] - ex[2])/delta))
                positions = np.array(np.meshgrid(x, y)).reshape(2, -1).T
                
                if (self.is_rectangular is False) or (self.has_holes is True):
                    # in this case, the positions you have sampled within the extent of the environment may not actually fall within it's legal area (i.e. they could be outside the polygon boundary or inside a hole). delete those that do for resampling later. 
                    delpos = [i for (i,pos) in enumerate(positions) if self.check_if_position_is_in_environment(pos) == False]
                    positions = np.delete(positions,delpos,axis=0) # this will delete illegal positions

                n_uniformly_distributed = positions.shape[0]
                if method[7:] == "_jitter":
                    # add jitter to the uniformly distributed positions
                    positions += np.random.uniform(
                        -0.45 * delta, 0.45 * delta, positions.shape 
                    ) 
                n_remaining = n - n_uniformly_distributed
                if n_remaining > 0:
                    # sample remaining from available positions with further jittering (delta = delta/2)
                    positions_remaining = np.array([positions[i] for i in np.random.choice(range(len(positions)),n_remaining, replace=True)])
                    delta /= 2
                    positions_remaining += np.random.uniform(
                        -0.45 * delta, 0.45 * delta, positions_remaining.shape
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
        vectors = utils.get_vectors_between(
            pos1=pos1, pos2=pos2, line_segments=line_segments
        )
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
            return_vectors (False): If True, returns the distances and the vectors (from pos2 to pos1) as a tuple
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
                assert (
                    boundary_conditions == "solid"
                ), "geodesic geometry is not available for periodic boundary conditions"
                assert (
                    len(walls) <= 5
                ), """unfortunately geodesic geometry is only defined in closed rooms with one additional wall. Try using "line_of_sight" or "euclidean" instead. 
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
                    flattened_wall_obstructs_view_of_cell = (
                        wall_obstructs_view_of_cell.reshape(-1)
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
        if self.dimensionality == "1D":
            if (pos[0] > self.extent[0]) and (pos[0] < self.extent[1]):
                return True
            else:
                return False

        if self.dimensionality == "2D":
            if (
                self.is_rectangular == True and self.holes is None
            ):  # fast way (don't use shapely)
                return all(
                    [
                        (pos[0] > self.extent[0]),
                        (pos[0] < self.extent[1]),
                        (pos[1] > self.extent[2]),
                        (pos[1] < self.extent[3]),
                    ]
                )
            else:  # the slow way (polygon check for environment boundaries and each hole within env)
                is_in = True
                is_in *= self.boundary_polygon.contains(
                    shapely.Point(pos)
                )  # assert inside area
                if self.has_holes is True:
                    for hole_poly in self.holes_polygons:
                        is_in *= not hole_poly.contains(
                            shapely.Point(pos)
                        )  # assert inside area, "not" because if it's in the hole it isn't in the environment
                return bool(is_in)

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
        walls_to_pos_vectors = utils.shortest_vectors_from_points_to_lines(
            pos, self.walls
        )[0]
        return walls_to_pos_vectors

    def apply_boundary_conditions(self, pos):
        """Performs a boundary condition check. If pos is OUTside the environment and the boundary conditions are solid then a different position, safely located 1cm within the environmnt, is returned. If pos is OUTside the environment but boundary conditions are periodic its position is looped to the other side of the environment appropriately.
        Args:
            pos (np.array): 1 or 2 dimensional position
        returns new_pos
        TODO update this so if pos is in one of the holes the Agent is returned to the ~nearest legal location inside the Environment
        """
        if self.check_if_position_is_in_environment(pos) is True: return

        if self.dimensionality == "1D":
            if self.boundary_conditions == "periodic":
                pos = pos % self.extent[1]
            if self.boundary_conditions == "solid":
                pos = min(max(pos, self.extent[0] + 0.01), self.extent[1] - 0.01)
                pos = np.reshape(pos, (-1))
        elif self.dimensionality == "2D":
            if self.is_rectangular == True:
                if not (
                    matplotlib.path.Path(self.boundary).contains_point(
                        pos, radius=-1e-10
                    )
                ):  # outside the bounding environment (i.e. not just in a hole), apply BCs
                    if self.boundary_conditions == "periodic":
                        pos[0] = pos[0] % self.extent[1]
                        pos[1] = pos[1] % self.extent[3]
                    if self.boundary_conditions == "solid":
                        # in theory this wont be used as wall bouncing catches it earlier on
                        pos[0] = min(
                            max(pos[0], self.extent[0] + 0.01),
                            self.extent[1] - 0.01,
                        )
                        pos[1] = min(
                            max(pos[1], self.extent[2] + 0.01),
                            self.extent[3] - 0.01,
                        )
                else:  # in this case, must just be in a hole. sample new position (there should be a better way to do this but, in theory, this isn't used)
                    pos = self.sample_positions(n=1, method="random").reshape(-1)
            else:  # polygon shaped env, just resample random position
                pos = self.sample_positions(n=1, method="random").reshape(-1)
        return pos
