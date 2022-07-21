from ratinabox.utils import *

verbose = False

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


"""NEURONS"""
"""Parent Class"""


class Neurons:
    """The Neuron class defines a population of Neurons. All Neurons have firing rates which depend explicity on (that is, they "encode") the state of the Agent. As the Agent moves the firing rate of the cells adjust accordingly. 

    All Neuron classes must be initalised with the Agent (to whom these cells belong) since the Agent determines teh firingrates through its position and velocity. The Agent class will itself contain the Environment. Both the Agent (position/velocity) and the Environment (geometry, walls etc.) determine the firing rates. Optionally (but likely) an input dictionary 'params' specifying other params will be given.
    
    This is a generic Parent class. We provide several SubClasses of it. These include: 
    • PlaceCells()
    • GridCells()
    • BoundaryVectorCells()
    • VelocityCells()
    • HeadDirectionCells()
    • SpeedCells()
    • FeedForwardLayer()

    The unique function in each child classes is get_state(). Whenever Neurons.update() is called Neurons.get_state() is then called to calculate and returns the firing rate of the cells at the current moment in time. This is then saved. In order to make your own Neuron subclass you will need to write a class with the following mandatory structure: 

    MyNeuronClass(Neurons):
        def __init__(self, 
                     Agent,
                     params={}): #<-- do not change these 

            default_params = {'a_default_param":3.14159}
            
            default_params.update(params)
            self.params = default_params
            super().__init__(self.params)
        
        def get_state(self,
                      evaluate_at='agent',
                      **kwargs) #<-- do not change these 
            
            firingrate = .....
            ###
                Insert here code which calculates the firing rate.
                This may work differently depending on what you set evaluate_at as. For example, evaluate_at == 'agent' should means that the position or velocity (or whatever determines the firing rate) will by evaluated using the agents current state. You might also like to have an option like evaluate_at == "all" (all positions across an environment are tested simultaneously - plot_rate_map() tries to call this, for example) or evaluate_at == "last" (in a feedforward layer just look at the last firing rate saved in the input layers saves time over recalculating them.). **kwargs allows you to pass position or velocity in manually.  

                By default, the Neurons.update() calls Neurons.get_state() raw. So write the default behaviour of get_state to be what you want it to do in the main training loop. 
            ###

            return firingrate 
            
    As we have written them, Neuron subclasses which have well defined ground truth receptive fields (PlaceCells, GridCells but not VelocityCells etc.) can also be queried for any arbitrary pos/velocity (i.e. not just the Agents current state) by passing these in directly to the function "get_state(evaluate_at='all') or get_state(evaluate_at=None, pos=my_array_of_positons)". This calculation is vectorised and relatively fast, returning an array of firing rates one for each position. It is what is used when you try Neuron.plot_rate_map(). 
    

    List of key functions...
        ..that you're likely to use: 
            • update()
            • plot_rate_timeseries()
            • plot_rate_map()
        ...that you might not use but could be useful:
            • save_to_history()
            • boundary_vector_preference_function()   
    """

    def __init__(self, Agent, params={}):
        """Initialise Neurons(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.
        
        Typically you will not actually initialise a Neurons() class, instead you will initialised by one of it's subclasses. 
        """
        default_params = {
            "n": 10,
            "name": "Neurons",
            "color": None,  # just for plotting
        }
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        update_class_params(self, self.params)

        self.firingrate = np.zeros(self.n)
        self.history = {}
        self.history["t"] = []
        self.history["firingrate"] = []
        self.history["spikes"] = []

        if verbose is True:
            print(
                f"\nA Neurons() class has been initialised with parameters f{self.params}. Use Neurons.update() to update the firing rate of the Neurons to correspond with the Agent.Firing rates and spikes are saved into the Agent.history dictionary. Plot a timeseries of the rate using Neurons.plot_rate_timeseries(). Plot a rate map of the Neurons using Neurons.plot_rate_map()."
            )

    def update(self):
        firingrate = self.get_state()
        self.firingrate = firingrate.reshape(-1)
        self.save_to_history()
        return

    def plot_rate_timeseries(
        self,
        t_start=0,
        t_end=None,
        chosen_neurons="all",
        spikes=True,
        fig=None,
        ax=None,
        xlim=None,
    ):
        """Plots a timeseries of the firing rate of the neurons between t_start and t_end

        Args:
            • t_start (int, optional): _description_. Defaults to 0.
            • t_end (int, optional): _description_. Defaults to 60.
            • chosen_neurons: Which neurons to plot. string "10" or 10 will plot ten of them, "all" will plot all of them, "12rand" will plot 12 random ones. A list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "all".
            chosen_neurons (str, optional): Which neurons to plot. string "10" will plot 10 of them, "all" will plot all of them, a list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "10".
            • plot_spikes (bool, optional): If True, scatters exact spike times underneath each curve of firing rate. Defaults to True.
            the below params I just added for help with animations
            • fig, ax: the figure, axis to plot on (can be None)
            xlim: fix xlim of plot irrespective of how much time you're plotting 
        Returns:
            fig, ax
        """
        t = np.array(self.history["t"])
        # times to plot
        if t_end is None:
            t_end = t[-1]
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end)))
        rate_timeseries = np.array(self.history["firingrate"])
        spike_data = np.array(self.history["spikes"])
        t = t[startid:endid]
        rate_timeseries = rate_timeseries[startid:endid]
        spike_data = spike_data[startid:endid]

        # neurons to plot
        chosen_neurons = self.return_list_of_neurons(chosen_neurons)

        firingrates = rate_timeseries[:, chosen_neurons].T
        fig, ax = mountain_plot(
            X=t / 60,
            NbyX=firingrates,
            color=self.color,
            xlabel="Time / min",
            ylabel="Neurons",
            xlim=None,
            fig=fig,
            ax=ax,
        )

        if spikes == True:
            for i in range(len(chosen_neurons)):
                time_when_spiked = t[spike_data[:, chosen_neurons[i]]] / 60
                h = (i + 1 - 0.1) * np.ones_like(time_when_spiked)
                ax.scatter(
                    time_when_spiked,
                    h,
                    color=(self.color or "C1"),
                    alpha=0.5,
                    s=2,
                    linewidth=0,
                )

        ax.set_xticks([t_start / 60, t_end / 60])
        if xlim is not None:
            ax.set_xlim(right=xlim / 60)
            ax.set_xticks([0, xlim / 60])

        return fig, ax

    def plot_rate_map(
        self,
        chosen_neurons="all",
        method="groundtruth",
        spikes=False,
        fig=None,
        ax=None,
        shape=None,
        t_start=0,
        t_end=None,
        **kwargs,
    ):
        """Plots rate maps of neuronal firing rates across the environment
        Args:
            •chosen_neurons: Which neurons to plot. string "10" will plot 10 of them, "all" will plot all of them, a list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "10".
            
            • method: "groundtruth" "history" "neither": which method to use. If "analytic" (default) tries to calculate rate map by evaluating firing rate at all positions across the environment (note this isn't always well defined. in which case...). If "history", plots ratemap by a weighting a histogram of positions visited by the firingrate observed at that position. If "neither" (or anything else), then neither. 

            • spikes: True or False. Whether to display the occurence of spikes. If False (default) no spikes are shown. If True both ratemap and spikes are shown.

            • fig, ax (the fig and ax to draw on top of, optional) 

            • shape is the shape of the multiplanlle figure, must be compatible with chosen neurons

            • t_start, t_end: i nthe case where you are plotting spike, or using historical data to get rate map, this restricts the timerange of data you are using 
            • kwargs are sent to get_state and can be ignore if you don't need to use them
        
        Returns:
            fig, ax 
        """
        if method == "groundtruth":
            try:
                rate_maps = self.get_state(evaluate_at="all", **kwargs)
            except Exception as e:
                print(
                    "It was not possible to get the rate map by evaluating the firing rate at all positions across the Environment. This is probably because the Neuron class does not support, or it does not have an groundtruth receptive field. Instead, plotting rate map by weighted position histogram method. Here is the error:"
                )
                print("Error: ", e)
                print("yeet")
                method = "history"

        if method == "history" or spikes == True:
            t = np.array(self.history["t"])
            # times to plot
            t_end = t_end or t[-1]
            startid = np.argmin(np.abs(t - (t_start)))
            endid = np.argmin(np.abs(t - (t_end)))
            pos = np.array(self.Agent.history["pos"])[startid:endid]
            t = t[startid:endid]

            if method == "history":
                rate_timeseries = np.array(self.history["firingrate"])[startid:endid].T
                if len(rate_timeseries) == 0:
                    print("No historical data with which to calculate ratemap.")
            if spikes == True:
                spike_data = np.array(self.history["spikes"])[startid:endid].T
                if len(spike_data) == 0:
                    print("No historical data with which to plot spikes.")

        if self.color == None:
            coloralpha = None
        else:
            coloralpha = list(matplotlib.colors.to_rgba(self.color))
            coloralpha[-1] = 0.5

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)

        if self.Agent.Environment.dimensionality == "2D":

            if fig is None and ax is None:
                if shape is None:
                    Nx, Ny = 1, len(chosen_neurons)
                else:
                    Nx, Ny = shape[0], shape[1]
                fig, ax = plt.subplots(
                    Nx, Ny, figsize=(3 * Ny, 3 * Nx), facecolor=coloralpha,
                )
            ax = np.array([ax])

            if len(chosen_neurons) != ax.size:
                print(
                    f"You are trying to plot a different number of neurons {len(chosen_neurons)} than the number of axes provided {ax.size}. Some might be missed. Either change this with the chosen_neurons argument or pass in a list of axes to plot on"
                )
            for (i, ax_) in enumerate(ax.flatten()):
                self.Agent.Environment.plot_environment(fig, ax_)

                if method == "groundtruth":
                    rate_map = rate_maps[chosen_neurons[i], :].reshape(
                        self.Agent.Environment.discrete_coords.shape[:2]
                    )
                    im = ax_.imshow(rate_map, extent=self.Agent.Environment.extent)
                if method == "history":
                    ex = self.Agent.Environment.extent
                    rate_timeseries_ = rate_timeseries[chosen_neurons[i], :]
                    rate_map = bin_data_for_histogramming(
                        data=pos, extent=ex, dx=0.05, weights=rate_timeseries_
                    )
                    im = ax_.imshow(rate_map, extent=ex, interpolation="bicubic")
                if spikes is True:
                    pos_where_spiked = pos[spike_data[chosen_neurons[i], :]]
                    ax_.scatter(
                        pos_where_spiked[:, 0],
                        pos_where_spiked[:, 1],
                        s=2,
                        linewidth=0,
                        alpha=0.7,
                    )
            return fig, ax

        if self.Agent.Environment.dimensionality == "1D":
            if method == "groundtruth":
                rate_maps = rate_maps[chosen_neurons, :]
                x = self.Agent.Environment.flattened_discrete_coords[:, 0]
            if method == "history":
                ex = self.Agent.Environment.extent
                pos_ = pos[:,0]
                rate_maps = []
                for neuron_id in chosen_neurons:
                    rate_map, x = bin_data_for_histogramming(
                        data=pos_,
                        extent=ex,
                        dx=0.01,
                        weights=rate_timeseries[neuron_id, :],
                    )
                    x, rate_map = interpolate_and_smooth(x, rate_map, sigma=0.03)
                    rate_maps.append(rate_map)
                rate_maps = np.array(rate_maps)

            if fig is None and ax is None:
                fig, ax = self.Agent.Environment.plot_environment(
                    height=len(chosen_neurons)
                )

            if method != "neither":
                fig, ax = mountain_plot(
                    X=x, NbyX=rate_maps, color=self.color, fig=fig, ax=ax,
                )

            if spikes is True:
                for i in range(len(chosen_neurons)):
                    pos_ = pos[:, 0]
                    pos_where_spiked = pos_[spike_data[chosen_neurons[i]]]
                    h = (i + 1 - 0.1) * np.ones_like(pos_where_spiked)
                    ax.scatter(
                        pos_where_spiked,
                        h,
                        color=(self.color or "C1"),
                        alpha=0.5,
                        s=2,
                        linewidth=0,
                    )
            ax.set_xlabel("Position / m")
            ax.set_ylabel("Neurons")

        return fig, ax

    def save_to_history(self):
        cell_spikes = np.random.uniform(0, 1, size=(self.n,)) < (
            self.Agent.dt * self.firingrate
        )
        self.history["t"].append(self.Agent.t)
        self.history["firingrate"].append(list(self.firingrate))
        self.history["spikes"].append(list(cell_spikes))

    def animate_rate_timeseries(self, t_end=None, chosen_neurons="all", speed_up=1):
        """Returns an animation (anim) of the firing rates, 25fps. 
        Should be saved using comand like 
        anim.save("./where_to_save/animations.gif",dpi=300)

        Args:
            • t_end (_type_, optional): _description_. Defaults to None.
            • chosen_neurons: Which neurons to plot. string "10" or 10 will plot ten of them, "all" will plot all of them, "12rand" will plot 12 random ones. A list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "all".

            • speed_up: #times real speed animation should come out at. 

        Returns:
            animation
        """

        if t_end == None:
            t_end = self.history["t"][-1]

        def animate(i, fig, ax, chosen_neurons, t_max, speed_up):
            t = self.history["t"]
            t_start = t[0]
            t_end = t[0] + (i + 1) * speed_up * 50e-3
            ax.clear()
            fig, ax = self.plot_rate_timeseries(
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                plot_spikes=True,
                fig=fig,
                ax=ax,
                xlim=t_max,
            )
            plt.close()
            return

        fig, ax = self.plot_rate_timeseries(
            t_start=0,
            t_end=10 * self.Agent.dt,
            chosen_neurons=chosen_neurons,
            xlim=t_end,
        )
        anim = matplotlib.animation.FuncAnimation(
            fig,
            animate,
            interval=50,
            frames=int(t_end / 50e-3),
            blit=False,
            fargs=(fig, ax, chosen_neurons, t_end, speed_up),
        )
        return anim

    def return_list_of_neurons(self, chosen_neurons="all"):
        """Returns a list of indices corresponding to neurons. 

        Args:
            which (_type_, optional): _description_. Defaults to "all".
                • "all": all neurons
                • "15" or 15:  15 neurons even spread from index 0 to n
                • "15rand": 15 randomly selected neurons
                • [4,8,23,15]: this list is returned (convertde to integers in case)
                • np.array([[4,8,23,15]]): the list [4,8,23,15] is returned 
        """
        if type(chosen_neurons) is str:
            if chosen_neurons == "all":
                chosen_neurons = np.arange(self.n)
            elif chosen_neurons.isdigit():
                chosen_neurons = np.linspace(0, self.n - 1, int(chosen_neurons)).astype(
                    int
                )
            elif chosen_neuron[-4:] == "rand":
                chosen_neurons = int(chosen_neurons[:-4])
                chosen_neurons = np.random.choice(
                    np.arange(self.n), size=chosen_neurons, replace=False
                )
        if type(chosen_neurons) is int:
            chosen_neurons = np.linspace(0, self.n - 1, chosen_neurons)
        if type(chosen_neurons) is list:
            chosen_neurons = list(np.array(chosen_neurons).astype(int))
            pass
        if type(chosen_neurons) is np.ndarray:
            chosen_neurons = list(chosen_neurons.astype(int))

        return chosen_neurons


"""Specific subclasses """


class PlaceCells(Neurons):
    """The PlaceCells class defines a population of PlaceCells. This class is a subclass of Neurons() and inherits it properties/plotting functions.  

    Must be initialised with an Agent and a 'params' dictionary. 

    PlaceCells defines a set of 'n' place cells scattered across the environment. The firing rate is a functions of the distance from the Agent to the place cell centres. This function (params['description'])can be:
        • gaussian (default)
        • gaussian_threshold
        • diff_of_gaussians
        • top_hat
        • one_hot
    
    List of functions: 
        • get_state()
        • plot_place_cell_locations()
    """

    def __init__(self, Agent, params={}):
        """Initialise PlaceCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.
        """
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
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        if self.place_cell_centres is None:
            self.place_cell_centres = self.Agent.Environment.sample_positions(
                n=self.n, method="uniform_jitter"
            )
        else:
            self.n = self.place_cell_centres.shape[0]
        self.place_cell_widths = self.widths * np.ones(self.n)

        if verbose is True:
            print(
                f"PlaceCells successfully initialised. You can see where they are centred at using PlaceCells.plot_place_cell_locations()"
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
        dist = self.Agent.Environment.get_distances_between___accounting_for_environment(
            self.place_cell_centres, pos, wall_geometry=self.wall_geometry
        )  # distances to place cell centres
        widths = np.expand_dims(self.place_cell_widths, axis=-1)

        if self.description == "gaussian":
            firingrate = np.exp(-(dist ** 2) / (2 * (widths ** 2)))
        if self.description == "gaussian_threshold":
            firingrate = np.maximum(
                np.exp(-(dist ** 2) / (2 * (widths ** 2))) - np.exp(-1 / 2), 0,
            ) / (1 - np.exp(-1 / 2))
        if self.description == "diff_of_gaussians":
            ratio = 1.5
            firingrate = np.exp(-(dist ** 2) / (2 * (widths ** 2))) - (
                1 / ratio ** 2
            ) * np.exp(-(dist ** 2) / (2 * ((ratio * widths) ** 2)))
            firingrate *= ratio ** 2 / (ratio ** 2 - 1)
        if self.description == "one_hot":
            closest_centres = np.argmin(np.abs(dist), axis=0)
            firingrate = np.eye(self.n)[closest_centres].T
        if self.description == "top_hat":
            firingrate = 1 * (dist < self.widths)

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def plot_place_cell_locations(self):
        fig, ax = self.Agent.Environment.plot_environment()
        place_cell_centres = self.place_cell_centres
        ax.scatter(
            place_cell_centres[:, 0],
            place_cell_centres[:, 1],
            c="C1",
            marker="x",
            s=15,
            zorder=2,
        )
        return fig, ax


class GridCells(Neurons):
    """The GridCells class defines a population of GridCells. This class is a subclass of Neurons() and inherits it properties/plotting functions.  

    Must be initialised with an Agent and a 'params' dictionary. 

    GridCells defines a set of 'n' grid cells with random orientations, grid scales and offsets (these can be set non-randomly of coursse). Grids are modelled as the rectified sum of three cosine waves at 60 degrees to each other. 

    List of functions: 
        • get_state()
    """

    def __init__(self, Agent, params={}):
        """Initialise GridCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}."""

        default_params = {
            "n": 10,
            "gridscale": 0.45,
            "random_orientations": True,
            "random_gridscales": True,
            "min_fr": 0,
            "max_fr": 1,
            "name": "GridCells",
        }
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        # Initialise grid cells
        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "grid cells only available in 2D"
        self.phase_offsets = np.random.uniform(0, self.gridscale, size=(self.n, 2))
        w = []
        for i in range(self.n):
            w1 = np.array([1, 0])
            if self.random_orientations == True:
                w1 = rotate(w1, np.random.uniform(0, 2 * np.pi))
            w2 = rotate(w1, np.pi / 3)
            w3 = rotate(w1, 2 * np.pi / 3)
            w.append(np.array([w1, w2, w3]))
        self.w = np.array(w)
        if self.random_gridscales == True:
            self.gridscales = np.random.uniform(
                2 * self.gridscale / 3, 1.5 * self.gridscale, size=self.n
            )
        if verbose is True:
            print(
                "GridCells successfully initialised. You can also manually set their gridscale (GridCells.gridscales), offsets (GridCells.phase_offset) and orientations (GridCells.w1, GridCells.w2,GridCells.w3 give the cosine vectors)"
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the grid cells.
        By default position is taken from the Agent and used to calculate firing rates. This can also by passed directly (evaluate_at=None, pos=pass_array_of_positions) or ou can use all the positions in the environment (evaluate_at="all").

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
        vecs = get_vectors_between(self.phase_offsets, pos)  # shape = (N_cells,N_pos,2)
        w1 = np.tile(np.expand_dims(self.w[:, 0, :], axis=1), reps=(1, pos.shape[0], 1))
        w2 = np.tile(np.expand_dims(self.w[:, 1, :], axis=1), reps=(1, pos.shape[0], 1))
        w3 = np.tile(np.expand_dims(self.w[:, 2, :], axis=1), reps=(1, pos.shape[0], 1))
        gridscales = np.tile(
            np.expand_dims(self.gridscales, axis=1), reps=(1, pos.shape[0])
        )
        phi_1 = ((2 * np.pi) / gridscales) * (vecs * w1).sum(axis=-1)
        phi_2 = ((2 * np.pi) / gridscales) * (vecs * w2).sum(axis=-1)
        phi_3 = ((2 * np.pi) / gridscales) * (vecs * w3).sum(axis=-1)
        firingrate = 0.5 * ((np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3)))
        firingrate[firingrate < 0] = 0

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate


class BoundaryVectorCells(Neurons):
    """The BoundaryVectorCells class defines a population of Boundary Vector Cells. This class is a subclass of Neurons() and inherits it properties/plotting functions.  

    Must be initialised with an Agent and a 'params' dictionary.  

    BoundaryVectorCells defines a set of 'n' BVCs cells with random orientations preferences, distance preferences  (these can be set non-randomly of course). We use the model described firstly by Hartley et al. (2000) and more recently de Cothi and Barry (2000).

    BVCs can have allocentric (mec,subiculum) OR egocentric (ppc, retrosplenial cortex) reference frames.

    List of functions: 
        • get_state()
        • boundary_vector_preference_function()
    """

    def __init__(self, Agent, params={}):
        """Initialise BoundaryVectorCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}."""

        default_params = {
            "n": 10,
            "reference_frame": "allocentric",
            "prefered_wall_distance_mean": 0.15,
            "angle_spread_degrees": 11.25,
            "xi": 0.08,  # as in de cothi and barry 2020
            "beta": 12,
            "min_fr": 0,
            "max_fr": 1,
            "name": "BoundaryVectorCells",
        }
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
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
        self.n_test_angles = 360
        self.dtheta = 2 * np.pi / self.n_test_angles
        for i in range(self.n_test_angles - 1):
            test_direction_ = rotate(test_direction, 2 * np.pi * i / 360)
            test_directions.append(test_direction_)
            test_angles.append(2 * np.pi * i / 360)
        self.test_directions = np.array(test_directions)
        self.test_angles = np.array(test_angles)
        self.sigma_angles = np.array(
            [(self.angle_spread_degrees / 360) * 2 * np.pi] * self.n
        )
        self.tuning_angles = np.random.uniform(0, 2 * np.pi, size=self.n)
        self.tuning_distances = np.random.rayleigh(
            scale=self.prefered_wall_distance_mean, size=self.n,
        )
        self.sigma_distances = self.tuning_distances / beta + xi

        # calculate normalising constants for BVS firing rates in the current environment. Any extra walls you add from here onwards you add will likely push the firingrate up further.
        locs = self.Agent.Environment.discretise_environment(dx=0.04)
        locs = locs.reshape(-1, locs.shape[-1])
        self.cell_fr_norm = np.ones(self.n)
        self.cell_fr_norm = np.max(self.get_state(evaluate_at=None, pos=locs), axis=1)

        if verbose is True:
            print(
                "BoundaryVectorCells (BVCs) successfully initialised. You can also manually set their orientation preferences (BVCs.tuning_angles, BVCs.sigma_angles), distance preferences (BVCs.tuning_distances, BVCs.sigma_distances)."
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Here we implement the same type if boundary vector cells as de Cothi et al. (2020), who follow Barry & Burgess, (2007). See equations there. 
    
        The way I do this is a little complex. I will describe how it works from a single position (but remember this can be called in a vectorised manner from an arary of positons in parallel)
            1. An array of normalised "test vectors" span, in all directions at 1 degree increments, from the position
            2. These define an array of line segments stretching from [pos, pos+test vector]
            3. Where these line segments collide with all walls in the environment is established, this uses the function "vector_intercepts()"
            4. This pays attention to only consider the first (closest) wall forawrd along a line segment. Walls behind other walls are "shaded" by closer walls. Its a little complex to do this and requires the function "boundary_vector_preference_function()"
            5. Now that, for every test direction, the closest wall is established it is simple a process of finding the response of the neuron to that wall at that angle (multiple of two gaussians, see de Cothi (2020)) and then summing over all the test angles. 
        
        We also apply a check in the middle to rotate teh reference frame into that of the head direction of the agent iff self.reference_frame='egocentric'.

        By default position is taken from the Agent and used to calculate firing rates. This can also by passed directly (evaluate_at=None, pos=pass_array_of_positions) or ou can use all the positions in the environment (evaluate_at="all").
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
        pos_lineseg_wall_intercepts = vector_intercepts(
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
                vel = self.Agent.pos
            else:
                vel = kwargs["vel"]
            vel = np.array(vel)
            head_direction_angle = get_angle(vel)
            test_angles = test_angles - head_direction_angle

        tuning_angles = np.tile(
            np.expand_dims(np.expand_dims(self.tuning_angles, axis=-1), axis=-1),
            reps=(1, N_pos, N_test),
        )  # (N_cell,N_pos,N_test)
        sigma_angles = np.tile(
            np.expand_dims(
                np.expand_dims(np.array(self.sigma_angles), axis=-1), axis=-1,
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

        g = gaussian(
            dist_to_first_wall, tuning_distances, sigma_distances, norm=1
        ) * von_mises(
            test_angles, tuning_angles, sigma_angles, norm=1
        )  # (N_cell,N_pos,N_test)

        firingrate = g.sum(axis=-1)  # (N_cell,N_pos)
        firingrate = firingrate / np.expand_dims(self.cell_fr_norm, axis=-1)
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def boundary_vector_preference_function(self, x):
        """This is a random function needed to efficiently produce boundary vector cells. x is any array of final dimension shape shape[-1]=2. As I use it here x has the form of the output of vector_intercepts. I.e. each point gives shape[-1]=2 lambda values (lam1,lam2) for where a pair of line segments intercept. This function gives a preference for each pair. Preference is -1 if lam1<0 (the collision occurs behind the first point) and if lam2>1 or lam2<0 (the collision occurs ahead of the first point but not on the second line segment). If neither of these are true it's 1/x (i.e. it prefers collisions which are closest).

        Args:
            x (array): shape=(any_shape...,2)

        Returns:
            the preferece values: shape=(any_shape)
        """
        assert x.shape[-1] == 2
        pref = np.piecewise(
            x=x,
            condlist=(x[..., 0] > 0, x[..., 0] < 0, x[..., 1] < 0, x[..., 1] > 1,),
            funclist=(1 / x[x[..., 0] > 0], -1, -1, -1,),
        )
        return pref[..., 0]

    def plot_BVC_receptive_field(self, chosen_neurons="all"):
        """Plots the receptive field (in polar corrdinates) of the BVC cells. For allocentric BVCs "up" in this plot == "North", for egocentric BVCs, up == the head direction of the animals

        Args:
            chosen_neurons: Which neurons to plot. Can be int, list, array or "all". Defaults to "all".

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

        fig, ax = plt.subplots(
            1,
            len(chosen_neurons),
            figsize=(3 * len(chosen_neurons), 3 * 1),
            subplot_kw={"projection": "polar"},
        )

        r = np.linspace(0, self.Agent.Environment.scale, 100)
        theta = np.linspace(0, 2 * np.pi, 360)
        [theta_meshgrid, r_meshgrid] = np.meshgrid(theta, r)

        def bvc_rf(theta, r, mu_r=0.5, sigma_r=0.2, mu_theta=0.5, sigma_theta=0.1):
            theta = pi_domain(theta)
            return gaussian(r, mu_r, sigma_r) * von_mises(theta, mu_theta, sigma_theta)

        for i, n in enumerate(chosen_neurons):
            mu_r = self.tuning_distances[n]
            sigma_r = self.sigma_angles[n]
            mu_theta = self.tuning_angles[n]
            sigma_theta = self.sigma_angles[n]
            receptive_field = bvc_rf(
                theta_meshgrid, r_meshgrid, mu_r, sigma_r, mu_theta, sigma_theta
            )
            ax[i].pcolormesh(
                theta, r, receptive_field, edgecolors="face", shading="nearest"
            )
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        return fig, ax


class VelocityCells(Neurons):
    """The VelocityCells class defines a population of Velocity cells. This class is a subclass of Neurons() and inherits it properties/plotting functions.  

    Must be initialised with an Agent and a 'params' dictionary. 

    VelocityCells defines a set of 'dim x 2' velocity cells. Encoding the East, West (and North and South) velocities in 1D (2D). The velocities are scaled according to the expected velocity of he agent (max firing rate acheive when velocity = mean + std)

    List of functions: 
        • get_state()
    """

    def __init__(self, Agent, params={}):
        """Initialise VelocityCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""
        default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "name": "VelocityCells",
        }
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        if self.Agent.Environment.dimensionality == "2D":
            self.n = 4  # one up, one down, one left, one right
        if self.Agent.Environment.dimensionality == "1D":
            self.n = 2  # one left, one right
        self.params["n"] = self.n
        self.one_sigma_speed = self.Agent.speed_mean + self.Agent.speed_std

        if verbose is True:
            print(
                f"VelocityCells successfully initialised. Your environment is {self.Agent.Environment.dimensionality} therefore you have {self.n} velocity cells"
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """In 2D 4 velocity cells report, respectively, the thresholded leftward, rightward, upward and downwards velocity. By default velocity is taken from the agent but this can also be passed as a kwarg 'vel'"""
        if evaluate_at == "agent":
            vel = self.Agent.history["vel"][-1]
        else:
            try:
                vel = np.array(kwargs["vel"])
            except KeyError:
                vel = np.zeros_like(self.Agent.velocity)

        if self.Agent.Environment.dimensionality == "1D":
            vleft_fr = max(0, vel[0]) / self.one_sigma_speed
            vright_fr = max(0, -vel[0]) / self.one_sigma_speed
            firingrate = np.array([vleft_fr, vright_fr])
        if self.Agent.Environment.dimensionality == "2D":
            vleft_fr = max(0, vel[0]) / self.one_sigma_speed
            vright_fr = max(0, -vel[0]) / self.one_sigma_speed
            vup_fr = max(0, vel[1]) / self.one_sigma_speed
            vdown_fr = max(0, -vel[1]) / self.one_sigma_speed
            firingrate = np.array([vleft_fr, vright_fr, vup_fr, vdown_fr])

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]

        return firingrate


class HeadDirectionCells(Neurons):
    """The HeadDirectionCells class defines a population of head direction cells. This class is a subclass of Neurons() and inherits it properties/plotting functions.  

    Must be initialised with an Agent and a 'params' dictionary. 

    HeadDirectionCells defines a set of 'dim x 2' velocity cells. Encoding the East, West (and North and South) heading directions in 1D (2D). The firing rates are scaled such that when agent travels due east/west/north,south the firing rate is  = [mfr,0,0,0]/[0,mfr,0,0]/[0,0,mfr,0]/[0,0,0,mfr] (mfr = max_fr)

    List of functions: 
        • get_state()
        """

    def __init__(self, Agent, params={}):
        """Initialise HeadDirectionCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""
        default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "name": "HeadDirectionCells",
        }
        self.Agent = Agent
        for key in params.keys():
            default_params[key] = params[key]
        self.params = default_params
        super().__init__(Agent, self.params)

        if self.Agent.Environment.dimensionality == "2D":
            self.n = 4  # one up, one down, one left, one right
        if self.Agent.Environment.dimensionality == "1D":
            self.n = 2  # one left, one right
        if verbose is True:
            print(
                f"HeadDirectionCells successfully initialised. Your environment is {self.Agent.Environment.dimensionality} therefore you have {self.n} head direction cells"
            )

    def get_state(self, evaluate_at="agent", **kwargs):
        """In 2D 4 head direction cells report the head direction of the animal. For example a population vector of [1,0,0,0] implies due-east motion. By default velocity (which determines head direction) is taken from the agent but this can also be passed as a kwarg 'vel'"""

        if evaluate_at == "agent":
            vel = self.Agent.history["vel"][-1]
        else:
            vel = np.array(kwargs["vel"])

        if self.Agent.Environment.dimensionality == "1D":
            hdleft_fr = max(0, np.sign(vel[0]))
            hdright_fr = max(0, -np.sign(vel[0]))
            firingrate = np.array([hdleft_fr, hdright_fr])
        if self.Agent.Environment.dimensionality == "2D":
            vel = vel / np.linalg.norm(vel)
            hdleft_fr = max(0, vel[0])
            hdright_fr = max(0, -vel[0])
            hdup_fr = max(0, vel[1])
            hddown_fr = max(0, -vel[1])
            firingrate = np.array([hdleft_fr, hdright_fr, hdup_fr, hddown_fr])

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]

        return firingrate


class SpeedCell(Neurons):
    """The SpeedCell class defines a single speed cell. This class is a subclass of Neurons() and inherits it properties/plotting functions.  

    Must be initialised with an Agent and a 'params' dictionary. 

    The firing rate is scaled according to the expected velocity of the agent (max firing rate acheive when velocity = mean + std)

    List of functions: 
        • get_state()
        """

    def __init__(self, Agent, params={}):
        """Initialise SpeedCell(), takes as input a parameter dictionary, 'params'. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""
        default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "name": "SpeedCell",
        }
        self.Agent = Agent
        for key in params.keys():
            default_params[key] = params[key]
        self.params = default_params
        super().__init__(Agent, self.params)
        self.n = 1
        self.one_sigma_speed = self.Agent.speed_mean + self.Agent.speed_std

        if verbose is True:
            print(
                f"SpeedCell successfully initialised. The speed of the agent is encoded linearly by the firing rate of this cell. Speed = 0 --> min firing rate of of {self.min_fr}. Speed = mean + 1std (here {self.one_sigma_speed} --> max firing rate of {self.max_fr}."
            )

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the speed cell. By default velocity (which determines speed) is taken from the agent but this can also be passed as a kwarg 'vel'

        Args:
            evaluate_at (str, optional): _description_. Defaults to 'agent'.

        Returns:
            firingrate: np.array([firingrate])
        """
        if evaluate_at == "agent":
            vel = self.Agent.history["vel"][-1]
        else:
            vel = np.array(kwargs["vel"])

        speed = np.linalg.norm(vel)
        firingrate = np.array([speed / self.one_sigma_speed])
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate


class FeedForwardLayer(Neurons):
    """The FeedForwardLayer class defines a layer of Neurons() whos firing rates are an activated linear combination of downstream input layers. This class is a subclass of Neurons() and inherits it properties/plotting functions.  

    *** Understanding this layer is crucial if you want to build multilayer networks of Neurons with RatInABox ***

    Must be initialised with an Agent and a 'params' dictionary. 
    Input params dictionary should contain a list of input_layers which feed into this layer. This list looks like [Input1, Input2,...] where each is a Neurons() class (typically this list will but you can have arbitrariy many layers feed into this one). You can also add inputs one-by-one using self.add_input()

    Each layer which feeds into this one is assigned a set of weights fr = activation_function(sum_over_layers(sum_over_inputs_in_this_layer(w_i I_i))) (you my be interested in accessing these weights in order to write a function which "learns" them, for example). A dictionary stores all the inputs, the key for each input layer is its name (e.g Input1.name = "Input1"), so to get the weights call 
        FeedForwardLayer.inputs["Input1"]['w'] --> returns the weight matrix from Input1 to FFL
        FeedForwardLayer.inputs["Input2"]['w'] --> returns the weight matrix from Input1 to FFL
        ...
    
    Currently supported activations include 'sigmoid' (paramterised by max_fr, min_fr, mid_x, width), 'relu' (gain, threshold) and 'linear' specified with the "activation_params" dictionary in the inout params dictionary. See also utils.activate() for full details. 

    Check that the input layers are all named differently. 
    List of functions: 
        • get_state()
        • add_input()
        """

    def __init__(self, Agent, params={}):
        default_params = {
            "n": 10,
            "input_layers": [],  # a list of input layers, or add one by one using self.adD_inout
            "activation_params": {
                "activation": "sigmoid",
                "max_fr": 1,
                "min_fr": 0,
                "mid_x": 1,
                "width_x": 2,
            },
            "name": "FeedForwardLayer",
        }
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        self.inputs = {}
        for input_layer in self.input_layers:
            self.add_input(input_layer)

        if verbose is True:
            print(
                f"FeedForwardLayer initialised with {len(self.inputs.keys())} layers ({self.inputs.keys()}). To add another layer use FeedForwardLayer.add_input_layer().\nTo set the weights manual edit them by changing self.inputs['key']['W']"
            )

    def add_input(self, input_layer, w_init_scale=1, **kwargs):
        """Adds an input layer to the class. Each input layer is stored in a dictionary of self.inputs. Each has an associated matrix of weights which are initialised randomly. 

        Note the inputs are stored in a dictionary. The keys are taken to be the name of each layer passed (input_layer.name). Make sure you set this correctly (and uniquely). 

        Args:
            • input_layer (_type_): the layer intself. Must be a Neurons() class object (e.g. can be PlaceCells(), etc...). 
            • w_init_scale: initial weights drawn from zero-centred gaussian with std w_init_scale / sqrt(N_in)
            • **kwargs any extra kwargs will get saved into the inputs dictionary in case you need these

        """
        n = input_layer.n
        name = input_layer.name
        w = np.random.normal(loc=0, scale=w_init_scale / np.sqrt(n), size=(self.n, n))
        I = np.zeros(n)
        if name in self.inputs.keys():
            print(
                f"There already exists a layer called {input_layer_name}. Overwriting it now."
            )
        self.inputs[name] = {}
        self.inputs[name]["layer"] = input_layer
        self.inputs[name]["w"] = w
        self.inputs[name]["w_init"] = w.copy()
        self.inputs[name]["I"] = I
        for (key, value) in kwargs.items():
            self.inputs[name][key] = value

    def get_state(self, evaluate_at="last", **kwargs):
        """Returns the firing rate of the feedforward layer cells. By default this layer uses the last saved firingrate from its input layers. Alternatively evaluate_at and kwargs can be set to be anything else which will just be passed to the input layer for evaluation. 
        Once the firing rate of the inout layers is established these are multiplied by the weight matrices and then activated to obtain the firing rate of this FeedForwardLayer.

        Args:
            evaluate_at (str, optional). Defaults to 'last'.
        Returns:
            firingrate: array of firing rates 
        """

        if evaluate_at == "last":
            V = np.zeros(self.n)
        elif evaluate_at == "all":
            V = np.zeros(
                (self.n, self.Agent.Environment.flattened_discrete_coords.shape[0])
            )
        else:
            V = np.zeros((self.n, kwargs["pos"].shape[0]))

        for inputlayer in self.inputs.values():
            w = inputlayer["w"]
            if evaluate_at == "last":
                I = inputlayer["layer"].firingrate
            else:  # kick can down the road let input layer decide how to evaluate the firingrate
                I = inputlayer["layer"].get_state(evaluate_at, **kwargs)
            inputlayer["I_temp"] = I
            V += np.matmul(w, I)
        firingrate = activate(V, other_args=self.activation_params)
        firingrate_prime = activate(V, other_args=self.activation_params, deriv=True)

        self.firingrate_temp = firingrate
        self.firingrate_prime_temp = firingrate_prime

        return firingrate

