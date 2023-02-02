import ratinabox 

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy import stats as stats

from ratinabox import utils

"""NEURONS"""
"""Parent Class"""


class Neurons:
    """The Neuron class defines a population of Neurons. All Neurons have firing rates which depend on the state of the Agent. As the Agent moves the firing rate of the cells adjust accordingly.

    All Neuron classes must be initalised with the Agent (to whom these cells belong) since the Agent determines the firingrates through its position and velocity. The Agent class will itself contain the Environment. Both the Agent (position/velocity) and the Environment (geometry, walls etc.) determine the firing rates. Optionally (but likely) an input dictionary 'params' specifying other params will be given.

    This is a generic Parent class. We provide several SubClasses of it. These include:
    • PlaceCells()
    • GridCells()
    • BoundaryVectorCells()
    • VelocityCells()
    • HeadDirectionCells()
    • SpeedCells()
    • FeedForwardLayer()

    The unique function in each child classes is get_state(). Whenever Neurons.update() is called Neurons.get_state() is then called to calculate and returns the firing rate of the cells at the current moment in time. This is then saved. In order to make your own Neuron subclass you will need to write a class with the following mandatory structure:

    ============================================================================================
    MyNeuronClass(Neurons):
        def __init__(self,
                     Agent,
                     params={}): #<-- do not change these

            default_params = {'a_default_param":3.14159} #note this params dictionary is passed upwards and used in all the parents classes of your class.

            default_params.update(params)
            self.params = default_params
            super().__init__(Agent,self.params)

        def get_state(self,
                      evaluate_at='agent',
                      **kwargs) #<-- do not change these

            firingrate = .....
            ###
                Insert here code which calculates the firing rate.
                This may work differently depending on what you set evaluate_at as. For example, evaluate_at == 'agent' should means that the position or velocity (or whatever determines the firing rate) will by evaluated using the agents current state. You might also like to have an option like evaluate_at == "all" (all positions across an environment are tested simultaneously - plot_rate_map() tries to call this, for example) or evaluate_at == "last" (in a feedforward layer just look at the last firing rate saved in the input layers saves time over recalculating them.). **kwargs allows you to pass position or velocity in manually.

                By default, the Neurons.update() calls Neurons.get_state() rwithout passing any arguments. So write the default behaviour of get_state() to be what you want it to do in the main training loop.
            ###

            return firingrate

        def any_other_functions_you_might_want(self):...
    ============================================================================================

    As we have written them, Neuron subclasses which have well defined ground truth spatial receptive fields (PlaceCells, GridCells but not VelocityCells etc.) can also be queried for any arbitrary pos/velocity (i.e. not just the Agents current state) by passing these in directly to the function "get_state(evaluate_at='all') or get_state(evaluate_at=None, pos=my_array_of_positons)". This calculation is vectorised and relatively fast, returning an array of firing rates one for each position. It is what is used when you try Neuron.plot_rate_map().

    List of key functions...
        ..that you're likely to use:
            • update()
            • plot_rate_timeseries()
            • plot_rate_map()
        ...that you might not use but could be useful:
            • save_to_history()
            • boundary_vector_preference_function()

    default_params = {
            "n": 10,
            "name": "Neurons",
            "color": None,  # just for plotting
        }
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

            "noise_std":0, #0 means no noise, std of the noise you want to add (Hz) 
            "noise_coherence_time":0.5,
        }
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        utils.update_class_params(self, self.params)

        self.firingrate = np.zeros(self.n)
        self.noise = np.zeros(self.n)

        self.history = {}
        self.history["t"] = []
        self.history["firingrate"] = []
        self.history["spikes"] = []


        if ratinabox.verbose is True:
            print(
                f"\nA Neurons() class has been initialised with parameters f{self.params}. Use Neurons.update() to update the firing rate of the Neurons to correspond with the Agent.Firing rates and spikes are saved into the Agent.history dictionary. Plot a timeseries of the rate using Neurons.plot_rate_timeseries(). Plot a rate map of the Neurons using Neurons.plot_rate_map()."
            )

    def update(self):
        #update noise vector
        dnoise = utils.ornstein_uhlenbeck(dt=self.Agent.dt,
                                          x = self.noise,
                                          drift=0,
                                          noise_scale = self.noise_std,
                                          coherence_time = self.noise_coherence_time)
        self.noise = self.noise + dnoise 

        #update firing rate 
        firingrate = self.get_state()
        self.firingrate = firingrate.reshape(-1)
        self.firingrate = self.firingrate + self.noise 
        self.save_to_history()
        return

    def plot_rate_timeseries(
        self,
        t_start=None,
        t_end=None,
        chosen_neurons="all",
        spikes=True,
        imshow=False,
        fig=None,
        ax=None,
        xlim=None,
        background_color=None,
        **kwargs,
    ):
        """Plots a timeseries of the firing rate of the neurons between t_start and t_end

        Args:
            • t_start (int, optional): _description_. Defaults to start of data, probably 0.
            • t_end (int, optional): _description_. Defaults to end of data.
            • chosen_neurons: Which neurons to plot. string "10" or 10 will plot ten of them, "all" will plot all of them, "12rand" will plot 12 random ones. A list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "all".
            chosen_neurons (str, optional): Which neurons to plot. string "10" will plot 10 of them, "all" will plot all of them, a list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "10".
            • spikes (bool, optional): If True, scatters exact spike times underneath each curve of firing rate. Defaults to True.
            the below params I just added for help with animations
            • imshow - if True will not dispaly as mountain plot but as an image (plt.imshow)
            • fig, ax: the figure, axis to plot on (can be None)
            xlim: fix xlim of plot irrespective of how much time you're plotting
            • background_color: color of the background if not matplotlib default (probably white)
            • kwargs sent to mountain plot function, you can ignore these

        Returns:
            fig, ax
        """
        t = np.array(self.history["t"])
        # times to plot
        if t_start is None:
            t_start = t[0]
        if t_end is None:
            t_end = t[-1]
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end)))
        rate_timeseries = np.array(self.history["firingrate"][startid:endid])
        spike_data = np.array(self.history["spikes"][startid:endid])
        t = t[startid:endid]

        # neurons to plot
        chosen_neurons = self.return_list_of_neurons(chosen_neurons)
        spike_data = spike_data[startid:endid,chosen_neurons]
        rate_timeseries = rate_timeseries[:,chosen_neurons]

        if imshow == False:
            firingrates = rate_timeseries.T
            fig, ax = utils.mountain_plot(
                X=t / 60,
                NbyX=firingrates,
                color=self.color,
                xlabel="Time / min",
                ylabel="Neurons",
                xlim=None,
                fig=fig,
                ax=ax,
                **kwargs,
            )

            if spikes == True:
                for i in range(len(chosen_neurons)):
                    time_when_spiked = t[spike_data[:, i]] / 60
                    h = (i + 1 - 0.1) * np.ones_like(time_when_spiked)
                    ax.scatter(
                        time_when_spiked,
                        h,
                        color=(self.color or "C1"),
                        alpha=0.5,
                        s=5,
                        linewidth=0,
                    )
            ax.set_xlim(left=t_start / 60, right=t_end / 60)
            ax.set_xticks([t_start / 60, t_end / 60])
            ax.set_xticklabels([round(t_start / 60, 2), round(t_end / 60, 2)])
            if xlim is not None:
                ax.set_xlim(right=xlim / 60)
                ax.set_xticks([round(t_start / 60, 2), round(xlim / 60, 2)])
                ax.set_xticklabels([round(t_start / 60, 2), round(xlim / 60, 2)])

            if background_color is not None:
                ax.set_facecolor(background_color)
                fig.patch.set_facecolor(background_color)

        elif imshow == True:
            if fig is None and ax is None:
                fig, ax = plt.subplots(figsize=(8, 4))
            data = rate_timeseries.T
            ax.imshow(data[::-1], aspect=0.3 * data.shape[1] / data.shape[0])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel("Time / min")
            ax.set_xticks([0 - 0.5, len(t) + 0.5])
            ax.set_xticklabels([round(t_start / 60, 2), round(t_end / 60, 2)])
            ax.set_yticks([])
            ax.set_ylabel("Neurons")

        return fig, ax

    def plot_rate_map(
        self,
        chosen_neurons="all",
        method="groundtruth",
        spikes=False,
        fig=None,
        ax=None,
        shape=None,
        colorbar=True,
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
            • colorbar: whether to show a colorbar
            • t_start, t_end: in the case where you are plotting spike, or using historical data to get rate map, this restricts the timerange of data you are using
            • kwargs are sent to get_state and utils.mountain_plot and can be ignore if you don't need to use them

        Returns:
            fig, ax
        """
        # GET DATA
        if method == "groundtruth":
            try:
                rate_maps = self.get_state(evaluate_at="all", **kwargs)
            except Exception as e:
                print(
                    "It was not possible to get the rate map by evaluating the firing rate at all positions across the Environment. This is probably because the Neuron class does not support, or it does not have an groundtruth receptive field. Instead, plotting rate map by weighted position histogram method. Here is the error:"
                )
                print("Error: ", e)
                import traceback

                traceback.print_exc()
                method = "history"

        if method == "history" or spikes == True:
            t = np.array(self.history["t"])
            # times to plot
            if len(t) == 0:
                print(
                    "Can't plot rate map by method='history' since there is no available data to plot. "
                )
                return
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

        # PLOT 2D
        if self.Agent.Environment.dimensionality == "2D":
            from mpl_toolkits.axes_grid1 import ImageGrid

            if fig is None and ax is None:
                if shape is None:
                    Nx, Ny = 1, len(chosen_neurons)
                else:
                    Nx, Ny = shape[0], shape[1]
                fig = plt.figure(figsize=(2 * Ny, 2 * Nx))
                if colorbar == True and (method in ["groundtruth", "history"]):
                    cbar_mode = "single"
                else:
                    cbar_mode = None
                axes = ImageGrid(
                    fig,
                    # (0, 0, 3, 3),
                    111,
                    nrows_ncols=(Nx, Ny),
                    axes_pad=0.05,
                    cbar_location="right",
                    cbar_mode=cbar_mode,
                    cbar_size="5%",
                    cbar_pad=0.05,
                )
                if colorbar == True:
                    cax = axes.cbar_axes[0]
                axes = np.array(axes)
            else:
                axes = np.array([ax]).reshape(-1)
                if method in ["groundtruth", "history"]:
                    if colorbar == True:
                        from mpl_toolkits.axes_grid1 import make_axes_locatable

                        divider = make_axes_locatable(axes[-1])
                        cax = divider.append_axes("right", size="5%", pad=0.05)
            for (i, ax_) in enumerate(axes):
                _, ax_ = self.Agent.Environment.plot_environment(fig, ax_)
            if len(chosen_neurons) != axes.size:
                print(
                    f"You are trying to plot a different number of neurons {len(chosen_neurons)} than the number of axes provided {axes.size}. Some might be missed. Either change this with the chosen_neurons argument or pass in a list of axes to plot on"
                )

            vmin, vmax = 0, 0
            ims = []
            if method in ["groundtruth", "history"]:
                for (i, ax_) in enumerate(axes):
                    ex = self.Agent.Environment.extent
                    if method == "groundtruth":
                        rate_map = rate_maps[chosen_neurons[i], :].reshape(
                            self.Agent.Environment.discrete_coords.shape[:2]
                        )
                        im = ax_.imshow(rate_map, extent=ex)
                    elif method == "history":
                        rate_timeseries_ = rate_timeseries[chosen_neurons[i], :]
                        rate_map = utils.bin_data_for_histogramming(
                            data=pos, extent=ex, dx=0.05, weights=rate_timeseries_
                        )
                        im = ax_.imshow(
                            rate_map,
                            extent=self.Agent.Environment.extent,
                            interpolation="bicubic",
                        )
                    ims.append(im)
                    vmin, vmax = (
                        min(vmin, np.min(rate_map)),
                        max(vmax, np.max(rate_map)),
                    )
                for im in ims:
                    im.set_clim((vmin, vmax))
                if colorbar == True:
                    cbar = plt.colorbar(ims[-1], cax=cax)
                    lim_v = vmax if vmax > -vmin else vmin
                    cbar.set_ticks([0, lim_v])
                    cbar.set_ticklabels([0, round(lim_v, 1)])
                    cbar.outline.set_visible(False)

            if spikes is True:
                for (i, ax_) in enumerate(axes):
                    pos_where_spiked = pos[spike_data[chosen_neurons[i], :]]
                    ax_.scatter(
                        pos_where_spiked[:, 0],
                        pos_where_spiked[:, 1],
                        s=2,
                        linewidth=0,
                        alpha=0.7,
                    )

            return fig, axes

        # PLOT 1D
        if self.Agent.Environment.dimensionality == "1D":
            if method == "groundtruth":
                rate_maps = rate_maps[chosen_neurons, :]
                x = self.Agent.Environment.flattened_discrete_coords[:, 0]
            if method == "history":
                ex = self.Agent.Environment.extent
                pos_ = pos[:, 0]
                rate_maps = []
                for neuron_id in chosen_neurons:
                    rate_map, x = utils.bin_data_for_histogramming(
                        data=pos_,
                        extent=ex,
                        dx=0.01,
                        weights=rate_timeseries[neuron_id, :],
                    )
                    x, rate_map = utils.interpolate_and_smooth(x, rate_map, sigma=0.03)
                    rate_maps.append(rate_map)
                rate_maps = np.array(rate_maps)

            if fig is None and ax is None:
                fig, ax = self.Agent.Environment.plot_environment(
                    height=0.5 * len(chosen_neurons)
                )

            if method != "neither":
                fig, ax = utils.mountain_plot(
                    X=x, NbyX=rate_maps, color=self.color, fig=fig, ax=ax, **kwargs
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

    def animate_rate_timeseries(
        self,
        t_start=None,
        t_end=None,
        chosen_neurons="all",
        fps=15,
        speed_up=1,
        **kwargs,
    ):
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
        dt = 1 / fps
        if t_start == None:
            t_start = self.history["t"][0]
        if t_end == None:
            t_end = self.history["t"][-1]

        def animate_(i, fig, ax, chosen_neurons, t_start, t_max, dt, speed_up):
            t = self.history["t"]
            # t_start = t[0]
            # t_end = t[0] + (i + 1) * speed_up * dt
            t_end = t_start + (i + 1) * speed_up * dt
            ax.clear()
            fig, ax = self.plot_rate_timeseries(
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                fig=fig,
                ax=ax,
                xlim=t_max,
                **kwargs,
            )
            plt.close()
            return

        fig, ax = self.plot_rate_timeseries(
            t_start=0,
            t_end=10 * self.Agent.dt,
            chosen_neurons=chosen_neurons,
            xlim=t_end,
            **kwargs,
        )

        from matplotlib import animation
        anim = matplotlib.animation.FuncAnimation(
            fig,
            animate_,
            interval=1000 * dt,
            frames=int((t_end - t_start) / (dt * speed_up)),
            blit=False,
            fargs=(fig, ax, chosen_neurons, t_start, t_end, dt, speed_up),
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
            elif chosen_neurons[-4:] == "rand":
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

    def __init__(self, Agent, params={}):
        """Initialise PlaceCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.
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
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        if self.place_cell_centres is None:
            self.place_cell_centres = self.Agent.Environment.sample_positions(
                n=self.n, method="uniform_jitter"
            )
        elif type(self.place_cell_centres) is str: 
            if self.place_cell_centres in ["random", "uniform", "uniform_jitter"]:
                self.place_cell_centres = self.Agent.Environment.sample_positions(
                    n=self.n, method=self.place_cell_centres
                )
        else:
            self.n = self.place_cell_centres.shape[0]
        self.place_cell_widths = self.widths * np.ones(self.n)

        # Assertions (some combinations of boundary condition and wall geometries aren't allowed)
        if self.Agent.Environment.dimensionality == "2D":
            if all([
                ((self.wall_geometry == "line_of_sight") or ((self.wall_geometry == "geodesic"))),
                (self.Agent.Environment.boundary_conditions == "periodic"),
                (self.Agent.Environment.dimensionality == "2D")
            ]):
                print(
                    f"{self.wall_geometry} wall geometry only possible in 2D when the boundary conditions are solid. Using 'euclidean' instead."
                )
                self.wall_geometry = "euclidean"
            if (self.wall_geometry == "geodesic") and (
                len(self.Agent.Environment.walls) > 5
            ):
                print(
                    "'geodesic' wall geometry only supported for enivoronments with 1 additional wall (4 boundaing walls + 1 additional). Sorry. Using 'line_of_sight' instead."
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

    def plot_place_cell_locations(self, fig=None, ax=None):
        """Scatter plots where the centre of the place cells are

        Args:
            fig, ax: if provided, will plot fig and ax onto these instead of making new.

        Returns:
            _type_: _description_
        """
        if fig is None and ax is None:
            fig, ax = self.Agent.Environment.plot_environment()
        else:
            _, _ = self.Agent.Environment.plot_environment(fig=fig, ax=ax)
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

    default_params = {
            "n": 10,
            "gridscale": 0.45,
            "random_orientations": True,
            "random_gridscales": True,
            "random_phase_offsets": True,
            "min_fr": 0,
            "max_fr": 1,
            "name": "GridCells",
        }
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
            "random_phase_offsets": True,
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
        if self.random_phase_offsets == True:
            self.phase_offsets = np.random.uniform(0, self.gridscale, size=(self.n, 2))
        else:
            self.phase_offsets = self.set_phase_offsets()
        w = []
        for i in range(self.n):
            w1 = np.array([1, 0])
            if self.random_orientations == True:
                w1 = utils.rotate(w1, np.random.uniform(0, 2 * np.pi))
            w2 = utils.rotate(w1, np.pi / 3)
            w3 = utils.rotate(w1, 2 * np.pi / 3)
            w.append(np.array([w1, w2, w3]))
        self.w = np.array(w)
        if self.random_gridscales == True:
            self.gridscales = np.random.rayleigh(scale=self.gridscale, size=self.n)
        else:
            self.gridscales = np.full(self.n, fill_value=self.gridscale)
        if ratinabox.verbose is True:
            print(
                "GridCells successfully initialised. You can also manually set their gridscale (GridCells.gridscales), offsets (GridCells.phase_offset) and orientations (GridCells.w1, GridCells.w2,GridCells.w3 give the cosine vectors)"
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
        vecs = utils.get_vectors_between(
            self.phase_offsets, pos
        )  # shape = (N_cells,N_pos,2)
        w1 = np.tile(np.expand_dims(self.w[:, 0, :], axis=1), reps=(1, pos.shape[0], 1))
        w2 = np.tile(np.expand_dims(self.w[:, 1, :], axis=1), reps=(1, pos.shape[0], 1))
        w3 = np.tile(np.expand_dims(self.w[:, 2, :], axis=1), reps=(1, pos.shape[0], 1))
        gridscales = np.tile(
            np.expand_dims(self.gridscales, axis=1), reps=(1, pos.shape[0])
        )
        phi_1 = ((2 * np.pi) / gridscales) * (vecs * w1).sum(axis=-1)
        phi_2 = ((2 * np.pi) / gridscales) * (vecs * w2).sum(axis=-1)
        phi_3 = ((2 * np.pi) / gridscales) * (vecs * w3).sum(axis=-1)
        firingrate = (1 / 3) * ((np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3)))
        firingrate[firingrate < 0] = 0

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def set_phase_offsets(self):
        """Set non-random phase_offsets. Most offsets (cell number: x*y) are grid-like, while the remainings (cell number: n - x*y) are random."""
        n_x = int(np.sqrt(self.n))
        n_y = self.n // n_x
        n_remaining = self.n - n_x * n_y

        dx = self.gridscale / n_x
        dy = self.gridscale / n_y

        grid = np.mgrid[
            (0 + dx / 2): (self.gridscale - dx / 2): (n_x * 1j),
            (0 + dy / 2): (self.gridscale - dy / 2): (n_y * 1j),
        ]
        grid = grid.reshape(2, -1).T
        remaining = np.random.uniform(0, self.gridscale, size=(n_remaining, 2))

        all_offsets = np.vstack([grid, remaining])

        return all_offsets


class BoundaryVectorCells(Neurons):
    """The BoundaryVectorCells class defines a population of Boundary Vector Cells. This class is a subclass of Neurons() and inherits it properties/plotting functions.

    Must be initialised with an Agent and a 'params' dictionary.

    BoundaryVectorCells defines a set of 'n' BVCs cells with random orientations preferences, distance preferences  (these can be set non-randomly of course). We use the model described firstly by Hartley et al. (2000) and more recently de Cothi and Barry (2000).

    BVCs can have allocentric (mec,subiculum) OR egocentric (ppc, retrosplenial cortex) reference frames.

    List of functions:
        • get_state()
        • boundary_vector_preference_function()

    default_params = {
            "n": 10,
            "reference_frame": "allocentric",
            "pref_wall_dist": 0.15,
            "angle_spread_degrees": 11.25,
            "xi": 0.08,  # as in de cothi and barry 2020
            "beta": 12,
            "dtheta":2, #angular resolution in degrees
            "min_fr": 0,
            "max_fr": 1,
            "name": "BoundaryVectorCells",
        }
    """

    def __init__(self, Agent, params={}):
        """Initialise BoundaryVectorCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}."""

        default_params = {
            "n": 10,
            "reference_frame": "allocentric",
            "pref_wall_dist": 0.15,
            "angle_spread_degrees": 11.25,
            "xi": 0.08,
            "beta": 12,
            "dtheta": 2,
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
        self.tuning_distances = np.random.rayleigh(
            scale=self.pref_wall_dist, size=self.n,
        )
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
            5. Now that, for every test direction, the closest wall is established it is simple a process of finding the response of the neuron to that wall at that angle (multiple of two gaussians, see de Cothi (2020)) and then summing over all the test angles.

        We also apply a check in the middle to utils.rotate the reference frame into that of the head direction of the agent iff self.reference_frame='egocentric'.

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
                vel = self.Agent.pos
            elif 'vel' in kwargs.keys():
                vel = kwargs["vel"]
            else: 
                vel = np.array([1,0])
            vel = np.array(vel)
            head_direction_angle = utils.get_angle(vel)
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
            condlist=(x[..., 0] > 0, x[..., 0] < 0, x[..., 1] < 0, x[..., 1] > 1,),
            funclist=(1 / x[x[..., 0] > 0], -1, -1, -1,),
        )
        return pref[..., 0]

    def plot_BVC_receptive_field(self, chosen_neurons="all", fig=None, ax=None):
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

        return fig, ax


class ObjectVectorCells(Neurons):
    """Initialises ObjectVectorCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Each object vector cell has a preferred tuning_distance and tuning_angle. Only when the angle is (with gaussian spread) close to this distance and angle away from the OVC wll the cell fire. 

        It is possible for these cells to be "field_of_view" in which case the cell fires iff the agent is looking towards it. Essentially this is an egocentric OVC with tuning angle set to zero (head on). 

        default_params = {
            "n": 10,
            "min_fr": 0,
            "max_fr": 1,
            "name": "ObjectVectorCell",
            "walls_occlude":True, #whether walls occuled OVC firing
            "field_of_view":False, #set to true for "field of view" OVC
            "object_locations":None, #otherwise random across Env, the length of this will overwrite "n" 
            "angle_spread_degrees":15, #can be an array, one for each object, spread of von Mises angular preferrence functinon for each OVC
            "pref_object_dist": 0.25, #can be an array, one for each object, otherwise randomly drawn from a Rayleigh with this sigma. How far away from OVC the OVC fires. 
            "xi": 0.08, #parameters determining the distance preferrence function std given the preferred distance. See BoundaryVectorCells or de cothi and barry 2020
            "beta": 12,
        }
    """
    def __init__(self, Agent, params={}):

        default_params = {
            "n": 10,
            "min_fr": 0,
            "max_fr": 1,
            "name": "ObjectVectorCell",
            "walls_occlude":True, 
            "field_of_view":False,
            "object_locations":None, 
            "angle_spread_degrees":15, 
            "pref_object_dist":0.25, 
            "xi": 0.08,
            "beta": 12,
        }

        self.Agent = Agent
        default_params.update(params)
        self.params = default_params

        assert (self.Agent.Environment.dimensionality == "2D"), "object vector cells only possible in 2D"
        
        if self.params['object_locations'] is None: 
            self.object_locations = self.Agent.Environment.sample_positions(self.n)
            print(f"No object locations passed so {self.n} object locations have been randomly sampled across the environment")
        else: self.params['n'] = len(params['object_locations'])

        super().__init__(Agent, self.params)

        
        #preferred distance and angle to objects and their tuning widths (set these yourself if needed)
        self.tuning_angles = np.random.uniform(0, 2 * np.pi, size=self.n)
        self.tuning_distances = np.random.rayleigh(scale=self.pref_object_dist, size=self.n)
        self.sigma_distances = self.tuning_distances / self.beta + self.xi
        self.sigma_angles = np.array([(self.angle_spread_degrees / 360) * 2 * np.pi] * self.n)

        if self.field_of_view == True:
            self.tuning_angles = np.zeros(self.n)

        if self.walls_occlude == True: self.wall_geometry = 'line_of_sight'
        else: self.wall_geometry = 'euclidean'

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
        pos = pos.reshape(-1, pos.shape[-1]) #(N_pos, 2)
        N_pos = pos.shape[0]
        N_cells = self.n

        
        (distances_to_OVCs, vectors_to_OVCs) = self.Agent.Environment.get_distances_between___accounting_for_environment(pos,self.object_locations,return_vectors=True,wall_geometry=self.wall_geometry,) #(N_pos,N_cells) (N_pos,N_cells,2)
        flattened_vectors_to_OVCs = vectors_to_OVCs.reshape(-1,2) #(N_pos x N_cells, 2)
        bearings_to_OVCs = utils.get_angle(flattened_vectors_to_OVCs,is_array=True).reshape(N_pos,N_cells) #(N_cells,N_pos)
        if self.field_of_view == True: 
            if evaluate_at == "agent":
                vel = self.Agent.velocity
            elif 'vel' in kwargs.keys():
                vel = kwargs["vel"]
            else:
                vel = np.array([1,0])
                print("Field of view OVCs require a velocity vector but none was passed. Using [1,0]")
            head_bearing = utils.get_angle(vel)
            bearings_to_OVCs -= head_bearing

        tuning_distances = np.tile(np.expand_dims(self.tuning_distances,axis=0),reps=(N_pos,1)) #(N_pos,N_cell)
        sigma_distances = np.tile(np.expand_dims(self.sigma_distances,axis=0),reps=(N_pos,1)) #(N_pos,N_cell)
        tuning_angles = np.tile(np.expand_dims(self.tuning_angles,axis=0),reps=(N_pos,1)) #(N_pos,N_cell)
        sigma_angles = np.tile(np.expand_dims(self.sigma_angles,axis=0),reps=(N_pos,1)) #(N_pos,N_cell)

        firingrate = (utils.gaussian(
            distances_to_OVCs, tuning_distances, sigma_distances, norm=1
        ) * utils.von_mises(
            bearings_to_OVCs, tuning_angles, sigma_angles, norm=1
        )).T #(N_cell,N_pos)
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate




class HeadDirectionCells(Neurons):
    """The HeadDirectionCells class defines a population of head direction cells. This class is a subclass of Neurons() and inherits it properties/plotting functions.

    Must be initialised with an Agent and a 'params' dictionary.

    HeadDirectionCells defines a set of 'n' head direction cells. Each cell has a preffered direction/angle (default evenly spaced across unit circle). In 1D there are always only n=2 cells preffering left and right directions. The firing rates are scaled such that when agent travels exactly along the preferred direction the firing rate of that cell is the max_fr. The firing field of a cell is a von mises centred around its preferred direction of default width 30 degrees (can be changed with parameter params["angular_spread_degrees"])

    To print/set preffered direction: self.preferred_angles

    List of functions:
        • get_state()

    default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "n":1,
            "angle_spread_degrees":30,
            "name": "HeadDirectionCells",
        }
    """

    def __init__(self, Agent, params={}):
        """Initialise HeadDirectionCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""
        default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "n":4,
            "angular_spread_degrees":30, #width of HDC preference function (degrees) 
            "name": "HeadDirectionCells",
        }
        self.Agent = Agent
        for key in params.keys():
            default_params[key] = params[key]
        self.params = default_params

        if self.Agent.Environment.dimensionality == "2D":
            self.n = self.params['n']
            self.preferred_angles = np.linspace(0,2*np.pi,self.n+1)[:-1]
            # self.preferred_directions = np.array([np.cos(angles),np.sin(angles)]).T #n HDCs even spaced on unit circle
            self.angular_tunings = np.array([self.params['angular_spread_degrees']*np.pi/180]*self.n)
        if self.Agent.Environment.dimensionality == "1D":
            self.n = 2  # one left, one right
        self.params["n"] = self.n
        super().__init__(Agent, self.params)
        if ratinabox.verbose is True:
            print(
                f"HeadDirectionCells successfully initialised. Your environment is {self.Agent.Environment.dimensionality}, you have {self.n} head direction cells"
            )

    def get_state(self, evaluate_at="agent", **kwargs):
        """In 2D n head direction cells encode the head direction of the animal. By default velocity (which determines head direction) is taken from the agent but this can also be passed as a kwarg 'vel'"""

        if evaluate_at == "agent":
            vel = self.Agent.history["vel"][-1]
        elif 'vel' in kwargs.keys():
            vel = np.array(kwargs["vel"])
        else: 
            print("HeadDirection cells need a velocity but not was given, taking...")
            if self.Agent.Environment.dimensionality == "2D":
                vel = np.array([1,0])
                print("...[1,0] as default")
            if self.Agent.Environment.dimensionality == "1D":
                vel = np.array([1])
                print("...[1] as default")
            
        if self.Agent.Environment.dimensionality == "1D":
            hdleft_fr = max(0, np.sign(vel[0]))
            hdright_fr = max(0, -np.sign(vel[0]))
            firingrate = np.array([hdleft_fr, hdright_fr])
        if self.Agent.Environment.dimensionality == "2D":
            current_angle = utils.get_angle(vel)
            firingrate = utils.von_mises(current_angle,self.preferred_angles,self.angular_tunings,norm=1)

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]

        return firingrate
    
    def plot_HDC_receptive_field(self,):
        return 


class VelocityCells(HeadDirectionCells):
    """The VelocityCells class defines a population of Velocity cells. This basically takes the output from a population of HeadDirectionCells and scales it proportional to the speed (dependence on speed and direction --> velocity). 

    Must be initialised with an Agent and a 'params' dictionary. Initalise tehse cells as if they are HeadDirectionCells 

    VelocityCells defines a set of 'dim x 2' velocity cells. Encoding the East, West (and North and South) velocities in 1D (2D). The firing rates are scaled according to the multiple current_speed / expected_speed where expected_speed = Agent.speed_mean + self.Agent.speed_std is just some measure of speed approximately equal to a likely ``rough`` maximum for the Agent. 


    List of functions:
        • get_state()

    default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "name": "VelocityCells",
        }
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
        self.one_sigma_speed = self.Agent.speed_mean + self.Agent.speed_std

        super().__init__(Agent, self.params)

        if ratinabox.verbose is True:
            print(
                f"VelocityCells successfully initialised. Your environment is {self.Agent.Environment.dimensionality} and you have {self.n} velocity cells"
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Takes firing rate of equivalent set of head direction cells and scales by how fast teh speed is realtive to one_sigma_speed (likely rough maximum speed)"""

        HDC_firingrates = super().get_state(evaluate_at, **kwargs)
        speed_scale = np.linalg.norm(self.Agent.velocity) / self.one_sigma_speed
        firingrate = HDC_firingrates * speed_scale
        return firingrate


class SpeedCell(Neurons):
    """The SpeedCell class defines a single speed cell. This class is a subclass of Neurons() and inherits it properties/plotting functions.

    Must be initialised with an Agent and a 'params' dictionary.

    The firing rate is scaled according to the expected velocity of the agent (max firing rate acheive when velocity = mean + std)

    List of functions:
        • get_state()

    default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "name": "SpeedCell",
        }
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

        if ratinabox.verbose is True:
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
    Input params dictionary should contain a list of input_layers which feed into this layer. This list looks like [Input1, Input2,...] where each is a Neurons() class (typically this list will be length 1 but you can have arbitrariy many layers feed into this one if you want). You can also add inputs one-by-one using self.add_input()

    Each layer which feeds into this one is assigned a set of weights
        firingrate = activation_function(sum_over_layers(sum_over_inputs_in_this_layer(w_i I_i)) + bias)
    (you my be interested in accessing these weights in order to write a function which "learns" them, for example). A dictionary stores all the inputs, the key for each input layer is its name (e.g Input1.name = "Input1"), so to get the weights call
        FeedForwardLayer.inputs["Input1"]['w'] --> returns the weight matrix from Input1 to FFL
        FeedForwardLayer.inputs["Input2"]['w'] --> returns the weight matrix from Input1 to FFL
        ...
    One set of biases are stored in self.biases (defaulting to an array of zeros), one for each neuron.

    Currently supported activations include 'sigmoid' (paramterised by max_fr, min_fr, mid_x, width), 'relu' (gain, threshold) and 'linear' specified with the "activation_params" dictionary in the inout params dictionary. See also utils.utils.activate() for full details. It is possible to write your own activatino function (not recommended) under the key {"function" : an_activation_function}, see utils.actvate for how one of these should be written.

    Check that the input layers are all named differently.
    List of functions:
        • get_state()
        • add_input()

    default_params = {
            "n": 10,
            "input_layers": [],  # a list of input layers, or add one by one using self.adD_inout
            "activation_params": {
                "activation": "linear",
            },
            "name": "FeedForwardLayer",
        }
        """

    def __init__(self, Agent, params={}):
        default_params = {
            "n": 10,
            "input_layers": [],  # a list of input layers, or add one by one using self.add_inout
            "activation_params": {"activation": "linear"},
            "name": "FeedForwardLayer",
            "biases": None,  # an array of biases, one for each neuron
        }
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        self.inputs = {}
        for input_layer in self.input_layers:
            self.add_input(input_layer)

        if self.biases is None:
            self.biases = np.zeros(self.n)

        if ratinabox.verbose is True:
            print(
                f"FeedForwardLayer initialised with {len(self.inputs.keys())} layers. To add another layer use FeedForwardLayer.add_input_layer().\nTo set the weights manually edit them by changing self.inputs['layer_name']['w']"
            )

    def add_input(self, input_layer, w=None, w_init_scale=1, **kwargs):
        """Adds an input layer to the class. Each input layer is stored in a dictionary of self.inputs. Each has an associated matrix of weights which are initialised randomly.

        Note the inputs are stored in a dictionary. The keys are taken to be the name of each layer passed (input_layer.name). Make sure you set this correctly (and uniquely).

        Args:
            • input_layer (_type_): the layer intself. Must be a Neurons() class object (e.g. can be PlaceCells(), etc...).
            • w: the weight matrix. If None these will be drawn randomly, see next argument.
            • w_init_scale: initial weights drawn from zero-centred gaussian with std w_init_scale / sqrt(N_in)
            • **kwargs any extra kwargs will get saved into the inputs dictionary in case you need these

        """
        n = input_layer.n
        name = input_layer.name
        if w is None:
            w = np.random.normal(
                loc=0, scale=w_init_scale / np.sqrt(n), size=(self.n, n)
            )
        I = np.zeros(n)
        if name in self.inputs.keys():
            if ratinabox.verbose is True:
                print(
                    f"There already exists a layer called {name}. Overwriting it now."
                )
        self.inputs[name] = {}
        self.inputs[name]["layer"] = input_layer
        self.inputs[name]["w"] = w
        self.inputs[name]["w_init"] = w.copy()
        self.inputs[name]["I"] = I
        for (key, value) in kwargs.items():
            self.inputs[name][key] = value
        if ratinabox.verbose is True:
            print(
                f'An input layer called {name} was added. The weights can be accessed with "self.inputs[{name}]["w"]"'
            )

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

        biases = self.biases
        if biases.shape != V.shape:
            biases = biases.reshape((-1, 1))
        V += biases

        firingrate = utils.activate(V, other_args=self.activation_params)
        # firingrate_prime = utils.activate(
        #    V, other_args=self.activation_params, deriv=True
        # )
        return firingrate
