import ratinabox

import copy
import pprint
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats as stats

from ratinabox import utils

"""NEURONS"""
"""Parent Class"""


class Neurons:
    """The Neuron class defines a population of Neurons. All Neurons have firing rates which depend on the state of the Agent. As the Agent moves the firing rate of the cells adjust accordingly.

    All Neuron classes must be initalised with the Agent (to whom these cells belong) since the Agent determines the firingrates through its position and velocity. The Agent class will itself contain the Environment. Both the Agent (position/velocity) and the Environment (geometry, walls, objects etc.) determine the firing rates. Optionally (but likely) an input dictionary 'params' specifying other params will be given.

    This is a generic Parent class. We provide several SubClasses of it. These include:
    • PlaceCells()
    • GridCells()
    • BoundaryVectorCells()
    • ObjectVectorCells()
    • VelocityCells()
    • HeadDirectionCells()
    • SpeedCells()
    • FeedForwardLayer()
    as well as (in  the contribs)
    • ValueNeuron()
    • FieldOfViewNeurons()

    The unique function in each child classes is get_state(). Whenever Neurons.update() is called Neurons.get_state() is then called to calculate and return the firing rate of the cells at the current moment in time. This is then saved. In order to make your own Neuron subclass you will need to write a class with the following mandatory structure:

    ============================================================================================
    MyNeuronClass(Neurons):

        default_params = {'a_default_param":3.14159} # default params dictionary is defined in the preamble, as a class attribute. Note its values are passed upwards and used in all the parents classes of your class.

        def __init__(self,
                     Agent,
                     params={}): #<-- do not change these


            self.params = copy.deepcopy(__class__.default_params) # to get the default param dictionary of the current class, defined in the preamble, use __class__. Then, make sure to deepcopy it, as only making a shallow copy can have unintended consequences (i.e., any modifications to it would be propagated to ALL instances of this class!).
            self.params.update(params)

            super().__init__(Agent,self.params)

        def get_state(self,
                      evaluate_at='agent',
                      **kwargs) #<-- do not change these

            firingrate = .....
            ###
                Insert here code which calculates the firing rate.
                This may work differently depending on what you set evaluate_at as. For example, evaluate_at == 'agent' should means that the position or velocity (or whatever determines the firing rate) will by evaluated using the agents current state. You might also like to have an option like evaluate_at == "all" (all positions across an environment are tested simultaneously - plot_rate_map() tries to call this, for example) or evaluate_at == "last" (in a feedforward layer just look at the last firing rate saved in the input layers saves time over recalculating them.). **kwargs allows you to pass position or velocity in manually.

                By default, the Neurons.update() calls Neurons.get_state() rwithout passing any arguments. So write the default behaviour of get_state() to be what you want it to do in the main training loop, .
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
            • reset_history()
            • boundary_vector_preference_function()

    default_params = {
            "n": 10,
            "name": "Neurons",
            "color": None,  # just for plotting
        }
    """

    default_params = {
        "n": 10,
        "name": "Neurons",
        "color": None,  # just for plotting
        "noise_std": 0,  # 0 means no noise, std of the noise you want to add (Hz)
        "noise_coherence_time": 0.5,
        "save_history": True,  # whether to save history (set to False if you don't intend to access Neuron.history for data after, for better memory performance)
    }

    def __init__(self, Agent, params={}):
        """Initialise Neurons(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.

        Typically you will not actually initialise a Neurons() class, instead you will initialised by one of it's subclasses.
        """

        self.Agent = Agent
        self.Agent.Neurons.append(self)

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        utils.update_class_params(self, self.params, get_all_defaults=True)
        utils.check_params(self, params.keys())

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

    @classmethod
    def get_all_default_params(cls, verbose=False):
        """Returns a dictionary of all the default parameters of the class, including those inherited from its parents."""
        all_default_params = utils.collect_all_params(cls, dict_name="default_params")
        if verbose:
            pprint.pprint(all_default_params)
        return all_default_params

    def update(self):
        # update noise vector
        dnoise = utils.ornstein_uhlenbeck(
            dt=self.Agent.dt,
            x=self.noise,
            drift=0,
            noise_scale=self.noise_std,
            coherence_time=self.noise_coherence_time,
        )
        self.noise = self.noise + dnoise

        # update firing rate
        if np.isnan(self.Agent.pos[0]):
            firingrate = np.zeros(self.n)  # returns zero if Agent position is nan
        else:
            firingrate = self.get_state()
        self.firingrate = firingrate.reshape(-1)
        self.firingrate = self.firingrate + self.noise
        if self.save_history is True:
            self.save_to_history()
        return

    def plot_rate_timeseries(
        self,
        t_start=0,
        t_end=None,
        chosen_neurons="all",
        spikes=False,
        imshow=False,
        fig=None,
        ax=None,
        xlim=None,
        color=None,
        background_color=None,
        autosave=None,
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
            • imshow - if True will not dispaly as mountain plot but as an image (plt.imshow). Thee "extent" will be (t_start, t_end, 0, 1) in case you want to plot on top of this
            • fig, ax: the figure, axis to plot on (can be None)
            xlim: fix xlim of plot irrespective of how much time you're plotting
            • color: color of the line, if None, defaults to cell class default (probalby "C1")
            • background_color: color of the background if not matplotlib default (probably white)
            • autosave: if True, will try to save the figure to the figure directory `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
            • kwargs sent to mountain plot function, you can ignore these

        Returns:
            fig, ax
        """
        t = np.array(self.history["t"])
        t_end = t_end or t[-1]
        slice = self.Agent.get_history_slice(t_start, t_end)
        rate_timeseries = np.array(self.history["firingrate"][slice])
        spike_data = np.array(self.history["spikes"][slice])
        t = t[slice]

        # neurons to plot
        chosen_neurons = self.return_list_of_neurons(chosen_neurons)
        n_neurons_to_plot = len(chosen_neurons)
        spike_data = spike_data[slice, chosen_neurons]
        rate_timeseries = rate_timeseries[:, chosen_neurons]

        was_fig, was_ax = (fig is None), (
            ax is None
        )  # remember whether a fig or ax was provided as xlims depend on this
        if color is None:
            color = self.color
        if imshow == False:
            firingrates = rate_timeseries.T
            fig, ax = utils.mountain_plot(
                X=t / 60,
                NbyX=firingrates,
                color=color,
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

            xmin = t_start / 60 if was_fig else min(t_start / 60, ax.get_xlim()[0])
            xmax = t_end / 60 if was_fig else max(t_end / 60, ax.get_xlim()[1])
            ax.set_xlim(
                left=xmin,
                right=xmax,
            )
            ax.set_xticks([xmin, xmax])
            ax.set_xticklabels([round(xmin, 2), round(xmax, 2)])
            if xlim is not None:
                ax.set_xlim(right=xlim / 60)
                ax.set_xticks([round(t_start / 60, 2), round(xlim / 60, 2)])
                ax.set_xticklabels([round(t_start / 60, 2), round(xlim / 60, 2)])

            if background_color is not None:
                ax.set_facecolor(background_color)
                fig.patch.set_facecolor(background_color)

        elif imshow == True:
            if fig is None and ax is None:
                fig, ax = plt.subplots(
                    figsize=(
                        ratinabox.MOUNTAIN_PLOT_WIDTH_MM / 25,
                        0.5 * ratinabox.MOUNTAIN_PLOT_WIDTH_MM / 25,
                    )
                )
            data = rate_timeseries.T
            ax.imshow(
                data[::-1],
                aspect="auto",
                # aspect=0.5 * data.shape[1] / data.shape[0],
                extent=(t_start, t_end, 0, 1),
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel("Time / min")
            ax.set_xticks([t_start, t_end])
            ax.set_xticklabels([round(t_start / 60, 2), round(t_end / 60, 2)])
            ax.set_yticks([])
            ax.set_ylabel("Neurons")

        ratinabox.utils.save_figure(fig, self.name + "_firingrate", save=autosave)
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
        autosave=None,
        **kwargs,
    ):
        """Plots rate maps of neuronal firing rates across the environment
        Args:
            •chosen_neurons: Which neurons to plot. string "10" will plot 10 of them, "all" will plot all of them, a list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "10".

            • method: "groundtruth" "history" "neither": which method to use. If "analytic" (default) tries to calculate rate map by evaluating firing rate at all positions across the environment (note this isn't always well defined. in which case...). If "history", plots ratemap by a weighting a histogram of positions visited by the firingrate observed at that position. If "neither" (or anything else), then neither.

            • spikes: True or False. Whether to display the occurence of spikes. If False (default) no spikes are shown. If True both ratemap and spikes are shown.

            • fig, ax (the fig and ax to draw on top of, optional)

            • shape is the shape of the multipanel figure, must be compatible with chosen neurons
            • colorbar: whether to show a colorbar
            • t_start, t_end: in the case where you are plotting spike, or using historical data to get rate map, this restricts the timerange of data you are using
            • autosave: if True, will try to save the figure to the figure directory `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
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
                    "It was not possible to get the rate map by evaluating the firing rate at all positions across the Environment. This is probably because the Neuron class does not support vectorised evaluation, or it does not have an groundtruth receptive field. Instead trying wit ha for-loop over all positions one-by-one (could be slow)Instead, plotting rate map by weighted position histogram method. Here is the error:"
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
            slice = self.Agent.get_history_slice(t_start, t_end)
            pos = np.array(self.Agent.history["pos"])[slice]
            t = t[slice]

            if method == "history":
                rate_timeseries = np.array(self.history["firingrate"])[slice].T
                if len(rate_timeseries) == 0:
                    print("No historical data with which to calculate ratemap.")
            if spikes == True:
                spike_data = np.array(self.history["spikes"])[slice].T
                if len(spike_data) == 0:
                    print("No historical data with which to plot spikes.")

        if self.color == None:
            coloralpha = None
        else:
            coloralpha = list(matplotlib.colors.to_rgba(self.color))
            coloralpha[-1] = 0.5

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)
        N_neurons = len(chosen_neurons)

        # PLOT 2D
        if self.Agent.Environment.dimensionality == "2D":
            from mpl_toolkits.axes_grid1 import ImageGrid

            if fig is None and ax is None:
                if shape is None:
                    Nx, Ny = 1, len(chosen_neurons)
                else:
                    Nx, Ny = shape[0], shape[1]
                env_fig, env_ax = self.Agent.Environment.plot_environment(
                    autosave=False
                )
                width, height = env_fig.get_size_inches()
                plt.close(env_fig)
                plt.show
                fig = plt.figure(figsize=(height * Ny, width * Nx))
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
            for i, ax_ in enumerate(axes):
                _, ax_ = self.Agent.Environment.plot_environment(
                    fig, ax_, autosave=False
                )
            if len(chosen_neurons) != axes.size:
                print(
                    f"You are trying to plot a different number of neurons {len(chosen_neurons)} than the number of axes provided {axes.size}. Some might be missed. Either change this with the chosen_neurons argument or pass in a list of axes to plot on"
                )

            vmin, vmax = 0, 0
            ims = []
            if method in ["groundtruth", "history"]:
                for i, ax_ in enumerate(axes):
                    ex = self.Agent.Environment.extent
                    if method == "groundtruth":
                        rate_map = rate_maps[chosen_neurons[i], :].reshape(
                            self.Agent.Environment.discrete_coords.shape[:2]
                        )
                        im = ax_.imshow(rate_map, extent=ex, zorder=0, cmap="inferno")
                    elif method == "history":
                        rate_timeseries_ = rate_timeseries[chosen_neurons[i], :]
                        rate_map = utils.bin_data_for_histogramming(
                            data=pos,
                            extent=ex,
                            dx=0.05,
                            weights=rate_timeseries_,
                            norm_by_bincount=True,
                        )
                        im = ax_.imshow(
                            rate_map,
                            extent=ex,
                            cmap="inferno",
                            interpolation="bicubic",
                            zorder=0,
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
                for i, ax_ in enumerate(axes):
                    pos_where_spiked = pos[spike_data[chosen_neurons[i], :]]
                    ax_.scatter(
                        pos_where_spiked[:, 0],
                        pos_where_spiked[:, 1],
                        s=5,
                        linewidth=0,
                        alpha=0.7,
                        zorder=1.2,
                        color=(self.color or "C1"),
                    )

        # PLOT 1D
        elif self.Agent.Environment.dimensionality == "1D":
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
                        norm_by_bincount=True,
                    )
                    x, rate_map = utils.interpolate_and_smooth(x, rate_map, sigma=0.03)
                    rate_maps.append(rate_map)
                rate_maps = np.array(rate_maps)

            if fig is None and ax is None:
                fig, ax = plt.subplots(
                    figsize=(
                        ratinabox.MOUNTAIN_PLOT_WIDTH_MM / 25,
                        N_neurons * ratinabox.MOUNTAIN_PLOT_SHIFT_MM / 25,
                    )
                )
                fig, ax = self.Agent.Environment.plot_environment(
                    autosave=False, fig=fig, ax=ax
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

            axes = ax
        ratinabox.utils.save_figure(fig, self.name + "_ratemaps", save=autosave)

        return fig, axes

    def save_to_history(self):
        cell_spikes = np.random.uniform(0, 1, size=(self.n,)) < (
            self.Agent.dt * self.firingrate
        )
        self.history["t"].append(self.Agent.t)
        self.history["firingrate"].append(list(self.firingrate))
        self.history["spikes"].append(list(cell_spikes))

    def reset_history(self):
        for key in self.history.keys():
            self.history[key] = []
        return

    def animate_rate_timeseries(
        self,
        t_start=None,
        t_end=None,
        chosen_neurons="all",
        fps=15,
        speed_up=1,
        autosave=None,
        **kwargs,
    ):
        """Returns an animation (anim) of the firing rates, 25fps.
        Should be saved using command like:
            >>> anim.save("./where_to_save/animations.gif",dpi=300) #or ".mp4" etc...
        To display within jupyter notebook, just call it:
            >>> anim

        Args:
            • t_end (_type_, optional): _description_. Defaults to None.
            • chosen_neurons: Which neurons to plot. string "10" or 10 will plot ten of them, "all" will plot all of them, "12rand" will plot 12 random ones. A list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "all".

            • speed_up: #times real speed animation should come out at.

        Returns:
            animation
        """

        plt.rcParams["animation.html"] = "jshtml"  # for animation rendering in jupyter

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
                autosave=False,
                **kwargs,
            )
            plt.close()
            return

        fig, ax = self.plot_rate_timeseries(
            t_start=0,
            t_end=10 * self.Agent.dt,
            chosen_neurons=chosen_neurons,
            xlim=t_end,
            autosave=False,
            **kwargs,
        )


        anim = matplotlib.animation.FuncAnimation(
            fig,
            animate_,
            interval=1000 * dt,
            frames=int((t_end - t_start) / (dt * speed_up)),
            blit=False,
            fargs=(fig, ax, chosen_neurons, t_start, t_end, dt, speed_up),
        )

        ratinabox.utils.save_animation(anim, "rate_timeseries", save=autosave)

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
                chosen_neurons = np.linspace(
                    0, self.n - 1, min(self.n, int(chosen_neurons))
                ).astype(int)
            elif chosen_neurons[-4:] == "rand":
                chosen_neurons = int(chosen_neurons[:-4])
                chosen_neurons = np.random.choice(
                    np.arange(self.n), size=chosen_neurons, replace=False
                )
        if type(chosen_neurons) is int:
            chosen_neurons = np.linspace(0, self.n - 1, min(self.n, chosen_neurons))
        if type(chosen_neurons) is list:
            chosen_neurons = list(np.array(chosen_neurons).astype(int))
            pass
        if type(chosen_neurons) is np.ndarray:
            chosen_neurons = list(chosen_neurons.astype(int))

        return chosen_neurons




