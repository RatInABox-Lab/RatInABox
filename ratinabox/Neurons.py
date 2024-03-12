import ratinabox

import copy
import warnings
import pprint
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy import stats as stats
import warnings
from matplotlib.collections import EllipseCollection

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
    • AgentVectorCells()
    • FieldOfViewBVCs()
    • FieldOfViewOVCs()
    • FieldOfViewAVCs()
    • VelocityCells()
    • HeadDirectionCells()
    • SpeedCells()
    • FeedForwardLayer()
    • RandomSpatialNeurons()
    as well as (in  the contribs)
    • ValueNeuron()
    • NeuralNetworkNeurons()

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
    """

    default_params = {
        "n": 10,
        "name": "Neurons",
        "color": None,  # just for plotting
        "noise_std": 0,  # 0 means no noise, std of the noise you want to add (Hz)
        "noise_coherence_time": 0.5,
        "min_fr":0.0, #not all cells use max_fr nd min_fr but we define them here in the parent class for those that do 
        "max_fr":1.0,
        "save_history": True,  # whether to save history (set to False if you don't intend to access Neuron.history for data after, for better memory performance)
    }

    def __init__(self, Agent, params={}):
        """Initialise Neurons(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            Agent. The RatInABox Agent these cells belong to. 
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

        self._last_history_array_cache_time = None
        self._history_arrays = {} # this dictionary is the same as self.history except the data is in arrays not lists BUT it should only be accessed via its getter-function `self.get_history_arrays()`. This is because the lists are only converted to arrays when they are accessed, not on every step, so as to save time.

        self.colormap = "inferno" # default colormap for plotting ratemaps 

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

    def update(self, **kwargs):
        """Update the firing rate of the Neurons() class. This is called by the Agent.update() function. This core function should be called by the user on each loop in order to refresh the firing rate of the Neurons() class in line with the Agent's current state. It will also save the firing rate and spikes to the Agent.history dictionary if self.save_history is True.
        
        Args: 
            • kwargs will be passed into get_state()  
        """

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
            firingrate = self.get_state(**kwargs)
        self.firingrate = firingrate.reshape(-1)
        self.firingrate = self.firingrate + self.noise
        if self.save_history is True:
            self.save_to_history()
        return

    def get_state(self, **kwargs):
        raise NotImplementedError("Neurons object needs a get_state() method")

    def get_head_direction_averaged_state(self, evaluate_at="agent", angular_resolution_degrees=10, **kwargs):
        """Like get_state() except it calculates it at all head directions 0-->2pi and then averages over those head directions. Note this will only be relevant (although it will always "work") for Neurons which have some kind of head direction selectivity i.e. Neurons where "head_direction" is a kwarg which is used by get_state(). These include HeadDirectionCells or egocentric-BoundaryVectorCells or any cells you might build out of these. Conversely for PlaceCells, the head_direction argument into get_state() will be ignored so this will just be a more inefficient way of calling get_state().
        
        Args: 
            • evaluate_at: "agent" or "all" or None. If "agent" (default) then the Agent's current position is used to calculate the firing rate. If "all" then the firing rate is calculated at all positions across the environment. If None then you must provide a position or velocity array as a kwarg "pos" or "vel" respectively. Defaults to "agent".
            • angular_resolution_degrees: the angular resolution in degrees at which to calculate the firing rate. Defaults to 10.
            • kwargs: passed into get_state().
        """
        firingrate = np.zeros_like(self.get_state(evaluate_at=evaluate_at, head_direction=[1,0], **kwargs))
        n_angles = int(360 / angular_resolution_degrees)
        firingrate = np.repeat(firingrate[:,:,np.newaxis],n_angles,axis=2)
        angles = np.linspace(0,2*np.pi,n_angles)
        for (i,ang) in enumerate(angles):
            head_direction = np.array([np.cos(ang),np.sin(ang)])
            firingrate[:,:,i] = self.get_state(evaluate_at=evaluate_at, head_direction=head_direction, **kwargs)
        firingrate = np.mean(firingrate,axis=2)
        return firingrate
    
    def plot_rate_timeseries(
        self,
        t_start=0.0,
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
            • spikes (bool, optional): If True, scatters exact spike times underneath each curve of firing rate. Defaults to False.
            the below params I just added for help with animations
            • imshow - if True will not dispaly as mountain plot but as an image (plt.imshow). Thee "extent" will be (t_start, t_end, 0, 1) in case you want to plot on top of this
            • fig, ax: the figure, axis to plot on (can be None)
            • xlim: fix xlim of plot irrespective of how much time you're plotting
            • color: color of the line, if None, defaults to cell class default (probalby "C1")
            • background_color: color of the background if not matplotlib default (probably white)
            • autosave: if True, will try to save the figure to the figure directory `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
            • kwargs sent to mountain plot function, you can ignore these

        Returns:
            fig, ax
        """
        history_arrays = self.get_history_arrays() #gets history data as dictionary of arrays
        t = history_arrays["t"]
        t_end = t_end or t[-1]
        slice = self.Agent.get_history_slice(t_start, t_end)
        rate_timeseries = history_arrays["firingrate"][slice]
        spike_data = history_arrays["spikes"][slice]
        t = t[slice]

        # neurons to plot
        chosen_neurons = self.return_list_of_neurons(chosen_neurons)
        n_neurons_to_plot = len(chosen_neurons)
        spike_data = spike_data[:, chosen_neurons]
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
                        color=(self.color if self.color is not None else "C1"),
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

            • method: "groundtruth" "history" "neither" "ratemaps_provided": which method to use. If "groundtruth" (default) tries to calculate rate map by evaluating firing rate at all positions across the environment (note this isn't always well defined. in which case...). If "groundtruth_headdirectionaveraged" calculates rate maps but averaged over all head directions (2D environments only). If "history", plots ratemap by a weighting a histogram of positions visited by the firingrate observed at that position. If "neither" (or anything else), then neither. If "ratemaps_provided" then you must provide a rate map as a numpy array of shape (n_neurons, n_positions) under the keyworkd argument "ratemaps". This is useful if you have already calculated the rate map and want to plot it without having to recalculate it.

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
        #Set kwargs (TODO make lots of params accessible here as kwargs) 
        spikes_color = kwargs.get("spikes_color", self.color) or "C1"
        bin_size = kwargs.get("bin_size", 0.04) #only relevant if you are plotting by method="history"


        # GET DATA
        if method[:11] == "groundtruth":
            try:
                if method == "groundtruth":
                    rate_maps = self.get_state(evaluate_at="all", **kwargs)
                elif method == "groundtruth_headdirectionaveraged":
                    rate_maps = self.get_head_direction_averaged_state(evaluate_at="all", **kwargs)
            except Exception as e:
                print(
                    "It was not possible to get the rate map by evaluating the firing rate at all positions across the Environment. This is probably because the Neuron class does not support vectorised evaluation, or it does not have an groundtruth receptive field. Instead trying wit ha for-loop over all positions one-by-one (could be slow)Instead, plotting rate map by weighted position histogram method. Here is the error:"
                )
                print("Error: ", e)
                import traceback

                traceback.print_exc()
                method = "history"

        if method == "history" or spikes == True:
            history_data = self.get_history_arrays() # converts lists to arrays (if this wasn't just done) and returns them in a dict same as self.history but with arrays not lists
            t = history_data["t"]
            # times to plot
            if len(t) == 0:
                print("Can't plot rate map by method='history', nor plot spikes, since there is no available data to plot. ")
                return
            t_end = t_end or t[-1]
            position_data_agent = kwargs.get("position_data_agent", self.Agent) # In rare cases you may like to plot this cells rate/spike data against the position of a diffferent Agent. This kwarg enables that. 
            position_agent_history_data = position_data_agent.get_history_arrays()
            slice = position_data_agent.get_history_slice(t_start, t_end)
            pos = position_agent_history_data["pos"][slice]
            t = t[slice]

            if method == "history":
                rate_timeseries = history_data["firingrate"][slice].T
                if len(rate_timeseries) == 0:
                    print("No historical data with which to calculate ratemap.")
            if spikes == True:
                spike_data = history_data["spikes"][slice].T
                if len(spike_data) == 0:
                    print("No historical data with which to plot spikes.")
        if method == "ratemaps_provided":
            try:
                rate_maps = kwargs["ratemaps"]
            except:
                print(
                    "You have specified method='ratemaps_provided' but have not provided the rate maps themselves. Please provide them as a numpy array of shape (n_neurons, n_positions) under the keyworkd argument 'ratemaps'."
                )
                return

        if self.color is None:
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
                    autosave=False, **kwargs,
                )
                width, height = env_fig.get_size_inches()
                plt.close(env_fig)
                plt.show
                fig = plt.figure(figsize=(height * Ny, width * Nx))
                if colorbar == True and (method in ["groundtruth", "history", "groundtruth_headdirectionaveraged"]):
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
                    fig, ax_, autosave=False, **kwargs
                )
            if len(chosen_neurons) != axes.size:
                print(
                    f"You are trying to plot a different number of neurons {len(chosen_neurons)} than the number of axes provided {axes.size}. Some might be missed. Either change this with the chosen_neurons argument or pass in a list of axes to plot on"
                )

            vmin, vmax = 0, 0
            ims = []
            if method in ["groundtruth", "history", "groundtruth_headdirectionaveraged"]:
                for i, ax_ in enumerate(axes):
                    ex = self.Agent.Environment.extent
                    if method[:11] == "groundtruth":
                        rate_map = rate_maps[chosen_neurons[i], :].reshape(
                            self.Agent.Environment.discrete_coords.shape[:2]
                        )
                        im = ax_.imshow(rate_map, extent=ex, zorder=0, cmap=self.colormap, 
                                        interpolation="bicubic", # smooths rate maps but this does slow down the plotting a bit 
                                        )
                    elif method == "history":
                        bin_size = kwargs.get("bin_size", 0.05)
                        rate_timeseries_ = rate_timeseries[chosen_neurons[i], :]
                        rate_map, zero_bins = utils.bin_data_for_histogramming(
                            data=pos,
                            extent=ex,
                            dx=bin_size,
                            weights=rate_timeseries_,
                            norm_by_bincount=True,
                            return_zero_bins=True,
                        )
                        #rather than just "nan-ing" the regions where no data was observed we'll plot ontop a "mask" overlay which blocks with a grey square regions where no data was observed. The benefit of this technique is it still allows us to use "bicubic" interpolation which is much smoother than the default "nearest" interpolation.
                        binary_colors = [(0,0,0,0),ratinabox.LIGHTGREY] #transparent if theres data, grey if there isn't
                        binary_cmap = matplotlib.colors.ListedColormap(binary_colors)
                        im = ax_.imshow(
                            rate_map,
                            extent=ex,
                            cmap=self.colormap,
                            interpolation="bicubic",
                            zorder=0,
                        )
                        no_data_mask = ax_.imshow(
                            zero_bins,
                            extent=ex,
                            cmap=binary_cmap,
                            interpolation="nearest",
                            zorder=0.001,
                        )
                    ims.append(im)
                    vmin, vmax = (
                        min(vmin, np.min(rate_map)),
                        max(vmax, np.max(rate_map)),
                    )
                if "zero_center" in kwargs.keys(): #good for diverging colormaps, makes sure the colorbar is centered on zero
                    if kwargs["zero_center"] == True:
                        vmax = max(abs(vmin), abs(vmax))
                        vmin = -vmax
                for im in ims:
                    im.set_clim((vmin, vmax))
                if colorbar == True:
                    cbar = plt.colorbar(ims[-1], cax=cax)
                    cbar.ax.tick_params(length=0)
                    cbar.set_label("Firing rate / Hz",labelpad=-10)
                    # lim_v = vmax if vmax > -vmin else vmin
                    cbar.set_ticks([vmin,vmax])
                    cbar.set_ticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
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
                        color=spikes_color,
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
                        color=spikes_color,
                        alpha=0.5,
                        s=2,
                        linewidth=0,
                    )
            ax.set_xlabel("Position / m")
            ax.set_ylabel("Neurons")

            axes = ax
        ratinabox.utils.save_figure(fig, self.name + "_ratemaps", save=autosave)

        return fig, axes

    def plot_angular_rate_map(self, chosen_neurons="all", fig=None, ax=None, autosave=None):
        """Plots the position-averaged firing rate map of the neuron as a function of head direction. To do this it calculates the spatial receptive fields at many head directions and averages them over position (therefore it may be slow). 
        Args:
            chosen_neurons (str, optional): The neurons to plot. Defaults to "all".
            fig, ax (_type_, optional): matplotlib fig, ax objects ot plot onto (optional).
            autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`. 
                                       Defaults to None in which case looks for global constant ratinabox.autosave_plots
        """
        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)
        if fig is None and ax is None:
            fig, ax = plt.subplots(
                1,
                len(chosen_neurons),
                figsize=(2 * len(chosen_neurons), 2),
                subplot_kw={"projection": "polar"},
            )


        # get rate maps at all head directions and all positions
        # the object will end up having shape (n_neurons, n_positions, n_headdirections)
        rm = np.zeros_like(self.get_state(evaluate_at='all',head_direction=np.array([1,0])))
        rm = np.repeat(rm[:,:,np.newaxis],100,axis=2)
        angles = np.linspace(0,2*np.pi,100)
        for (i,ang) in enumerate(angles):
            head_direction = np.array([np.cos(ang),np.sin(ang)])
            rm[:,:,i] = self.get_state(evaluate_at='all', head_direction=head_direction)
        
        # average over positions leaving just head direction selectivity
        rm_hd = np.mean(rm,axis=1)
        
        # plot head direction rate map
        for (i,n) in enumerate(chosen_neurons):
            perneuron_rm = rm_hd[n,:]
            ax[i].plot(angles,perneuron_rm,linewidth=2,color=self.color)
            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax[i].fill_between(angles, perneuron_rm, 0, alpha=0.2,facecolor=self.color)
            ax[i].set_ylim([0, 1.1*np.max(rm_hd[n,:])])
            ax[i].tick_params(pad=-20)
            ax[i].set_xticklabels(["E", "N", "W", "S"])
        


        for i, ax_ in enumerate(axes):
            _, ax_ = self.Agent.Environment.plot_environment(
                fig, ax_, autosave=False, **kwargs
            )
        
        ratinabox.utils.save_figure(fig, self.name + "_angularratemaps", save=autosave)

        return fig, ax 
    
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

        from matplotlib import animation

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
    
    def get_history_arrays(self):
        """Returns the history dataframe as a dictionary of numpy arrays (as opposed to lists). This getter-function only updates the self._history_arrays if the Agent/Neuron has updates since the last time it was called. This avoids expensive repeated conversion of lists to arrays during animations."""
        if (self._last_history_array_cache_time != self.Agent.t): 
            self._history_arrays = {}
            self._last_history_array_cache_time = self.Agent.t
            for key in self.history.keys():
                try: #will skip if for any reason this key cannot be converted to an array, so you can still save random stuff into the history dict without breaking this function
                    self._history_arrays[key] = np.array(self.history[key])
                except: pass 
        return self._history_arrays
    

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
    #TO-DO • tanni_harland  https://pubmed.ncbi.nlm.nih.gov/33770492/

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

    def __init__(self, Agent, params={}):
        """Initialise PlaceCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.
        """

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        if self.params["place_cell_centres"] is None:
            self.params["place_cell_centres"] = self.Agent.Environment.sample_positions(
                n=self.params["n"], method="uniform_jitter"
            )
        elif type(self.params["place_cell_centres"]) is str:
            if self.params["place_cell_centres"] in [
                "random",
                "uniform",
                "uniform_jitter",
            ]:
                self.params[
                    "place_cell_centres"
                ] = self.Agent.Environment.sample_positions(
                    n=self.params["n"], method=self.params["place_cell_centres"]
                )
            else:
                raise ValueError(
                    "self.params['place_cell_centres'] must be None, an array of locations or one of the instructions ['random', 'uniform', 'uniform_jitter']"
                )
        else:
            self.params["n"] = self.params["place_cell_centres"].shape[0]
        self.place_cell_widths = self.params["widths"] * np.ones(self.params["n"])

        super().__init__(Agent, self.params)

        # Assertions (some combinations of boundary condition and wall geometries aren't allowed)
        if self.Agent.Environment.dimensionality == "2D":
            if all(
                [
                    (
                        (self.wall_geometry == "line_of_sight")
                        or ((self.wall_geometry == "geodesic"))
                    ),
                    (self.Agent.Environment.boundary_conditions == "periodic"),
                    (self.Agent.Environment.dimensionality == "2D"),
                ]
            ):
                print(
                    f"{self.wall_geometry} wall geometry only possible in 2D when the boundary conditions are solid. Using 'euclidean' instead."
                )
                self.wall_geometry = "euclidean"
            if (self.wall_geometry == "geodesic") and (
                len(self.Agent.Environment.walls) > 5
            ):
                print(
                    "'geodesic' wall geometry only supported for enivironments with 1 additional wall (4 bounding walls + 1 additional). Sorry. Using 'line_of_sight' instead."
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
        dist = (
            self.Agent.Environment.get_distances_between___accounting_for_environment(
                self.place_cell_centres, pos, wall_geometry=self.wall_geometry
            )
        )  # distances to place cell centres
        widths = np.expand_dims(self.place_cell_widths, axis=-1)

        if self.description == "gaussian":
            firingrate = np.exp(-(dist**2) / (2 * (widths**2)))
        if self.description == "gaussian_threshold":
            firingrate = np.maximum(
                np.exp(-(dist**2) / (2 * (widths**2))) - np.exp(-1 / 2),
                0,
            ) / (1 - np.exp(-1 / 2))
        if self.description == "diff_of_gaussians":
            ratio = 1.5
            firingrate = np.exp(-(dist**2) / (2 * (widths**2))) - (
                1 / ratio**2
            ) * np.exp(-(dist**2) / (2 * ((ratio * widths) ** 2)))
            firingrate *= ratio**2 / (ratio**2 - 1)
        if self.description == "one_hot":
            closest_centres = np.argmin(np.abs(dist), axis=0)
            firingrate = np.eye(self.n)[closest_centres].T
        if self.description == "top_hat":
            firingrate = 1 * (dist < self.widths)

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def plot_place_cell_locations(
        self,
        fig=None,
        ax=None,
        autosave=None,
    ):
        """Scatter plots where the centre of the place cells are

        Args:
            fig, ax: if provided, will plot fig and ax onto these instead of making new.
            autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`.Defaults to None in which case looks for global constant ratinabox.autosave_plots

        Returns:
            _type_: _description_
        """
        if fig is None and ax is None:
            fig, ax = self.Agent.Environment.plot_environment(autosave=False)
        else:
            _, _ = self.Agent.Environment.plot_environment(
                fig=fig, ax=ax, autosave=False
            )
        place_cell_centres = self.place_cell_centres

        x = place_cell_centres[:, 0]
        if self.Agent.Environment.dimensionality == "1D":
            y = np.zeros_like(x)
        elif self.Agent.Environment.dimensionality == "2D":
            y = place_cell_centres[:, 1]

        ax.scatter(
            x,
            y,
            c="C1",
            marker="x",
            s=15,
            zorder=2,
        )
        ratinabox.utils.save_figure(fig, "place_cell_locations", save=autosave)

        return fig, ax

    def remap(self):
        """Resets the place cell centres to a new random distribution. These will be uniformly randomly distributed in the environment (i.e. they will still approximately span the space)"""
        self.place_cell_centres = self.Agent.Environment.sample_positions(
            n=self.n, method="uniform_jitter"
        )
        np.random.shuffle(self.place_cell_centres)
        return


class GridCells(Neurons):
    """The GridCells class defines a population of 'n' grid cells with orientations, grid scales and offsets (these can be set randomly or non-randomly). Grids are modelled as the rectified or shifted sum of three cosine waves at 60 degrees to each other.

    To initialise grid cells you specify their (i) params['gridscale'], (ii) params['orientation'] and (iii) params['phase_offset']. These can be handed in as lists/arrays (in which case they are set to these exact values, one per cell) or tuples where the values inside the tuples define the parameters of a distribution (the string defined by params['<param>_distribution']) from which the parameters are sampled. An up-to-date list of avaiable distributions and their parameters in utils.distribution_sampler(), currently avaiable distributions are:
    - uniform ------------------------------- (low, high) or just a single param p which gives (0.5*p, 1.5*p)
    - rayleigh ------------------------------ (scale)
    - normal -------------------------------- (loc, scale)
    - logarithmic --------------------------- (low, high)
    - delta --------------------------------- (the_single_value)
    - modules ------------------------------- (module1_val, module2_val, module3_val, ...)
    - truncnorm ----------------------------- (low, high, loc, scale)
    For example to get three modules of gridcells I could set params = {'gridscale_distribution':'modules', 'gridscale':(0.5, 1, 1.5)} which would give me three modules of grid cells with grid scales 0.5, 1 and 1.5. 

    params['description'] gives the place cells model being used. Currently either rectified sum of three cosines "three_rectified_cosines" or a shifted sum of three cosines "three_shifted_cosines" (which is similar, just a little softer at the edges, see Solstad et al. 2006)
    
    List of functions:
        • get_state()
        • set_phase_offsets()

        }
    """

    default_params = {
        "n": 10,
        "gridscale_distribution": "uniform",
        "gridscale": (0.50, 1),
        "orientation_distribution": "uniform",
        "orientation": (0, 2 * np.pi), #radians 
        "phase_offset_distribution": "uniform",
        "phase_offset": (0, 2 * np.pi), #degrees 
        "description": "three_rectified_cosines",  # can also be "three_shifted_cosines" as in Solstad 2006 Eq. (2)
        "min_fr": 0,
        "max_fr": 1,
        "name": "GridCells",
    }

    def __init__(self, Agent, params={}):
        """Initialise GridCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        # deprecation warnings
        if (
            ("random_gridscales" in self.params)
            or ("random_orientations" in self.params)
            or ("random_phase_offsets" in self.params)
        ):
            warnings.warn(
                "the GridCell API has changed slightly, 'random_gridscales', 'random_orientations' and 'random_phase_offsets' are no longer accepted as parameters. Please use 'gridscale','gridscale_distribution','orientation','orientation_distribution','phase_offset' and 'phase_offset_distribution' instead. See docstring or 1.7.0 release notes for instructions."
            )

        # Initialise the gridscales
        if type(self.params["gridscale"]) in (
            list,
            np.ndarray,
        ):  # assumes you are manually passing gridscales, one for each neuron
            self.gridscales = np.array(self.params["gridscale"])
            self.params["n"] = len(self.gridscales)
        elif type(self.params["gridscale"]) in (
            float,
            tuple,
            int,
        ):  # assumes you are passing distribution parameters
            self.gridscales = utils.distribution_sampler(
                distribution_name=self.params["gridscale_distribution"],
                distribution_parameters=self.params["gridscale"],
                shape=(self.params["n"],),
            )

        # Initialise Neurons parent class
        super().__init__(Agent, self.params)

        # Initialise phase offsets for each grid cell
        if (type(self.params["phase_offset"]) in (list,np.ndarray,)
            and len(np.array(self.params["phase_offset"]).shape) == 2
        ):
            self.phase_offsets = np.array(self.params["phase_offset"])
            assert (
                len(self.phase_offsets) == self.params["n"]
            ), "number of phase offsets supplied incompatible with number of neurons"
        else:
            self.phase_offsets = utils.distribution_sampler(
                distribution_name=self.params["phase_offset_distribution"],
                distribution_parameters=self.params["phase_offset"],
                shape=(self.params["n"], 2),
            )
            if self.params["phase_offset_distribution"] == "grid":
                self.phase_offsets = self.set_phase_offsets_on_grid()

        # Initialise orientations for each grid cell
        if type(self.params["orientation"]) in (
            list,
            np.ndarray,
        ):
            self.orientations = np.array(self.params["orientation"])
            assert (
                len(self.orientations) == self.params["n"]
            ), "number of orientations supplied incompatible with number of neurons"
        else:
            self.orientations = utils.distribution_sampler(
                distribution_name=self.params["orientation_distribution"],
                distribution_parameters=self.params["orientation"],
                shape=(self.params["n"],),
            )

        # Initialise grid cells
        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "grid cells only available in 2D"

        w = []
        for i in range(self.n):
            w1 = np.array([1, 0])
            w1 = utils.rotate(w1, self.orientations[i])
            w2 = utils.rotate(w1, np.pi / 3)
            w3 = utils.rotate(w1, 2 * np.pi / 3)
            w.append(np.array([w1, w2, w3]))
        self.w = np.array(w)

        if ratinabox.verbose is True:
            print(
                "GridCells successfully initialised. You can also manually set their gridscale (GridCells.gridscales), offsets (GridCells.phase_offsets) and orientations (GridCells.w1, GridCells.w2,GridCells.w3 give the cosine vectors)"
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
        origin = self.gridscales.reshape(-1, 1) * self.phase_offsets / (2 * np.pi)
        vecs = utils.get_vectors_between(origin, pos)  # shape = (N_cells,N_pos,2)
        w1 = np.tile(np.expand_dims(self.w[:, 0, :], axis=1), reps=(1, pos.shape[0], 1))
        w2 = np.tile(np.expand_dims(self.w[:, 1, :], axis=1), reps=(1, pos.shape[0], 1))
        w3 = np.tile(np.expand_dims(self.w[:, 2, :], axis=1), reps=(1, pos.shape[0], 1))
        gridscales = np.tile(
            np.expand_dims(self.gridscales, axis=1), reps=(1, pos.shape[0])
        )
        phi_1 = ((2 * np.pi) / gridscales) * (vecs * w1).sum(axis=-1)
        phi_2 = ((2 * np.pi) / gridscales) * (vecs * w2).sum(axis=-1)
        phi_3 = ((2 * np.pi) / gridscales) * (vecs * w3).sum(axis=-1)

        if self.description == "three_rectified_cosines":
            firingrate = (1 / 3) * ((np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3)))
            firingrate[firingrate < 0] = 0
        elif self.description == "three_shifted_cosines":
            firingrate = (2 / 3) * (
                (1 / 3) * (np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3)) + (1 / 2)
            )

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def set_phase_offsets_on_grid(self):
        """Set non-random phase_offsets. Most offsets (n_on_grid, the largest square numer before self.n) will tile a grid of 0 to 2pi in x and 0 to 2pi in y, while the remainings (cell number: n - n_on_grid) are random."""
        n_x = int(np.sqrt(self.n))
        n_y = self.n // n_x
        n_remaining = self.n - n_x * n_y

        dx = 2 * np.pi / n_x
        dy = 2 * np.pi / n_y

        grid = np.mgrid[
            (0 + dx / 2) : (2 * np.pi - dx / 2) : (n_x * 1j),
            (0 + dy / 2) : (2 * np.pi - dy / 2) : (n_y * 1j),
        ]
        grid = grid.reshape(2, -1).T
        remaining = np.random.uniform(0, 2 * np.pi, size=(n_remaining, 2))

        all_offsets = np.vstack([grid, remaining])

        return all_offsets


class VectorCells(Neurons):
    """
    The VectorCells class defines a population of VectorCells. It should only be used as a parent class for one of the follwing subclasses:
        • BoundaryVectorCells ("BVCs")
            • FieldOfViewBVCs
        • ObjectVectorCells ("OVCs")
            • FieldOfViewOVCs
        • AgentVectorCells ("AVCs")
            • FieldOfViewAVCs
    
    All vector cells are similar in that every cell responds to <something> in the Environment with a preferred tuning_distance, tuning_angle, sigma_distances (i.w. width of receptive field in distance) and sigma_angle. They are different in what they respond to (<something> = walls, objects, agents) and how they are arranged (randomly, in a field-of-view, etc...).

    All vector cells have receptive fields which is a von Mises distributions in angle (mean = tuning_angle, 1/sqrt_kappa ~= std = sigma_angle) and 
    a Gaussian in distance (mean = tuning_distance, std = sigma_distance). There are many ways we might like to set these parameters for each neuron...

    A summary of the parameters:
    • params['n']: number of cells (if any params are passed in as lists or arrays, this is overwritten)
    • params['reference_frame']: "allocentric" or "egocentric"
    • parmas['cell_arrangement'] (str): how to set the tuning parameters for each cell. Possible cell arrangements are: 
        • Randomly ["cell_arrangement": "random"] (default)
        • Field-of-view ["cell_arrangement": "uniform_manifold" or "diverging_manifold"]. This is automatically done by the by the `FieldOfViewOVCs/BVCs/AVCs` subclasses. 
        • User defined: ["cell_arrangement": my_funky_tuning_parameter_setting_function] pass a function that returns 4 lists of the same length corresponding to the tuning distances, tuning angles, sigma distances and sigma angles 
    
    iff cell_arrangement is "random" 8 additional parameters, through which to control the parameters of cell receptive fields, become relevant. the following parameters become relevant:
    • params["tuning_distance_distribution"] (str): distribution from which to sample the tuning distances
    • params["tuning_distance"] (tuple or array): if tuple, parameters of the distribution from which to sample the tuning_distance parameter. If array, the exact values of the tuning_distance parameter for each cell
    • params["tuning_angle_distribution"] (str): name of the distribution from which to sample tuning angles parameter
    • params["tuning_angle"] (tuple or array): if tuple, parameters of the distribution from which to sample the tuning_angle parameter. If array, the exact values of the tuning_angle parameter for each cell
    • params["sigma_distance_distribution"] (str): name of the distribution from which to sample the distance spreads
    • params["sigma_distance"] (tuple or array): if tuple, parameters of the distribution from which to sample the sigma_distance parameter. If array, the exact values of the sigma_distance parameter for each cell
    • params["sigma_angle_distribution"] (str): name of the distribution from which to sample the angular spreads
    • params["sigma_angle"] (tuple or array): if tuple, parameters of the distribution from which to sample the sigma_angle parameter. If array, the exact values of the sigma_angle parameter for each cell

    "tuning_distance", "tuning_angle", "sigma_distance" and "sigma_angle" can be handed in as lists/arrays (in which case they are set to these exact values, one per cell) or tuples where the values inside the tuples define the parameters of a distribution (the string defined by <param>_distribution) from which the parameters are sampled. An up to date list of avaiable distributions and their parameters in utils.distribution_sampler(), currently avaiable distributions are:
    - uniform ------------------------------- (low, high) or just a single param p which gives (0.5*p, 1.5*p)
    - rayleigh ------------------------------ (scale)
    - normal -------------------------------- (loc, scale)
    - logarithmic --------------------------- (low, high)
    - delta --------------------------------- (the_single_value)
    - modules ------------------------------- (module1_val, module2_val, module3_val, ...)
    - truncnorm ----------------------------- (low, high, loc, scale)
    The only diversion is that, in addition to the above, sigma_distance_distribution which can be set to "diverging" in which case the radial width of the receptive field is set according to the Hartley model (Hartley et al. 2000). In this case the parameters give (xi, beta) the (offset, inverse-slope) of the linear relationship between sigma and tuning_distance.
    """
    
    default_params = {
        "n": 10,
        "reference_frame": "allocentric",
        "cell_arrangement": "random", 
        #the following are only used if cell_arrangement is "random"
        "tuning_distance_distribution": "uniform",
        "tuning_distance":(0.05,0.3),
        "sigma_distance_distribution": "diverging", #If diverging then params give (xi, beta)
        "sigma_distance" : (0.08, 12),
        "tuning_angle_distribution": "uniform",
        "tuning_angle":(0.0,360),
        "angular_spread_distribution": "uniform",
        "angular_spread": (10, 30),
    }

    def __init__(self, Agent, params={}):
        """Initialise VectorCells(), takes as input an Agent and a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary.
        Args:
            Agent. The RatInABox Agent these cells belong to. 
            params (dict, optional). Defaults to {}.
        """

        if type(self) == VectorCells:
            raise RuntimeError("Cannot instantiate VectorCells on their own. Must be instantiated through one of the subclasses, e.g., ObjectVectorCells, FieldOfView_BVCs, etc.")

        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "Vector cells only possible in 2D"

        #Deprecation warnings
        if 'pref_wall_dist_distribution' in params.keys():
            warnings.warn("'pref_wall_dist_distribution' param is deprecated and will be removed in future versions. Please use 'distance_distribution' instead")
            params['distance_distribution'] = params['pref_wall_dist_distribution']
            del params['pref_wall_dist_distribution']
        if 'pref_wall_dist' in params.keys():
            warnings.warn("'pref_wall_dist' param is deprecated. Please use 'distance' instead")
            params['distance'] = params['pref_wall_dist']
            del params['pref_wall_dist']
        if 'pref_object_dist' in params.keys():
            warnings.warn("'pref_object_dist' param is deprecated. Please use 'distance' instead")
            params['distance'] = params['pref_object_dist']
            del params['pref_object_dist']
        if "distance_spread" in params.keys():
            warnings.warn("'distance_spread' param is deprecated. Please use 'sigma_distance' instead")
            params['sigma_distance'] = params['distance_spread']
            del params['distance_spread']
        if "distance_spread_distribution" in params.keys():
            warnings.warn("'distance_spread_distribution' param is deprecated. Please use 'sigma_distance_distribution' instead")
            params['sigma_distance_distribution'] = params['distance_spread_distribution']
            del params['distance_spread_distribution']

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Agent, self.params)

        self.tuning_angles = None
        self.tuning_distances = None
        self.sigma_angles = None
        self.sigma_distances = None
        
        # set the parameters of each cell. 
        (self.tuning_distances, 
         self.tuning_angles, 
         self.sigma_distances, 
         self.sigma_angles) = self.set_tuning_parameters(**self.params)

        # records whether n was passed as a parameter.
        if not hasattr(self, "_warn_if_n_changes"):
            self._warn_if_n_changes = ("n" in params.keys() and params["n"] is not None)

        # raises a warning if n was passed as a parameter, but will change.
        if self._warn_if_n_changes:
            dont_check_equality = (isinstance(self.params["cell_arrangement"], str) and self.params["cell_arrangement"].endswith("manifold"))
            if dont_check_equality or self.n != len(self.tuning_distances):
                warnings.warn(f"Ignoring 'n' parameter value ({params['n']}) that was passed, and setting number of {self.name} neurons to {len(self.tuning_distances)}, inferred from the cell arrangement parameter.")
        
        self.n = len(self.tuning_distances) # ensure n is correct
 

        self.firingrate = np.zeros(self.n)
        self.noise = np.zeros(self.n)
        self.cell_colors = None 

  
    def set_tuning_parameters(self, **kwargs):
        """Get the tuning parameters for the vector cells.
         Args:
            • cell_arrangement (str | function ): "random_manifold" (randomly generated manifolds) or
                                                  "uniform_manifold" (all receptive fields the same size or 
                                                  "diverging_manifold" (receptive fields are bigger, the further away from the agent they are)
                                                  OR any function that returns 4 lists of the same length corresponding to 
                                                  the tuning distances, tuning angles, sigma distances and sigma angles

        Returns:
            • manifold (dict): a tuple containing all details of the manifold of cells including
                • "tuning_distances" : distance preferences
                • "tuning_angles" : angular preferences (radians)
                • "sigma_distances" : distance tunings - variance for the von miss distribution
                • "sigma_angles" : angular tunings (radians) - variance for the von miss distribution

        """

        mu_d, mu_theta, sigma_d, sigma_theta = None, None, None, None

        # check if the manifold is a function passed by the user 
        if callable(self.params['cell_arrangement']): #users passed own function 
            mu_d, mu_theta, sigma_d, sigma_theta =  self.params['cell_arrangement'](**kwargs)
        elif (self.cell_arrangement is None) or self.cell_arrangement[:6] == "random": #random tuning for each cell 
            mu_d, mu_theta, sigma_d, sigma_theta = utils.create_random_assembly(**kwargs)
        elif self.cell_arrangement == "uniform_manifold": #radial assembly uniform width of all cells 
            mu_d, mu_theta, sigma_d, sigma_theta =  utils.create_uniform_radial_assembly(**kwargs)
        elif self.cell_arrangement == "diverging_manifold": #radial assembly diverging width of all cells 
            mu_d, mu_theta, sigma_d, sigma_theta =  utils.create_diverging_radial_assembly(**kwargs)
        else:
            raise ValueError("cell_arrangement must be either 'uniform_manifold' or 'diverging_manifold' or a function")
        
        # ensure the lists are not None
        assert mu_d is not None, "Tuning distance must not be None"
        assert mu_theta is not None, "Tuning angles must not be None"
        assert sigma_d is not None, "Sigma distance must not be None"
        assert sigma_theta is not None, "Sigma angles must not be None"
        
        #convert the lists to arrays 
        mu_d, mu_theta, sigma_d, sigma_theta = (
            np.array(mu_d),
            np.array(mu_theta),
            np.array(sigma_d),
            np.array(sigma_theta),
        )

        #ensure correct shape of the arrays all the same length
        assert len(mu_d) == len(mu_theta) == len(sigma_d) == len(sigma_theta), "All manifold tuning parameters must be of the same length"

        return mu_d, mu_theta, sigma_d, sigma_theta


    def display_manifold(self,
                        fig=None,
                        ax=None,
                        t=None,
                        **kwargs):
        '''This is an alias for the display_vector_cells() function. See that for more details.
            This is not deprecated. Please use display_vector_cells() instead.
        '''
        warnings.warn("display_manifold() is deprecated. Please use display_vector_cells() instead.")
        return self.display_vector_cells(fig=fig, ax=ax, t=t, **kwargs)

    def display_vector_cells(self, 
                         fig=None, 
                         ax=None, 
                         t=None,
                         **kwargs):
        """Visualises the current firing rate of these cells relative to the Agent. 
        Essentially this plots the "manifold" ontop of the Agent. 
        Each cell is plotted as an ellipse where the alpha-value of its facecolor reflects the current firing rate 
        (normalised against the approximate maximum firing rate for all cells, but, take this just as a visualisation).
        Each ellipse is an approximation of the receptive field of the cell which is a von Mises distribution in angule and a Gaussian in distance.
        The width of the ellipse in r and theta give 1 sigma of these distributions (for von Mises: kappa ~= 1/sigma^2).

        This assumes the x-axis in the Agent's frame of reference is the heading direction. 
        (Or the heading diection is the X-axis for egocentric frame of reference). IN this case the Y-axis is the towards the "left" of the agent.

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

        

        y_axis_wrt_agent = np.array([0, 1])
        x_axis_wrt_agent = np.array([1,0])
        head_direction = self.Agent.history["head_direction"][t_id]
        head_direction_angle = 0.0
        

        if self.reference_frame == "egocentric":
            head_direction = self.Agent.history["head_direction"][t_id]
            # head direction angle (CCW from true North)
            head_direction_angle = (180 / np.pi) * ratinabox.utils.get_angle(head_direction)  
            
            # this assumes the "x" dimension is the agents head direction and "y" is to its left
            x_axis_wrt_agent = head_direction / np.linalg.norm(head_direction)  
            y_axis_wrt_agent = utils.rotate(x_axis_wrt_agent, np.pi / 2)


        fr = np.array(self.history["firingrate"][t_id])

        x = self.tuning_distances * np.cos(self.tuning_angles)
        y = self.tuning_distances * np.sin(self.tuning_angles)

        pos_of_cells = pos + np.outer(x, x_axis_wrt_agent) + np.outer(y, y_axis_wrt_agent)

        ww = self.sigma_angles * self.tuning_distances
        hh = self.sigma_distances
        aa  = 1.0 * head_direction_angle + self.tuning_angles * 180 / np.pi

        ec = EllipseCollection(ww,hh, aa, units = 'x',
                                offsets = pos_of_cells,
                                offset_transform = ax.transData,
                                linewidth=0.5,
                                edgecolor="dimgrey",
                                zorder = 2.1,
                                )
        if self.cell_colors is None: 
            facecolor = self.color if self.color is not None else "C1"
            facecolor = np.array(matplotlib.colors.to_rgba(facecolor))
            facecolor_array = np.tile(np.array(facecolor), (self.n, 1))
        else:
            facecolor_array = self.cell_colors #made in child class init. Each cell can have a different plot color. 
            # e.g. if cells are slective to different object types or however you like 
        facecolor_array[:, -1] = 0.7*np.maximum(
            0, np.minimum(1, fr / (0.5 * self.max_fr))
        ) # scale alpha so firing rate shows as how "solid" (up to 0.7 so you can _just_ seen whats beneath) to color of this vector cell is. 
        ec.set_facecolors(facecolor_array)
        ax.add_collection(ec) 

        return fig, ax


class BoundaryVectorCells(VectorCells):
    """The BoundaryVectorCells class defines a population of Boundary Vector Cells.
    These are a subclass of VectorCells() which are selective to walls and boundaries in the environment.

    By default cell tuning params are initialised randomly and are "allocentric" (see VecTorCells doc string for more details).
    
    BVCs can be arranged in an egocentric "field of view" (see FieldOfViewBVCs subclass). 

    List of functions:
        • get_state()
        • boundary_vector_preference_function()
        • plot_BVC_receptive_field()
    """

    default_params = {
        "n": 10,
        "name": "BoundaryVectorCells",
        "dtheta": 2, #angular resolution in degrees used for integration over all angles (smaller is more accurate but slower)
    }

    def __init__(self, Agent, params={}):
        """
        Initialise BoundaryVectorCells(), takes as input a parameter dictionary. 
        Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.
        """

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        # records whether n was passed as a parameter.
        if not hasattr(self, "_warn_if_n_changes"):
            self._warn_if_n_changes = ("n" in params.keys() and params["n"] is not None)

        super().__init__(Agent, self.params)

        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "boundary cells only possible in 2D"
        assert (
            self.Agent.Environment.boundary_conditions == "solid"
        ), "boundary cells only possible with solid boundary conditions"

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
        
        
        # calculate normalising constants for BVS firing rates in the current environment. Any extra walls you add from here onwards you add will likely push the firingrate up further.
        locs = self.Agent.Environment.discretise_environment(dx=0.04)
        locs = locs.reshape(-1, locs.shape[-1])
        
        self.cell_fr_norm = np.ones(self.n) #value for initialization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignores the warning raised during initialisation of the BVCs
            self.cell_fr_norm = np.max(self.get_state(evaluate_at=None, pos=locs), axis=1)

        # list of colors for each cell, just used by `.display_vector_cells()` plotting function
        color = np.array(matplotlib.colors.to_rgba(self.color if self.color is not None else "C1")).reshape(1,-1)
        self.cell_colors = np.tile(color,(self.n,1))

        if ratinabox.verbose is True:
            print(
                "BoundaryVectorCells (BVCs) successfully initialised. You can also manually set their orientation preferences (BVCs.tuning_angles, BVCs.sigma_angles), distance preferences (BVCs.tuning_distances, BVCs.sigma_distances)."
            )
        return
    

    def get_state(self, evaluate_at="agent", **kwargs):
        """
        Here we implement the same type if boundary vector cells as de Cothi et al. (2020), 
        who follow Barry & Burgess, (2007). See equations there.

        The way we do this is a little complex. We will describe how it works from a single position 
        (but remember this can be called in a vectorised manner from an array of positons in parallel)
            1. An array of normalised "test vectors" span, in all directions at small increments, from the current position
            2. These define an array of line segments stretching from [pos, pos+test vector]
            3. Where these line segments collide with all walls in the environment is established, 
               this uses the function "utils.vector_intercepts()"
            4. This pays attention to only consider the first (closest) wall forawrd along a line segment. 
               Walls behind other walls are "shaded" by closer walls. Its a little complex to do this and 
               requires the function "boundary_vector_preference_function()"
            5. Now that, for every test direction, the closest wall is established it is simple a process 
               of finding the response of the neuron to that wall segment at that angle 
               (multiple of two gaussians, see de Cothi (2020)) and then summing over all wall segments 
               for all test angles.

        We also apply a check in the middle to utils.rotate the reference frame into that of the head direction 
        of the agent iff self.reference_frame='egocentric'.

        By default position is taken from the Agent and used to calculate firing rates. This can also by passed 
        directly (evaluate_at=None, pos=pass_array_of_positions) or you can use all the positions in the 
        environment (evaluate_at="all").
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
                head_direction = self.Agent.head_direction
            elif "head_direction" in kwargs.keys():
                head_direction = kwargs["head_direction"]
            elif "vel" in kwargs.keys():
                # just to make backwards compatible
                warnings.warn("'vel' kwarg deprecated in favour of 'head_direction'")
                head_direction = kwargs["vel"]
            else:
                head_direction = np.array([1, 0])
                warnings.warn(
                    "BVCs in egocentric plane require a head direction vector but none was passed. Using [1,0]"
                )
            head_bearing = utils.get_angle(head_direction)
            test_angles -= head_bearing  # account for head direction

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
        """This is a random function needed to efficiently produce boundary vector cells. 
        x is any array of final dimension shape shape[-1]=2. As I use it here x has the form of the 
        output of utils.vector_intercepts. I.e. each point gives shape[-1]=2 lambda values (lam1,lam2) 
        for where a pair of line segments intercept. This function gives a preference for each pair. 
        
        Preference is -1 if lam1<0 (the collision occurs behind the first point) and if lam2>1 or lam2<0 
        (the collision occurs ahead of the first point but not on the second line segment). If neither of these 
        are true it's 1/x (i.e. it prefers collisions which are closest).

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
        """
        Plots the receptive field (in polar corrdinates) of the BVC cells. For allocentric BVCs 
        "up" in this plot == "East", for egocentric BVCs, up == the head direction of the animals

        Args:
            chosen_neurons: Which neurons to plot. Can be int, list, array or "all". Defaults to "all".
            fig, ax: the figure/ax object to plot onto (optional)
            autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`. 
                                    Defaults to None in which case looks for global constant ratinabox.autosave_plots

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


class FieldOfViewBVCs(BoundaryVectorCells):
    """FieldOfViewBVCs  are collection of boundary vector cells organised so as to represent
    the local field of view i.e. what walls agent can "see" in the local vicinity. They work as follow:

    General FieldOfView cell description (also applies to FieldOfViewOVCs):
        A radial assembly of vector cells tiling the agents field of view (FoV) is created. Users define
        the extent of the FoV (distance_range nad angle_range) then concentric rows of egocentric cells
        at increasing distances from the Agent are placed to tile this FoV. If a feature (object or boundary) 
        is located in the agents FoV then cells tiling this section of the FoV will fire. Each cell is roughly
        circular so its angular width (in meters at the corresponding radius) matches its radial width.
        The size of the receptive field either diverge with radius params["cell_arrangement"] = "diverging_manifold"
        (as observed in the brain, default here) or stay the same "uniform_manifold".

        In order to visuale the assembly, we created a plotting function. First make a trajectory figure, 
        then pass this into the plotting func:
            >>> fig, ax = Ag.plot_trajectory()
            >>> fig, ax = my_FoVNeurons.display_vector_cells(fig, ax)
        or animate it with
            >>> fig, ax = Ag.animate_trajectory(additional_plot_func=my_FoVNeurons.display_vector_cells)
        (this plotting function lives in the vector cell parent class so actually non-field of view cells can be plotted like this too)
        We have a demo showing how this works. 
    """

    default_params = {
        "distance_range": [0.02, 0.4],  # min and max distances the agent can "see"
        "angle_range": [0,75],  # angluar FoV in degrees (will be symmetric on both sides, so give range in 0 (forwards) to 180 (backwards)
        "spatial_resolution": 0.02,  # resolution of the inner row of cells (in meters)
        "cell_arrangement": "diverging_manifold",  # cell receptive field widths can diverge with radius "diverging_manifold" or stay constant "uniform_manifold".
        "beta": 5, # smaller means larger rate of increase of cell size with radius in diverging type manifolds
        "color":ratinabox.DARKGREY,
        }

    def __init__(self,Agent,params={}):

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        self.params["reference_frame"] = "egocentric"
        assert self.params["cell_arrangement"] is not None, "cell_arrangement must be set for FoV Neurons"

        super().__init__(Agent, self.params)




class ObjectVectorCells(VectorCells):
    """The ObjectVectorCells class defines a population of Object Vector Cells.
    These are a subclass of VectorCells() which are selective to objects in the environment.

    By default cell tuning params are initialised randomly and are "allocentric" (see VectorCells doc string for more details). Which "type" of object these are selective for can be specified with the "object_tuning_type" param.
    
    OVCs can be arranged in an egocentric "field of view" (see FieldOfViewOVCs subclass). 

    List of functions:
        • get_state()
        • set_tuning_types()
    """

    default_params = {
        "n": 10,
        "name": "ObjectVectorCell",
        "walls_occlude": True, #objects behind walls cannot be seen
        "object_tuning_type" : "random", #can be "random", any integer, or a list of integers of length n. The tuning types of the OVCs. 
    }

    def __init__(self, Agent, params={}):
        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        assert (
            self.Agent.Environment.dimensionality == "2D"
        ), "object vector cells only possible in 2D"

        # records whether n was passed as a parameter.
        if not hasattr(self, "_warn_if_n_changes"):
            self._warn_if_n_changes = ("n" in params.keys() and params["n"] is not None)

        super().__init__(Agent, self.params)

        self.object_locations = self.Agent.Environment.objects["objects"]
        if len(self.object_locations) == 0:
            raise RuntimeError(f"Cannot initialize {self.params['name']}, as there are no objects in the environment.")

        self.tuning_types = None


        # preferred object types
        self.set_tuning_types(self.object_tuning_type)

        if self.walls_occlude == True:
            self.wall_geometry = "line_of_sight"
        else:
            self.wall_geometry = "euclidean"


        # list of colors for each cell, just used by `.display_vector_cells()` plotting function
        self.cell_colors = []
        cmap = matplotlib.colormaps[self.Agent.Environment.object_colormap]
        for i in range(self.n):
            c = cmap(self.tuning_types[i] /  (self.Agent.Environment.n_object_types - 1 + 1e-8))
            self.cell_colors.append(np.array(matplotlib.colors.to_rgba(c)))
        self.cell_colors = np.array(self.cell_colors)
        self.color = (self.params['color'] or self.cell_colors[0]) #this will obviously not work if you have different colors for each cell but most of the time itll work great. Passing in a color to the params dict will override this.

        if ratinabox.verbose is True:
            print(
                "ObjectVectorCells (OVCs) successfully initialised. \
                You can also manually set their orientation preferences \
                (OVCs.tuning_angles, OVCs.sigma_angles), distance preferences \
                (OVCs.tuning_distances, OVCs.sigma_distances)."
            )
        
        
        return


    def set_tuning_types(self, tuning_types=None):
        """Sets the preferred object types for each OVC.

        This is called automatically when the OVCs are initialised.
        """

        if tuning_types == "random":
            self.object_types = self.Agent.Environment.objects["object_types"]
            self.tuning_types = np.random.choice(
                np.unique(self.object_types), replace=True, size=(self.n,)
            )
        else:
            if isinstance(tuning_types, int):
                tuning_types = np.repeat(tuning_types, self.n)
            elif isinstance(tuning_types, list):
                tuning_types = np.array(tuning_types)
            
            assert isinstance(tuning_types, np.ndarray), "tuning_types must be an integer, list or numpy array"
            assert tuning_types.shape[0] == self.n, f"Tuning types must be a vector of length of the number of neurons: ({self.n},)"

            self.tuning_types = tuning_types


    
    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the ObjectVectorCells.

        The way we do this is a little complex. We will describe how it works from a single position to a single OVC
        (but remember this can be called in a vectorised manner from an array of positons in parallel and there are 
        in principle multiple OVCs)
            1. A vector from the position to the object is calculated.
            2. The bearing of this vector is calculated and its length. Note if self.reference_frame == "egocentric" 
               then the bearing is relative to the heading direction of the agent (along its current velocity), not true-north.
            3. Since the distance to the object is calculated taking the environment into account if there is a wall 
               occluding the agent from the obvject this object will not fire.
            4. It is now simple to calculate the firing rate of the cell. Each OVC has a preferred distance and angle
               away from it which cause it to fire. Its a multiple of a gaussian (distance) and von mises (for angle) which creates the eventual firing rate.

        By default position is taken from the Agent and used to calculate firing rates. This can also by passed directly 
        (evaluate_at=None, pos=pass_array_of_positions) or you can use all the positions in the environment (evaluate_at="all").

        Returns:
            firingrates: an array of firing rates
        """
        if evaluate_at == "agent":
            pos = self.Agent.pos
        elif evaluate_at == "all":
            pos = self.Agent.Environment.flattened_discrete_coords
        else:
            pos = kwargs["pos"]

        object_locations = self.Agent.Environment.objects["objects"]
        pos = np.array(pos)
        pos = pos.reshape(-1, pos.shape[-1])  # (N_pos, 2)
        N_pos = pos.shape[0]
        N_cells = self.n
        N_objects = len(object_locations)


        if N_objects == 0:
            # no objects in the environment
            return np.zeros((N_cells, N_pos))

        # 1. GET VECTORS FROM POSITIONS TO OBJECTS 
        (
            distances_to_objects,
            vectors_to_objects,
        ) = self.Agent.Environment.get_distances_between___accounting_for_environment(
            pos,
            object_locations,
            return_vectors=True,
            wall_geometry=self.wall_geometry,
        )  # (N_pos,N_objects) (N_pos,N_objects,2)
        flattened_vectors_to_objects = -1 * vectors_to_objects.reshape(
            -1, 2
        )  # (N_pos x N_objects, 2) #vectors go from pos2 to pos1 so must multiply by -1 
        # flatten is just for the get angle API, reshaping it later
        bearings_to_objects = (
            utils.get_angle(flattened_vectors_to_objects, is_array=True).reshape(
                N_pos, N_objects
            )
        )  # (N_pos,N_objects) 

        # 2. ACCOUNT FOR HEAD DIRECTION IF EGOCENTRIC. THEN CALCULATE BEARINGS TO OBJECTS
        if self.reference_frame == "egocentric":
            if evaluate_at == "agent":
                head_direction = self.Agent.head_direction
            elif "head_direction" in kwargs.keys():
                head_direction = kwargs["head_direction"]
            elif "vel" in kwargs.keys():
                # just to make backwards compatible
                warnings.warn("'vel' kwarg deprecated in favour of 'head_direction'")
                head_direction = kwargs["vel"]
            else:
                head_direction = np.array([1, 0])
                warnings.warn(
                    "OVCs in egocentric plane require a head direction vector but none was passed. Using [1,0]"
                )
            head_bearing = utils.get_angle(head_direction)
            bearings_to_objects -= head_bearing  # account for head direction

        # 3. COLLECT TUNING DETAILS OF EACH CELL AND PUT IN THE RIGHT SHAPE 
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
        ) # (N_objects,N_cell)
        object_types = np.tile(
            np.expand_dims(self.Agent.Environment.objects["object_types"], axis=-1), reps=(1, N_cells)
        ) # (N_objects,N_cell)

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
        ).T  # (N_cell,N_pos), sum over objects which this cell is selective to #TODO: a single cell can fire with 2 types of objects 
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate



class FieldOfViewOVCs(ObjectVectorCells):
    """FieldOfViewOVCs  are collection of object vector cells organised so as to represent
    the local field of view i.e. what objects agent can "see" in the local vicinity. They work as follow:

    General FieldOfView cell description (also applies to FieldOfViewOVCs):
        Please see fieldOfViewBVCs doc string
    
    Users should specify the object type they are selective for with the "object_tuning_type" parameter. 
    """

    default_params = {
        "distance_range": [0.02, 0.4],  # min and max distances the agent can "see"
        "angle_range": [0,75],  # angluar FoV in degrees (will be symmetric on both sides, so give range in 0 (forwards) to 180 (backwards)
        "spatial_resolution": 0.02,  # resolution of each BVC tiling FoV
        "beta": 5, # smaller means larger rate of increase of cell size with radius in hartley type manifolds
        "cell_arrangement": "diverging_manifold",  # whether all cells have "uniform" receptive field sizes or they grow ("hartley") with radius.
        "object_tuning_type" : None, #can be "random", any integer, or a list of integers of length n. The tuning types of the OVCs.
    }

    def __init__(self,Agent,params={}):

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        if self.params["object_tuning_type"] is None:
            warnings.warn("For FieldOfViewOVCs you must specify the object type they are selective for with the 'object_tuning_type' parameter. This can be 'random' (each cell in the field of view chooses a random object type) or any integer (all cells have the same preference for this type). For now defaulting to params['object_tuning_type'] = 0.")
            self.params["object_tuning_type"] = 0

        self.params["reference_frame"] = "egocentric"
        assert self.params["cell_arrangement"] is not None, "cell_arrangement must be set for FOV Neurons"

        super().__init__(Agent, self.params)


class AgentVectorCells(VectorCells):
    """The AgentVectorCells class defines a population of Agent Vector Cells.
    These are a subclass of VectorCells() which are selective to other agents in the environment.

    By default cell tuning params are initialised randomly and are "allocentric" (see VectorCells doc string for more details).
    This is one of the few riab classes which takes another agent as an argument. This is the agent which the cells are selective for. It must be passed as the second argument to the constructor, "Other_Agent".

    AVCs can be arranged in an egocentric "field of view" (see FieldOfViewAVCs subclass). 

    List of functions:
        • get_state()
    """


    default_params = {
        "name":'AgentVectorCell',
        "walls_occlude": True, #objects behind walls cannot be seen
    }

    def __init__(self, 
                 Agent,
                 Other_Agent, #this must be another riab Agent object
                 params={}):

        self.Agent = Agent
        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        # records whether n was passed as a parameter.
        if not hasattr(self, "_warn_if_n_changes"):
            self._warn_if_n_changes = ("n" in params.keys() and params["n"] is not None)

        super().__init__(Agent, self.params)

        # have a list to detect which agent will the cell detect 
        self.tuning_type_agent = Other_Agent

        if self.walls_occlude == True:
            self.wall_geometry = "line_of_sight"
        else:
            self.wall_geometry = "euclidean"

        # list of colors for each cell, just used by `.display_vector_cells()` plotting function
        color = np.array(matplotlib.colors.to_rgba(f"C{self.tuning_type_agent.agent_idx}")).reshape(1,-1) #TODO: colours 
        self.cell_colors = np.repeat(color, self.n, axis=0)

        if ratinabox.verbose is True:
            print(
                "AgentVectorCells (OVCs) successfully initialised."
            )
        
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Returns the firing rate of the AgentVectorCells.

        The way we do this is a little complex. We will describe how it works from a single position to a single VC
        (but remember this can be called in a vectorised manner from an array of positons in parallel and there are 
        in principle multiple VCs)
            1. A vector from the position to the agent is calculated.
            2. The bearing of this vector is calculated and its length. Note if self.reference_frame == "egocentric" 
               then the bearing is relative to the heading direction of the agent (along its current velocity), not true-north.
            3. Since the distance to the object is calculated taking the environment into account if there is a wall 
               occluding the agent from the obvject this object will not fire.
            4. It is now simple to calculate the firing rate of the cell. Each OVC has a preferred distance and angle
               away from it which cause it to fire. Its a multiple of a gaussian (distance) and von mises (for angle) which creates the eventual firing rate.

        By default position is taken from the Agent and used to calculate firing rates. This can also by passed directly 
        (evaluate_at=None, pos=pass_array_of_positions) or you can use all the positions in the environment (evaluate_at="all").

        Returns:
            firingrates: an array of firing rates
        """
        if evaluate_at == "agent":
            pos = self.Agent.pos
        elif evaluate_at == "all":
            pos = self.Agent.Environment.flattened_discrete_coords
        else:
            pos = kwargs["pos"]

        if self.tuning_type_agent is None:
            return np.zeros_like(self.firingrate)

        agent_location = self.tuning_type_agent.pos

        pos = np.array(pos)
        pos = pos.reshape(-1, pos.shape[-1])  # (N_pos, 2)
        N_pos = pos.shape[0]
        N_cells = self.n
        N_objects = 1

        # 1. GET VECTORS FROM POSITIONS TO OBJECTS 
        (
            distances_to_other_agent, # (N_pos,N_objects)
            vectors_to_agents, # (N_pos,N_objects,2)
        ) = self.Agent.Environment.get_distances_between___accounting_for_environment(
            pos,
            agent_location,
            return_vectors=True,
            wall_geometry=self.wall_geometry,
        )  
        flattened_vectors_to_objects = -1 * vectors_to_agents.reshape(
            -1, 2
        )  # (N_pos x N_objects, 2) #vectors go from pos2 to pos1 so must multiply by -1 
        # flatten is just for the get angle API, reshaping it later
        bearings_to_objects = (
            utils.get_angle(flattened_vectors_to_objects, is_array=True).reshape(
                N_pos, N_objects
            )
        )  # (N_pos,N_objects) 

        # 2. ACCOUNT FOR HEAD DIRECTION IF EGOCENTRIC. THEN CALCULATE BEARINGS TO OBJECTS
        if self.reference_frame == "egocentric":
            if evaluate_at == "agent":
                head_direction = self.Agent.head_direction
            elif "head_direction" in kwargs.keys():
                head_direction = kwargs["head_direction"]
            elif "vel" in kwargs.keys():
                # just to make backwards compatible
                warnings.warn("'vel' kwarg deprecated in favour of 'head_direction'")
                head_direction = kwargs["vel"]
            else:
                head_direction = np.array([1, 0])
                warnings.warn(
                    "OVCs in egocentric plane require a head direction vector but none was passed. Using [1,0]"
                )
            head_bearing = utils.get_angle(head_direction)
            bearings_to_objects -= head_bearing  # account for head direction

        # 3. COLLECT TUNING DETAILS OF EACH CELL AND PUT IN THE RIGHT SHAPE 
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
        object_types = np.tile(
            np.expand_dims(self.Agent.Environment.objects["object_types"], axis=-1), reps=(1, N_cells)
        ) # (N_objects,N_cell)

        distances_to_other_agent = np.tile(
            np.expand_dims(distances_to_other_agent, axis=-1), reps=(1, 1, N_cells)
        )  # (N_pos,N_objects,N_cells)
        bearings_to_objects = np.tile(
            np.expand_dims(bearings_to_objects, axis=-1), reps=(1, 1, N_cells)
        )  # (N_pos,N_objects,N_cells)

        firingrate = utils.gaussian(
            distances_to_other_agent, tuning_distances, sigma_distances, norm=1
        ) * utils.von_mises(
            bearings_to_objects, tuning_angles, sigma_angles, norm=1
        )  # (N_pos,N_objects,N_cell)

        firingrate = np.sum(
            firingrate, axis=1
        ).T  # (N_cell,N_pos), sum over objects which this cell is selective to #TODO: a single cell can fire with 2 types of objects 
        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]
        return firingrate


class FieldOfViewAVCs(AgentVectorCells):
    """FieldOfViewAVCs are collection of object vector cells organised so as to represent
    the local field of view i.e. what agents the agent can "see" in the local vicinity. They work as follow:

    General FieldOfView cell description (also applies to FieldOfViewOVCs and BVCs):
        Please see FieldOfViewBVCs doc string
    
    Users should specify the agent type they are selective for with the "Other_Agent" argument. 
    """

    default_params = {
        "distance_range": [0.02, 0.4],  # min and max distances the agent can "see"
        "angle_range": [0,75],  # angluar FoV in degrees (will be symmetric on both sides, so give range in 0 (forwards) to 180 (backwards)
        "spatial_resolution": 0.02,  # resolution of each BVC tiling FoV
        "beta": 5, # smaller means larger rate of increase of cell size with radius in hartley type manifolds
        "cell_arrangement": "diverging_manifold",  # whether all cells have "uniform" receptive field sizes or they grow ("hartley") with radius.
    }

    def __init__(self,Agent,Other_Agent,params={}):

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)


        self.params["reference_frame"] = "egocentric"

        assert self.params["cell_arrangement"] is not None, "cell_arrangement must be set for FOV Neurons"

        super().__init__(Agent,Other_Agent,self.params)


    

      
class HeadDirectionCells(Neurons):
    """The HeadDirectionCells class defines a population of head direction cells. This class is a subclass of Neurons() 
    and inherits it properties/plotting functions.

    Must be initialised with an Agent and a 'params' dictionary.

    HeadDirectionCells defines a set of 'n' head direction cells. Each cell has a preffered direction/angle 
    (default evenly spaced across unit circle). In 1D there are always only n=2 cells preffering left and right directions. 
    The firing rates are scaled such that when agent travels exactly along the preferred direction the firing rate of that 
    cell is the max_fr. The firing field of a cell is a von mises centred around its preferred direction of default 
    width 30 degrees (can be changed with parameter params["angular_spread_degrees"])

    To print/set preffered direction: self.preferred_angles

    List of functions:
        • get_state()

    default_params = {
            "min_fr": 0,
            "max_fr": 1,
            "n":10,
            "angle_spread_degrees":30,
            "name": "HeadDirectionCells",
        }
    """

    default_params = {
        "min_fr": 0,
        "max_fr": 1,
        "n": 10,
        "angular_spread_degrees": 45,  # width of HDC preference function (degrees)
        "name": "HeadDirectionCells",
    }

    def __init__(self, Agent, params={}):
        """
        Initialise HeadDirectionCells(), takes as input a parameter dictionary. Any values not provided by the 
        params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        if self.Agent.Environment.dimensionality == "2D":
            self.n = self.params["n"]
            self.preferred_angles = np.linspace(0, 2 * np.pi, self.n + 1)[:-1]
            # self.preferred_directions = np.array([np.cos(angles),np.sin(angles)]).T #n HDCs even spaced on unit circle
            self.angular_tunings = np.array(
                [self.params["angular_spread_degrees"] * np.pi / 180] * self.n
            )
        if self.Agent.Environment.dimensionality == "1D":
            if "n" in params.keys() and params["n"] != 2:
                warnings.warn(f"Ignoring 'n' parameter value ({params['n']}) that was passed for {self.params['name']}. Only 2 head direction cells are needed for a 1D environment.")
            self.n = 2  # one left, one right
        self.params["n"] = self.n
        super().__init__(Agent, self.params)
        if ratinabox.verbose is True:
            print(
                f"HeadDirectionCells successfully initialised. Your environment is {self.Agent.Environment.dimensionality}, you have {self.n} head direction cells"
            )

    def get_state(self, evaluate_at="agent", use_velocity=False, **kwargs):
        """In 2D n head direction cells encode the head direction of the animal. By default velocity 
        (which determines head direction) is taken from the agent but this can also be passed as a kwarg 'head_direction'
        
        If use_velocity is True, will use the normalised velocity of the agent instead of the head direction. In this instance uset he kwarg "velocity" to manually passin a velocity vector."""
        
        #Head direction uses actual head direction of agent
        if use_velocity is False: 
            if evaluate_at == "agent":
                direction = self.Agent.head_direction
            elif "head_direction" in kwargs.keys(): #overrides "agent" hd if provided
                direction = kwargs["head_direction"]
            elif "vel" in kwargs.keys():
                direction = np.array(kwargs["vel"])
                warnings.warn("'vel' kwarg deprecated in favour of 'head_direction'")
            else:
                print("HeadDirection cells need a head direction but you didn't pass one. Taking ",end="")
                if self.Agent.Environment.dimensionality == "2D":
                    direction = np.array([1, 0])
                    print("[1,0] as default",end="")
                if self.Agent.Environment.dimensionality == "1D":
                    direction = np.array([1])
                    print("[1] as default",end="")
                print("Recommended to pass one in the 'head_direction' argument of get_state()")
        # Head direction uses normalised velocity of agent
        elif use_velocity is True:
            if evaluate_at == "agent":
                vel = self.Agent.velocity
            elif "velocity" in kwargs.keys():
                vel = kwargs["velocity"]
            else:
                print("HeadDirection cells need a velocity but you didn't pass one. Taking ",end="")
                if self.Agent.Environment.dimensionality == "2D":
                    vel = np.array([1, 0])
                    print("[1,0] as default",end="")
                if self.Agent.Environment.dimensionality == "1D":
                    vel = np.array([1])
                    print("[1] as default",end="")
                print("Recommended to pass one in the 'velocity' argument of get_state()")
            direction = vel / np.linalg.norm(vel)
        
        if self.Agent.Environment.dimensionality == "1D":
            hdleft_fr = max(0, np.sign(direction[0]))
            hdright_fr = max(0, -np.sign(direction[0]))
            firingrate = np.array([hdleft_fr, hdright_fr])
        if self.Agent.Environment.dimensionality == "2D":
            current_angle = utils.get_angle(direction)
            firingrate = utils.von_mises(
                current_angle, self.preferred_angles, self.angular_tunings, norm=1
            )

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )  # scales from being between [0,1] to [min_fr, max_fr]

        #Since these cells are not position selective, just tile the array to be the same size as the position array
        if evaluate_at == "all":
            pos_shape = self.Agent.Environment.flattened_discrete_coords.shape[0]
        elif "pos" in kwargs.keys():
            pos_shape = kwargs["pos"].shape[0]
        else:
            pos_shape = 1
        firingrate = np.tile(firingrate, (pos_shape, 1)).T

        return firingrate

                         
            
        
    def plot_HDC_receptive_field(
        self, chosen_neurons="all", fig=None, ax=None, autosave=None
    ):
        """Plots the receptive fields, in polar coordinates, of hte head direction cells. The receptive field 
        is a von mises function centred around the preferred direction of the cell. Note this only differs from the global `Neurons.plot_angular_rate_map()` function in that it analytically plots a von mises, rather than numerically averaging over positions for various head directions.

        Args:
            chosen_neurons (str, optional): The neurons to plot. Defaults to "all".
            fig, ax (_type_, optional): matplotlib fig, ax objects ot plot onto (optional).
            autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`. 
                                       Defaults to None in which case looks for global constant ratinabox.autosave_plots

        Returns:
            fig, ax
        """
        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)
        if fig is None and ax is None:
            fig, ax = plt.subplots(
                1,
                len(chosen_neurons),
                figsize=(2 * len(chosen_neurons), 2),
                subplot_kw={"projection": "polar"},
            )

        for i, n in enumerate(chosen_neurons):
            theta = np.linspace(0, 2 * np.pi, 100)
            pref_theta = self.preferred_angles[n]
            sigma_theta = self.angular_tunings[n]
            fr = utils.von_mises(theta, pref_theta, sigma_theta, norm=1)
            fr = fr * (self.max_fr - self.min_fr) + self.min_fr
            ax[i].plot(theta, fr, linewidth=2, color=self.color, zorder=11)
            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax[i].fill_between(theta, fr, 0, color=self.color, alpha=0.2)
            ax[i].set_ylim([0, self.max_fr])
            ax[i].tick_params(pad=-18)
            ax[i].set_xticklabels(["E", "N", "W", "S"])

        ratinabox.utils.save_figure(fig, self.name + "_ratemaps", save=autosave)

        return fig, ax


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

    default_params = {
        "min_fr": 0,
        "max_fr": 1,
        "name": "VelocityCells",
    }

    def __init__(self, Agent, params={}):
        """Initialise VelocityCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""
        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        self.one_sigma_speed = self.Agent.speed_mean + self.Agent.speed_std

        super().__init__(Agent, self.params)

        if ratinabox.verbose is True:
            print(
                f"VelocityCells successfully initialised. Your environment is {self.Agent.Environment.dimensionality} and you have {self.n} velocity cells"
            )
        return

    def get_state(self, evaluate_at="agent", **kwargs):
        """Takes firing rate of equivalent set of head direction cells and scales by how fast teh speed is realtive to one_sigma_speed (likely rough maximum speed)"""

        HDC_firingrates = super().get_state(evaluate_at, use_velocity=True, **kwargs)
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

    default_params = {
        "min_fr": 0,
        "max_fr": 1,
        "name": "SpeedCell",
    }

    def __init__(self, Agent, params={}):
        """Initialise SpeedCell(), takes as input a parameter dictionary, 'params'. Any values not provided by the params dictionary are taken from a default dictionary below.
        Args:
            params (dict, optional). Defaults to {}."""

        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Agent, self.params)

        if "n" in params.keys() and params["n"] != 1:
            warnings.warn(f"Ignoring 'n' parameter value ({params['n']}) that was passed for {self.name}. Only 1 speed cell is needed.")
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

    
    The activation function defaults to linear but can be set with the "activation_function" parameter. Users have two options: 
        • PREMADE (default): if a dictionary is passed this must contain parameters for the RatInABox premade activation functions stored in utils.activate(), one key in the dictionary is always "activation" from specifying the function-type whilst the other keys should give parameters of the activation function. For example:
            • {"activation":"linear", } #no additional params, DEFAULT
            • {"activation":"sigmoid", "max_fr":1, "min_fr":0, "mid_x":1, "width":2}
            • {"activation":"relu", "gain":1, "threshold":0}
            • {"activation":"tanh", "gain":1, "threshold":0}
            • {"activation":"retanh", "gain":1, "threshold":0}
            • {"activation":"softmax", "gain":1, "threshold":0}
        • BESPOKE: Users can pass their own handmade activation function, it must be vectorised and of the following format: 
            activation_func(x, deriv=False):
                if deriv == False: 
                    return φ(x)
                elif deriv == True:
                    return φ'(x)
    
    Check that the input layers are all named differently.
    List of functions:
        • get_state()
        • add_input()

    default_params = {
            "n": 10,
            "input_layers": [],  # a list of input layers, or add one by one using self.adD_inout
            "activation_function": {"activation": "linear",}, 
            "name": "FeedForwardLayer",
        }
    """

    default_params = {
        "n": 10,
        "input_layers": [],  # a list of input layers, or add one by one using self.add_input
        "activation_function": {"activation": "linear"}, #dict or function, explained in doc string
        "name": "FeedForwardLayer",
        "biases": None,  # an array of biases, one for each neuron
    }

    def __init__(self, Agent, params={}):
        self.Agent = Agent

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        #deprecation warning for activation_params --> activation_function
        if "activation_params" in self.params.keys():
            warnings.warn("The parameter 'activation_params' is deprecated. Use 'activation_function' instead.")
            self.params["activation_function"] = self.params["activation_params"]
    
        super().__init__(Agent, self.params)

        assert isinstance(
            self.input_layers, list
        ), "param['input_layers'] must be a list."
        if len(self.input_layers) == 0:
            warnings.warn(
                "No input layers have been provided. Either hand them in in the params dictionary params['input_layers']=[list,of,inputs] or use self.add_input_layer() to add them manually."
            )

        # if activation_function is a dict
        if isinstance(self.activation_function, dict):
            other_args = copy.deepcopy(self.activation_function)
            activation = other_args['activation']
            #This is a wrapper on utils.activate to set activation and other_args arguments and then return a function which can be called like self.activation_function(x, deriv) 
            def lambda_activation_function(activation, other_args):
                #Here x and deriv don't matter as they are 
                return lambda x, deriv: utils.activate(x, activation, deriv, other_args) 
            #Now make the callable function
            self.activation_function = lambda x, deriv : utils.activate(x, activation, deriv, other_args)
        else:
            # self.activation_function is already a function, passed by the user, of form             activation_func(x, deriv=False):
            pass

        self.inputs = {}
        for input_layer in self.input_layers:
            self.add_input(input_layer)

        if self.biases is None:
            self.biases = np.zeros(self.n)

        self.firingrate_prime = np.zeros_like(
            self.firingrate
        )  # this will hold the firingrate except passed through the derivative of the activation func

        if ratinabox.verbose is True:
            print(
                f"FeedForwardLayer initialised with {len(self.inputs.keys())} layers. To add another layer use FeedForwardLayer.add_input_layer().\nTo set the weights manually edit them by changing self.inputs['layer_name']['w']"
            )

    def add_input(self, input_layer, w=None, w_init_scale=1, recurrent=False, **kwargs):
        """Adds an input layer to the class. Each input layer is stored in a dictionary of self.inputs. Each has an associated matrix of weights which are initialised randomly.

        Note the inputs are stored in a dictionary. The keys are taken to be the name of each layer passed (input_layer.name). Make sure you set this correctly (and uniquely).

        Args:
            • input_layer (_type_): the layer intself. Must be a Neurons() class object (e.g. can be PlaceCells(), etc...).
            • w: the weight matrix. If None these will be drawn randomly, see next argument.
            • w_init_scale: initial weights drawn from zero-centred gaussian with std w_init_scale / sqrt(N_in)
            • recurrent: if True the input layer introduces recurrency. This can be taken into account when computing groundtruth rate maps to avoid infinite recursion. If a circuit includes a recurrent loop, mark the minimum necessary number of input connections as recurrent to prevent infinite looping.
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
                    f"There already exists a layer called {name}. This may be because you have two input players with the same attribute `InputLayer.name`. Overwriting it now."
                )
        self.inputs[name] = {}
        self.inputs[name]["layer"] = input_layer
        self.inputs[name]["w"] = w
        self.inputs[name]["w_init"] = w.copy()
        self.inputs[name]["I"] = I #stores the last input to this layer, only updated when "evaluate_at"=="last" i.e. not during plotting 
        self.inputs[name]["n"] = input_layer.n  # a copy for convenience
        self.inputs[name]["recurrent"] = recurrent
        for key, value in kwargs.items():
            self.inputs[name][key] = value
        if ratinabox.verbose is True:
            print(
                f'An input layer called {name} was added. The weights can be accessed with "self.inputs[{name}]["w"]"'
            )

    def get_state(self, evaluate_at="last", max_recurrence=None, **kwargs):
        """Returns the firing rate of the feedforward layer cells. By default this layer uses the last saved firingrate from its input layers. Alternatively evaluate_at and kwargs can be set to be anything else which will just be passed to the input layer for evaluation.
        Once the firing rate of the inout layers is established these are multiplied by the weight matrices and then activated to obtain the firing rate of this FeedForwardLayer.

        Args:
            evaluate_at (str, optional). Defaults to 'last'.
            max_recurrence: The maximum number of time get_state() recursively calls recurrent inputs (prevents infinite recursion error).  
            **kwargs: any extra kwargs will get passed to the input layer get_state() call for evaluation.
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
            pass_max_recurrence = max_recurrence
            if max_recurrence is not None and inputlayer['recurrent']:
                if max_recurrence <= 0:
                    continue
                pass_max_recurrence = max_recurrence - 1
            w = inputlayer["w"]
            if evaluate_at == "last":
                I = inputlayer["layer"].firingrate
                inputlayer["I"] = I
            else:  # kick can down the road let input layer decide how to evaluate the firingrate. this is core to feedforward layer as this recursive call will backprop through the upstraem layers until it reaches a "core" (e.g. place cells) layer which will then evaluate the firingrate.
                I = inputlayer["layer"].get_state(evaluate_at, max_recurrence=pass_max_recurrence, **kwargs)
            V += np.matmul(w, I)

        biases = self.biases
        if biases.shape != V.shape:
            biases = biases.reshape((-1, 1))
        V += biases

        # firingrate = utils.activate(V, other_args=self.activation_params)
        firingrate = self.activation_function(V, deriv=False)
        # saves current copy of activation derivative at firing rate (useful for learning rules)
        if (
            evaluate_at == "last"
        ):  # save copy of the firing rate through the dervative of the activation function
            # self.firingrate_prime = utils.activate(
            #     V, other_args=self.activation_params, deriv=True
            # )
            self.firingrate_prime = self.activation_function(V, deriv=True)

        return firingrate


    def plot_rate_map(self, method="groundtruth", max_recurrence=None, **kwargs):
        """
        If groundtruth rate maps are plotted, then a maximum recursion depth is passed.
        
        max_recurrence: The maximum number of time get_state() recursively calls recurrent inputs (prevents infinite recursion error).  
        """

        if method.startswith("groundtruth"):
            return super().plot_rate_map(method=method, max_recurrence=max_recurrence, **kwargs)
        else:
            return super().plot_rate_map(method=method, **kwargs)




class RandomSpatialNeurons(Neurons):
    """This RandomSpatialNeurons class defines a population of Neurons with smooth random spatial tunings. 
    
    The model is non-parameteric: for each neuron a smooth random function is sampled from a prior (a Gaussian process) over smooth random functions with the desired lengthscale. For a deeper understanding we recommend reading David Mackay's textbook chapter 45. Note that to be well defined over the whole domain we must sample locations _at least_ as densely than the lengthscale, so this simulation slows down for small lengthscales < 0.05. The covariance function is a squared exponential kernel and the output is passed through a sigmoid to scale it between max_fr and min_fr.

    We envisage this Neuron being used to support modelling efforts where the spatial tuning of neurons is not known or where you require spatially tuned feautres but don't want to assume place cells, grid cells etc.

    List of functions:
        • kernel() 
    """
    default_params = {'lengthscale':0.1, #lengthscale of the random function 
                      'max_fr':1, #maximum firing rate 
                      'min_fr':0, #minimum firing rate
                      'n':10, #number of neurons
                      'wall_geometry':'geodesic', #how to account for walls when calculating distance between points (only relevant in 2D)
                      'name':'RandomSpatialNeurons', #name of the class
                      }

    def __init__(self, Agent, params={}):
        """Initialise the RandomSpatialNeurons class. Takes an Agent object and a dictionary of parameters which defaults to the default_parametets dictionary."""
        
        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)
        super().__init__(Agent, params)

        if self.Agent.Environment.dimensionality == "2D":
            if self.wall_geometry == "geodesic":
                if len(self.Agent.Environment.walls) > 5:
                    print("Geodesic wall geometry only possible in environments with one or no additional walls. Using 'line_of_sight' instead. If this is slow, consider trying 'euclidean'")
                    self.wall_geometry = 'line_of_sight'

        # We densely sample locations across the environment and then distance covariance matrix using a squared exponential kernel. Using this covariance we sample targets   (according to an exponential because it is way too expensive to use the default dx=0.01 we have to initially create a lower resolution array of poionts. We will sample from the prior over these points and then, for any other points, "infer" the firing rate from these samples. For all other points we'll take the mean of the posterior based on these samples.

        assert self.lengthscale >= 0.02, "lengthscale must be greater than 0.02 m"
        self.X = self.Agent.Environment.discretise_environment(dx=min(0.05,self.lengthscale)) 
        self.X = self.X.reshape(-1, self.X.shape[-1])
        self.Q = self.kernel(self.X, self.X)
        # If doing full Bayes: Diagonalise and add a small amount of noise to the diagonal to ensure invertibility (if doing full Bayes) then precalculate some inverses 
        # self.Q = 0.5 * (self.Q + self.Q.T)
        # self.Q += 1e-6*np.eye(n_samples)
        #self.Qinv_on_target = self.Qinv @ self.targets
        #self.Qinv = np.linalg.inv(self.Q)

        #Sample targets (eqn 45.33 in Mackay textbook)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.targets = np.random.multivariate_normal(mean=np.zeros(self.Q.shape[0]), cov=self.Q, size=self.n).T # targets
        warnings.filterwarnings("default", category=RuntimeWarning)
        self.targets = ratinabox.utils.activate(self.targets, activation="sigmoid",
                                                other_args = {"max_fr": self.max_fr, "min_fr": self.min_fr, "mid_x": 0, "width_x": 2}) # activation function
        return 

    def get_state(self, evaluate_at="agent", **kwargs):
        """At inference time any position in the Environment can be queried and its firing rate will be returned. This is done by taking the mean of the targets weights by how close those targets are to the position (as measured by the covariance) - we note that this isn't the "full" Bayesian way to estimate the posterior of unseen points but since we have densely samples target locations it should be good enough. The full Bayesian way (Mackay Eqn. 45.42) is provided but commented out for now due to instabilities when there are additional walls.
        
        By default position is taken from the Agent and used to calculate firinf rates. This can also by passed directly (evaluate_at=None, pos=pass_array_of_positions) or ou can use all the positions in the environment (evaluate_at="all").

        Returns:
            firingrates: an array of firing rates"""
        if evaluate_at == "all":
            pos = self.Agent.Environment.flattened_discrete_coords
        elif evaluate_at == "agent":
            pos = self.Agent.pos
        else:
            pos = kwargs["pos"]
        pos = pos.reshape(-1, pos.shape[-1])
        k = self.kernel(pos, self.X)

        #this is the "poorman's" way of doing this. It just uses the kernel values to weights the target values and sets the firing rate to their mean. Since our targets have been estimated quite densely this is a good approximation the full Bayesian way given below but, for numerical reasons, is unstable in environments with lots of walls.
        #LOCAL AVERAGE OF TARGETS 
        k = k / np.sum(k, axis=1, keepdims=True)
        mean = k @ self.targets

        #FULL BAYES (UNSTABLE WITH WALLS) 
        # mean = k @ self.Qinv_on_target
        # return k 
        firingrate = mean.T 
        return firingrate
    

    def kernel(self, x1, x2):
        """Returns the covariance matrix of shape (len(x1),len(x2)) between all locations in lists x1 and x2. For now this is just the squared exponential kernel where the "distance" between two points is the "environmental distance" (this may account for any walls etc.). In future we may consider adding more types of kernels. 
        Args: 
            x1: (N1, D) array of N1 points in D dimensions
            x2: (N2, D) array of N2 points in D dimensions
        Returns:
            (N1, N2) kernel between all points in x1 and all points in x2 
        """
        d_mat = self.Agent.Environment.get_distances_between___accounting_for_environment(x1,x2,wall_geometry=self.wall_geometry)
        kernel_matrix = np.exp(-(d_mat**2)/(2*self.lengthscale**2))
        return kernel_matrix
