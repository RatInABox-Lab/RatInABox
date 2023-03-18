# List of `RatInABox` plotting functions: 

In this markdown we describe teh plotting functions available with `RatInABox`. The following definitions hold: 
* `Env`: a 2D `Environment()` class with a wall and solid boundary conditions
* `Env1D`: a 1D `Environment()` class with periodic boundary conditions


# `Environment`

## `Environment.plot_environment()`
Displays the environment. Works for both 1 or 2D environments. 
Examples: 
* `Env.plot_environment()` 

<img src="../.images/plotting_examples/plot_env.svg" height="150">



# `Agent`


## `Agent.plot_trajectory()`
Plots the agent trajectory. Works for 1 or 2D.

* `Ag.plot_trajectory(t_end=120)` 

<img src="../.images/plotting_examples/plot_traj.svg" height="150">

* `Ag1D.plot_trajectory(t_end=120)` 

<img src="../.images/plotting_examples/plot_traj_1D.svg" height="150">


## `Agent.animate_trajectory()`
Makes an animation of the agents trajectory. 
<img src="../.images/plotting_examples/trajectory_animation.gif" width="150">


## `Agent.plot_position_heatmap()`
Plots a heatmap of the Agents past locations (2D and 1D example shown)

<img src="../.images/plotting_examples/plot_heatmap.svg" height="150">

<img src="../.images/plotting_examples/plot_heatmap_1D.svg" height="23">





## `Agent.plot_histogram_of_speeds()` 
<img src="../.images/plotting_examples/plot_histogram_speed.svg" height="150">




## `Agent.plot_histogram_of_rotational_velocities()` 
<img src="../.images/plotting_examples/plot_histogram_rotvel.svg" height="150">


# `Neurons`


## `Neurons.plot_rate_timeseries()`
Plots a timeseries of the firing rates
<img src="../.images/plotting_examples/gc_plotrts.svg" height="150">

## `Neurons.plot_rate_timeseries(imshow=True)`
Plots a timeseries of the firing rates as an image 
<img src="../.images/plotting_examples/gc_plotrts_imshow.svg" height="200">

Plots a timeseries of the firing rates
<img src="../.images/plotting_examples/gc_plotrts.svg" height="150">

## `Neurons.animate_rate_timeseries()`
Makes an animation of the firing rates timeseries
<img src="../.images/plotting_examples/animate_rate_timeseries.gif" height="150">


## `Neurons.plot_ratemap()`
Depending on the parameters passed this function will either 

1. Analytically calculate the rate map (`method = 'analytic'`),
2. Infer the rate map from past activity (`method = 'history'`) or 
3. Plot the observed spikes (`spikes=True`). 
As an example here we show this function for a set of 3 (two dimensional) grid cells and 10 (one-dimensional) place cells. 

* `Neurons.plot_ratemap(method=`analytic`)

<img src="../.images/plotting_examples/gc_plotrm.svg" height="150">

<img src="../.images/plotting_examples/pc1d_plotrm.svg" height="150">

* `Neurons.plot_ratemap(method=`history`)

<img src="../.images/plotting_examples/gc_plotrm_history.svg" height="150">

<img src="../.images/plotting_examples/pc1d_plotrm_history.svg" height="150">

* `Neurons.plot_ratemap(method=`neither`, spikes=True)

<img src="../.images/plotting_examples/gc_plotrm_spikes.svg" height="150">

<img src="../.images/plotting_examples/pc1d_plotrm_spikes.svg" height="150">



## `PlaceCells.plot_place_cell_centres()`

Scatters where the place cells are centres 

<img src="../.images/plotting_examples/pc_locations.svg" height="150">


## `BoundaryVectorCells.plot_BVC_receptive_field()`

<img src="../.images/plotting_examples/bvc_rfs.svg" height="150">


# Other details: 

* All plotting functions return a tuple (`fig`, `ax`) of `matplotlib` figure objects. 
* Most plotting functions support being passed `fig` and `ax` and will plot whatever it is they plot ontop of that. This means we can make: 
1. Layered figures (e.g. plot a trajectory on top of a rate map): 
```python
fig, ax = Neurons.plot_rate_map(chosen_neuron="1")
fig, ax = Ag.plot_trajectory(fig=fig, ax=ax)
```

<img src="../.images/plotting_examples/trajectory_on_ratemap.svg" height="150">

2. Multipanel figures: 
```python 
fig, axes = plt.subplots(1,5,figsize=(20,4))
Ag.plot_trajectory(fig=fig,ax=axes[0])
Neurons.plot_rate_map(fig=fig,ax=[axes[1],axes[2],axes[3]],chosen_neurons='3') #<-- to plot more than 1 neuron pass an array of axes
Neurons.plot_rate_timeseries(fig=fig,ax=axes[4]) 
```

<img src="../.images/plotting_examples/multipanel_riab.svg" height="150">

* For rate maps and timeseries' by default **all** the cells will be plotted. This may take a long time if the number of cells is large. Control this with the `chosen_neurons` argument

* We use the `matplotlib` wrapper [`tomplotlib`](https://github.com/TomGeorge1234/tomplotlib) to format and save figures. This is not necesary for the workings of `ratinabox` but note your figures might look slightly different without it.
