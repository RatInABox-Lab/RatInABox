# RatInABox 
![Tests](https://github.com/TomGeorge1234/RatInABox/actions/workflows/test.yml/badge.svg)   [![PyPI version](https://badge.fury.io/py/ratinabox.svg)](https://badge.fury.io/py/ratinabox) [![Downloads](https://static.pepy.tech/badge/ratinabox)](https://pepy.tech/project/ratinabox)

`RatInABox` (see [paper](https://www.biorxiv.org/content/10.1101/2022.08.10.503541v3)) is a toolkit for generating locomotion trajectories and complementary neural data for spatially and/or velocity selective cell types in complex continuous environments. 

[**Install**](#installing-and-importing) | [**Demos**](#get-started) | [**Features**](#feature-run-down) | [**Contributions and Questions**](#contribute) | [**Cite**](#cite)

<img src=".images/readme/ratinabox.gif" width=850>

With `RatInABox` you can: 

* **Generate realistic trajectories** for rats exploring complex 1 and 2D environments under a smooth random policy, an external control signal, or your own trajectory data.
* **Generate artificial neuronal data** for various location- or velocity-selective cells found in the Hippocampal formation, or build your own more complex cell types. 
* **Build and train complex multi-layer networks** of cells, powered by data generated with `RatInABox`. 

`RatInABox` is an open source project welcoming [contributions](#contribute). If you use `RatInABox` please [cite](#cite) the paper and consider giving this repository a star ☆. It contains three classes: 

1. `Environment`📦: The environment/maze (or "box") that the agent lives in. 1- or 2-dimensional.
2. `Agent`      🐀: The agent (or "rat") moving around the `Environment`. 
3. `Neurons`    🧠: A population of neurons with firing rates determined by the state (position and velocity) of the `Agent`. Make your own or use one of our premade cell types including: 
    * `PlaceCells`
    * `GridCells`
    * `BoundaryVectorCells` (egocentric or allocentric)
    * `ObjectVectorCells`
    * `VelocityCells`
    * `SpeedCells`
    * `HeadDirectionCells`
    * `FeedForwardLayer` (a generic class analagous to a feedforward layer in a deep neural network)
    * ...

The top animation shows an example use case: an `Agent` randomly explores a 2D `Environment` with a wall. Three populations of `Neurons` (`PlaceCells`, `GridCells`, `BoundaryVectorCells`) fire according to the receptive fields shown. All data is saved into the history for downstream use. `RatInABox` is fully continuous is space; this means that position and neuronal firing rates are calculated rapidly online with float precision rather than pre-calculated over a discretised mesh. `RatInABox` is flexibly discretised in time; `dt` can be set by the user (defaulting to 10 ms) depending on requirements.


## Key features
* **Non-specific**: Trajectories can be randomly generated, imported, or adaptively controlled making `RatInABox` a powerful engine for many tasks involving continuous motion (e.g. control theory or [reinforcement learning](#policy-control)). 
* **Biological**:   Simulate large populations of spatially and/or velocity modulated cell types. Neurons can be rate based or spiking. The random motion model is fitted to match real rodent motion. 
* **Flexible**:     Simulate environments in 1D or 2D with arbitrarily wall arrangements.  Combine premade or bespoke `Neurons` classes into arbitrary deep networks (examples given).
* **Fast**:         Simulating 1 minute of exploration in a 2D environment with 100 place cells (dt=10 ms) take just 2 seconds on a laptop (no GPU needed).
* **Precise**:      No more prediscretised positions, tabular state spaces, or jerky movement policies. It's all continuous. 
* **Easy**:         Sensible default parameters mean you can have realisitic simulation data to work with in ~10 lines of code.
* **Visual**        Plot or animate trajectories, firing rate timeseries', spike rasters, receptive fields, heat maps, velocity histograms...using the plotting functions ([summarised here](./demos/list_of_plotting_fuctions.md)). 


## Get started 
Many [demos](./demos/) are provided. Reading through the [example scripts](#example-scripts) (one simple and one extensive, duplicated at the bottom of the readme) these should be enough to get started. We also provide numerous interactive jupyter scripts as more in-depth case studies; for example one where `RatInABox` is used for [reinforcement learning](./demos/reinforcement_learning_example.ipynb), another for [neural decoding](./demos/decoding_position_example.ipynb) of position from firing rate. Jupyter scripts reproducing all figures in the [paper](./demos/paper_figures.ipynb) and [readme](./demos/readme_figures.ipynb) are also provided. All [demos](./demos/) can be run on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./demos/)

## Installing and Importing
**Requirements** are minimal (`python3`, `numpy`, `scipy` and `matplotlib`, listed in `setup.cfg`) and will be installed automatically. 

**Install** the latest, stable version using `pip` at the command line with
```console
$ pip install ratinabox
```
Alternatively, in particular if you would like to develop `RatInABox` code or if you want the bleeding edge (may occasioanlly break), install from this repo using  
```console
$ git clone --depth 1 https://github.com/TomGeorge1234/RatInABox.git
$ cd RatInABox
$ pip install -e .
```
n.b. the "editable" `-e` handle means changes made to your clone will be reflected when you next import `RatInABox` into your code.

**Import** into your python project with  
```python
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells #...
```

## Feature run-down
Here is a list of features loosely organised into those pertaining to 

(i) the [`Environment`](#i-environment-features)
* [Adding walls](#walls)
* [Polygon-shaped Environments](#polygon-shaped-environments)
* [Holes](#holes)
* [Boundary conditions](#boundary-conditions)
* [1- or 2-dimensions](#1--or-2-dimensions) 

(ii) the [`Agent`](#ii-agent-features)
* [Random motion](#random-motion-model)
* [Importing trajectories](#importing-trajectories)
* [Policy control](#policy-control)
* [Wall repelling](#wall-repelling)
* [Advanced `Agent` classes](#advanced-agent-classes)

(iii) the [`Neurons`](#iii-neurons-features).
* [Cell types](#multiple-cell-types) 
* [Noise](#noise)
* [Spikes vs rates](#spiking)
* [Plotting rate maps](#rate-maps)
* [Place cell models](#place-cell-models) 
* [Place cell geometry](#geometry-of-placecells)
* [Egocentric encodings](#egocentric-encodings)
* [Deep neural networks](#more-complex-neuron-types-and-networks-of-neurons)

(iv) [Figures and animations plotting](#iv-figures-and-animations)

Specific details can be found in the [paper](https://www.biorxiv.org/content/10.1101/2022.08.10.503541v3). 



### (i) `Environment` features
#### Walls 
Arbitrarily add walls to the environment to produce arbitrarily complex mazes:
```python 
Environment.add_wall([[x0,y0],[x1,y1]])
```
Here are some easy to make examples.

<img src=".images/readme/walls.png" width=1000>


#### Complex `Environment`s: Polygons, curves, and holes
By default, `Environments` in RatInABox are square (or rectangular if `aspect != 1`). It is possible to create arbitrary environment shapes using the `"boundary"` parameter at initialisation. 

One can all add holes to the `Environment` using the `"holes"` parameter at initialisation. Positions sampled from the Environment (e.g. at initialisation) won't be inside holes.

Any curved environments can be made by creating a boundary of many small walls (uyse sparingly, walls may slow down computations)

```python 
#A trapezium shaped Environment
Env = Environment(params={
    'boundary':[[0,-0.2],[0,0.2],[1.5,0.5],[1.5,-0.5]],
    })

#An environment with two holes makign a figure of 8
Env = Environment(params={
    'aspect':1.8,
    'holes' : [[[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]],
               [[1,0.2],[1.6,0.2],[1.6,0.8],[1,0.8]]],
    })

#A circular environment made from many small walls
Env = Environment(params = {
    'boundary':[[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)],
    })
```


<img src=".images/readme/complex_envs.png" width=1000>



#### Boundary conditions 
Boundary conditions (for default square/rectangular environments) can be "periodic" or "solid". Place cells and the motion of the Agent will respect these boundaries accordingly. 
```python
Env = Environment(
    params = {'boundary_conditions':'periodic'} #or 'solid' (default)
) 
```

<img src=".images/readme/boundary_conditions.png" width=500>

#### 1- or 2-dimensions 
`RatInABox` supports 1- or 2-dimensional `Environment`s. Almost all applicable features and plotting functions work in both. The following figure shows 1 minute of exploration of an `Agent` in a 1D environment with periodic boundary conditions spanned by 10 place cells. 
```python 
Env = Environment(
    params = {'dimensionality':'1D'} #or '2D' (default)
) 
```

<img src=".images/readme/one_dimension.png" width=500>



### (ii) `Agent` features

#### Random motion model
By defaut the `Agent` follows a random motion policy.  Random motion is stochastic but smooth. The speed (and rotational speed, if in 2D) of an Agent take constrained random walks governed by Ornstein-Uhlenbeck processes. You can change the means, variance and coherence times of these processes to control the shape of the trajectory. Default parameters are fit to real rat locomotion data from Sargolini et al. (2006): 

<img src=".images/readme/riab_vs_sargolini.gif" width=500>

The default parameters can be changed to obtain different style trajectories. The following set of trajectories were generated by modifying the rotational speed parameter `Agent.rotational_velocity_std`:

```python
Agent.speed_mean = 0.08 #m/s
Agent.speed_coherence_time = 0.7
Agent.rotation_velocity_std = 120 * np.pi/180 #radians 
Agent.rotational_velocity_coherence_time = 0.08
```

<img src=".images/readme/motion_model.png" width=800>


#### Importing trajectories
`RatInABox` supports importing external trajectory data (rather than using the in built random motion policy). Imported data can be of low temporal resolution. It will be smoothly upsampled using a cubic splines interpolation technique. We provide a 10 minute trajectory from the open-source data set of Sargolini et al. (2006) ready to import. In the following figure blue shows (low resolution) trajectory data imported into an `Agent` and purple shows the smoothly upsampled trajectory taken by the `Agent` during exploration. 
```python
Agent.import_trajectory(dataset='sargolini')
#or 
Agent.import_trajectory(times=array_of_times,
                        positions=array_of_positions)

```

<img src=".images/readme/imported_trajectory.png" width=200>

#### Policy control 
By default the movement policy is an random and uncontrolled (e.g. displayed above). It is possible, however, to manually pass a "drift_velocity" to the Agent on each `Agent.update()` step. This 'closes the loop' allowing, for example, Actor-Critic systems to control the Agent policy. As a demonstration that this method can be used to control the agent's movement we set a radial drift velocity to encourage circular motion. We also use RatInABox to perform a simple model-free RL task and find a reward hidden behind a wall (the full script is given as an example script [here](./demos/reinforcement_learning_example.ipynb))
```python
Agent.update(drift_velocity=drift_velocity)
```

<img src=".images/readme/motion.gif" width=600>

#### Wall repelling 
Under the random motion policy, walls in the environment mildly "repel" the `Agent`. Coupled with the finite turning speed this replicates an effect (known as thigmotaxis, sometimes linked to anxiety) where the `Agent` is biased to over-explore near walls and corners (as shown in these heatmaps) matching real rodent behaviour. It can be turned up or down with the `thigmotaxis` parameter.
```python 
Αgent.thigmotaxis = 0.8 #1 = high thigmotaxis (left plot), 0 = low (right)
```

<img src=".images/readme/wall_repel.png" width=900>


#### Advanced `Agent` classes
One can make more advanced Agent classes, for example `ThetaSequenceAgent()` where the position "sweeps" (blue) over the position of an underlying true (regular) `Agent()` (purple), highly reminiscent of theta sequences observed when one decodes position from the hippocampal populaton code on sub-theta (10 Hz) timescales. This class can be found in the [`contribs`](./ratinabox/contribs/) directory. 

<img src=".images/readme/theta_sequences.gif" width=350>


### (iii) `Neurons` features 

#### Multiple cell types: 
We provide a list of premade `Neurons` subclasses. These include: 

* `PlaceCells` 
* `GridCells`
* `BoundaryVectorCells` (can be egocentric or allocentric)
* `ObjectVectorCells` (can be used as visual cues, i.e. only fire when `Agent` is looking towards them)
* `HeadDirectionCells`
* `VelocityCells`
* `SpeedCells`
* `FeedForwardLayer` - calculates activated weighted sum of inputs from a provide list of input `Neurons` layers.
* `FieldOfViewNeurons` - Egocentric encoding of what the `Agent` can see 

This last class, `FeedForwardLayer` deserves special mention. Instead of its firing rate being determined explicitly by the state of the `Agent` it summates synaptic inputs from a provided list of input layers (which can be any `Neurons` subclass). This layer is the building block for how more complex networks can be studied using `RatInABox`. 


#### Noise 
Use the `Neurons.noise_std` and `Neurons.noise_coherence_time` parameters to control the amount of noise (Hz) and autocorrelation timescale of the noise (seconds). For example (work with all `Neurons` classes, not just `PlaceCells`): 

```python
PCs = PlaceCells(Ag,params={
    'noise_std':0.1, #defaults to 0 i.e. no noise
    'noise_coherence_time':0.5, #autocorrelation timescale of additive noise vector 
})
```

<img src=".images/readme/noise.png" width="1000">

#### Spiking 
All neurons are rate based. However, at each update spikes are sampled as though neurons were Poisson neurons. These are stored in `Neurons.history['spikes']`. The max and min firing rates can be set with `Neurons.max_fr` and  `Neurons.min_fr`.
```
Neurons.plot_ratemap(spikes=True)
```

<img src=".images/readme/spikes.png" width="1000">


#### Rate maps 
`PlaceCells`, `GridCells` and allocentric `BoundaryVectorCells` (among others) have firing rates which depend exclusively on the position of the agent. These rate maps can be displayed by querying their firing rate at an array of positions spanning the environment, then plotting. This process is done for you using the function `Neurons.plot_rate_map()`. 

More generally, however, cells firing is not only determined by position but potentially other factors (e.g. velocity, or historical effects if the layer is part of a recurrent network). In these cases the above method of plotting rate maps will necessarily fail. A more robust way to display the receptive field is to plot a heatmap of the positions of the Agent has visited where each positions contribution to a bin is weighted by the firing rate observed at that position. Over time, as coverage become complete, the firing fields become visible.
```
Neurons.plot_rate_map() #attempts to plot "ground truth" rate map 
Neurons.plot_rate_map(method="history") #plots rate map by firing-rate-weighted position heatmap
``` 

<img src=".images/readme/rate_map.png" width=600>


#### Place cell models

Place cells come in multiple types (given by `params['description']`), or it would be easy to write your own:
* `"gaussian"`: normal gaussian place cell 
* `"gaussian_threshold"`: gaussian thresholded at 1 sigma
* `"diff_of_gaussians"`: gaussian(sigma) - gaussian(1.5 sigma)
* `"top_hat"`: circular receptive field, max firing rate within, min firing rate otherwise
* `"one_hot"`: the closest palce cell to any given location is established. This and only this cell fires. 

This last place cell type, `"one_hot"` is particularly useful as it essentially rediscretises space and tabularises the state space (gridworld again). This can be used to  contrast and compare learning algorithms acting over continuous vs discrete state spaces. This figure compares the 5 place cell models for population of 9 place cells (top left shows centres of place cells, and in all cases the `"widths"` parameters is set to  0.2 m, or irrelevant in the case of `"one_hot"`s)

<img src=".images/readme/placecellmodels.png" width=800>

These place cells (with the exception of `"one_hot"`s) can all be made to phase precess by instead initialising them with the `PhasePrecessingPlaceCells()` class currently residing in the `contribs` folder. This figure shows example output data. 

<img src=".images/readme/phaseprecession.png" width=500>


#### Geometry of `PlaceCells` 
Choose how you want `PlaceCells` to interact with walls in the `Environment`. We provide three types of geometries.  

<img src=".images/readme/wall_geometry.png" width=900>


#### Egocentric encodings
Most `RatInABox` cell classes are allocentric (e.g. `PlaceCells`, `GridCells` etc. do not depend on the agents point of view) not egocentric. `BoundaryVectorCells` (BVCs) and `ObjectVectorCells` (OVCs) can be either. `FieldOfViewNeurons` exploit this by arranging sets of egocentric BVC or OVCs to tile to agents local field of view creating a comprehensive egocentric encoding of what boundaries or objects the agent can 'see' from it's current point of view. A custom plotting function displays the tiling and the firing rates as shown below. With an adequately defined field of view these can make, for example, "whisker cells". 

```python
FoV_BVCs = FieldOfViewNeurons(Ag)
FoV_OVCs = FieldOfViewNeurons(Ag,params={
    'cell_type':'OVC',
    })
FoV_whiskers = FieldOfViewNeurons(Ag,params={
    "FoV_angles":[75,105],
    "FoV_distance":[0.1,0.2],
    "spatial_resolution":0.02,})
```

<img src=".images/readme/field_of_view.gif" width=600>



#### More complex Neuron types and networks of Neurons
We encourage users to create their own subclasses of `Neurons`. This is easy to do, see comments in the `Neurons` class within the [code](./ratinabox/Neurons.py) for explanation. By forming these classes from the parent `Neurons` class, the plotting and analysis features described above remain available to these bespoke Neuron types. Additionally we provide a `Neurons` subclass called `FeedForwardLayer`. This neuron sums inputs from any provied list of other `Neurons` classes and can be used as the building block for constructing complex multilayer networks of `Neurons`, as we do [here](./demos/path_integration_example.ipynb) and [here](./demos/reinforcement_learning_example.ipynb). 



### (iv) Figures and animations 
`RatInABox` is built to be highly visual. It is easy to plot or animate data and save these plots/animations. Here are some tips

#### Saving
* `ratinabox.figure_directory` a global variable specifying the directory into which figures/animations will be saved 
* `ratinabox.utils.save_figure(fig,fig_name)` saves a figure (or animation) into a dated folder within the figure directory  as both `".svg"` and `".png"` (`".mp4"` or `".gif"`). The current time will be appended to the `fig_name` so you won't ever overwrite. 


#### Saving (but automatically)
* Setting `ratinabox.autosave_plots = True` means RatInABox figure will be automatically saved in the figure directory without having to indvidually call the `utils` function above. 

#### Styling
* `ratinabox.stylize_plots()` this call sets some global matplotlib rcParams to make plots look pretty/exactly like they do in this repo


#### Most important plotting functions
The most important plotting functions are (see source code for the available arguments/kwargs):

```python
Environment.plot_environment() #visualises current environment with walls and objects
Agent.plot_trajectory() #plots trajectory
Agent.animate_trajectory() #animate trajectory
Neurons.plot_rate_map() # plots the rate map of the neurons at all positions
Neurons.plot_rate_timeseries() # plots activities of the neurons over time 
Neurons.animate_rate_timeseries() # animates the activity of the neurons over time 
```

Most plotting functions accept `fig` and `ax` as optional arguments and if passed will plot ontop of these. This can be used to make comolex or multipanel figures. For a comprehensive list of plotting functions see [here](./demos/list_of_plotting_fuctions.md). 

## Example Scripts
In the folder called [demos](./demos/) we provide numerous script and demos which will help when learning `RatInABox`. In approximate order of complexity, these include:
* [simple_example.ipynb](./demos/simple_example.ipynb): a very simple tutorial for importing RiaB, initialising an Environment, Agent and some PlaceCells, running a brief simulation and outputting some data. Code copied here for convenience.
```python 
import ratinabox #IMPORT 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
#INITIALISE CLASSES
Env = Environment() 
Ag = Agent(Env)
PCs = PlaceCells(Ag)
#EXPLORE
for i in range(int(20/Ag.dt)): 
    Ag.update()
    PCs.update()
#ANALYSE/PLOT
print(Ag.history['pos'][:10]) 
print(PCs.history['firingrate'][:10])
fig, ax = Ag.plot_trajectory()
fig, ax = PCs.plot_rate_timeseries()
```
* [extensive_example.ipynb](./demos/extensive_example.ipynb): a more involved tutorial. More complex enivornment, more complex cell types and more complex plots are used. 
* [list_of_plotting_functions.md](./demos/list_of_plotting_fuctions.md): All the types of plots available for are listed and explained. 
* [readme_figures.ipynb](./demos/readme_figures.ipynb): (Almost) all plots/animations shown in the root readme are produced from this script (plus some minor formatting done afterwards in powerpoint).
* [paper_figures.ipynb](./demos/paper_figures.ipynb): (Almost) all plots/animations shown in the paper are produced from this script (plus some major formatting done afterwards in powerpoint).
* [decoding_position_example.ipynb](./demos/decoding_position_example.ipynb): Postion is decoded from neural data generated with RatInABox. Place cells, grid cell and boundary vector cells are compared. 
* [reinforcement_learning_example.ipynb](./demos/reinforcement_learning_example.ipynb): RatInABox is use to construct, train and visualise a small two-layer network capable of model free reinforcement learning in order to find a reward hidden behind a wall. 
* [path_integration_example.ipynb](./demos/path_integration_example.ipynb): RatInABox is use to construct, train and visualise a large multi-layer network capable of learning a "ring attractor" capable of path integrating a position estimate using only velocity inputs.

## Contribute 
`RatInABox` is an open source project, and we actively encourage community contributions, for example bug fixes, new cells types, new features, new plotting functions, new motion datasets, documentation, citations of relevant work, or additional experiment notebooks. Typically the best way to go about this is by opening an issue or feel free to make a pull request. 

We have a dedicated [contribs](./ratinabox/contribs/) directory where you can safely add awesome scripts and new `Neurons` classes etc.

*Questions?* If you can't figure out how something works from the readme, demos, code comments etc. then ask! Open an issue, I'm usually pretty quick to respond. Here's our [official theme tune](https://www.youtube.com/watch?v=dY-FOI-9SOE) by the way. 



## Cite
If you use `RatInABox` in your research or educational material, please cite the work as follows: 
Bibtex:
```
@article{ratinabox2022,
	doi = {10.1101/2022.08.10.503541},
	url = {https://doi.org/10.1101%2F2022.08.10.503541},
	year = 2022,
	month = {aug},
	publisher = {Cold Spring Harbor Laboratory},
	author = {Tom M George and William de Cothi and Claudia Clopath and Kimberly Stachenfeld and Caswell Barry},
	title = {{RatInABox}: A toolkit for modelling locomotion and neuronal activity in continuous environments}
}
```

Formatted:
```
Tom M George, William de Cothi, Claudia Clopath, Kimberly Stachenfeld, Caswell Barry. "RatInABox: A toolkit for modelling locomotion and neuronal activity in continuous environments" (2022).
``` 
The research paper corresponding to the above citation can be found [here](https://www.biorxiv.org/content/10.1101/2022.08.10.503541v3).