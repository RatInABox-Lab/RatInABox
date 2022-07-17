#  RatInABox 

`RatInABox` 🐀📦 (paper here) is a toolkit for simulating motion and various cell types found in the Hippocampal formation. `RatInABox` is fully continuous is space and time: position and neuronal firing rates are calculated rapidly online with float precision. With it you can:

* **Generate realistic trajectories** for rats exploring complex 1- and 2-dimensional environments under a random policy or using imported data
* **Generate artificial neuronal data** Simulate various location or velocity selective cells found in the Hippocampal-Entorhinal system, or build your own more complex cell type. 
* **Build complex networks** Build, train and analyse complex networks of cells, powered by `RatInABox`. 

<img src="./readme_figs/riab.gif">

`RatInABox` contains three classes: 

1. `Environment()`: The environment/maze (or "box") that the agent lives in. 1- or 2-dimensional.
2. `Agent()`: The agent (or "rat") moving around the `Environment`. 
3. `Neurons()`: A population of neurons with firing rates determined by the state (position and velocity) of the `Agent`. Make your own or use one of our premade cell types: 
    * `PlaceCells()`
    * `GridCells()`
    * `BoundaryVectorCells()` (egocentric or allocentric)
    * `VelocityCells()`
    * `SpeedCells()`
    * `HeadDirectionCells()`
    * `FeedForwardLayer()`

The top animation shows the kind of simulation you can easily run using this toolbox. In it an `Agent` randomly explores a 2D `Environment` with a wall. Three populations of `Neurons` (`PlaceCells`, `GridCells`, `BoundaryVectorCells`) vary their activity and "fire" as the `Agent` explores. 

## Key features

* **Flexible**: Generate arbitrarily complex environments. 
* **Biological**: Simulate large populations of spatially and/or velocity modulated cell types. Neurons can be rate based or spiking. Motion model fitted to match real rodent motion. 
* **Fast**: Simulating 1 minute of exploration in a 2D environment with 100 place cells (dt=10 ms) take just 2 seconds on a laptop (no GPU needed).
* **Precise**: No more pre-discretised positions, tabular state spaces, or jerky movement policies. It's all continuous. 
* **Visual** Plot or animate trajectories, firing rate timeseries', spike rasters, receptive fields, heat maps, velocity histograms...using the plotting functions ([summarised here](./example_scripts/list_of_plotting_fuctions.md)). 
* **Easy**: Sensible default parameters mean you can have realisitic simulation data to work with in ~10 lines of code.
* **General**: Build your own bespoke `Neurons()` classes and combine them into complex networks of neurons (example scripts given).

(a)[./here]

## Get started 
At the bottom of this readme we provide [example scripts](./example_scripts/): one simple and one extensive. Reading through this section should be enough to get started. We also provide two case studies where `RatInABox` is used in a [reinforcement learning project](./example_scripts/reinforcement_learning_example.ipynb) and a [path integration](./example_scripts/path_integration_example.ipynb) project. Jupyter scripts reproducing all figures in the [paper](./example_scripts/paper_figures.ipynb) and [readme](./example_scripts/readme_figures.ipynb) are also provided.

In addition 

## Requirements
* Python 3.7+
* Numpy
* Scipy
* Matplotlib
* Jupyter (optional)
* tqdm

## Installation 
I will streamline this soon. For now just clone the directory and get started. With `RatInABox/` in your code directory import by calling 
If you intend to meddle with the inner workings then we recommend cloning this repository into your working directory and importing `RatInABox` using
```python
sys.path.append("./RatInABox")
import ratinabox
from ratinabox import * #or just the classes/functions you need
```

Otherwise `RatInABox` can be installed using
```python
pip install git+https://github.com/TomGeorge1234/ratinabox.git
```
or clonign hte reposity and running 
```python
python setup.py install
```
## Feature run-down
Here is a list of features loosely organised into three categories: those pertaining to (i) the `Environment()`, (ii) the `Agent()` and (iii) the `Neurons()`. Specific details can be found in the paper, [here](link/to/paper/on/arxiv). 


### (i) `Environment()` features
#### Walls 
Arbitrarily add walls to the environment to produce arbitrarily complex mazes:
```python 
Environment.add_wall([[x0,y0],[x1,y1]])
```
Here are some easy to make examples.
![](./readme_figs/walls.png)

#### Boundary conditions 
Boundary conditions can be "periodic" or "solid". Place cells and the motion of the Agent will respect these boundaries accordingly. 
```python
Env = Environment(
    params = {'boundary_conditions':'periodic'} #or 'solid' (default)
) 
```
<img src="./readme_figs/boundary_conditions.png" height="200">

#### 1- or 2-dimensions 
`RatInABox` supports 1- or 2-dimensional `Environments`. Almost all applicable features and plotting functions work in both. The following figure shows 1 minute of exploration of an `Agent` in a 1D environment with periodic boundary conditions spanned by 10 place cells. 
```python 
Env = Environment(
    params = {'dimensionality':'1D'} #or '2D' (default)
) 
```
![](./readme_figs/one_dimension.png)



### (ii) `Agent()` features

#### Random motion model
Random motion is stochastic but smooth. The speed (and rotational speed, if in 2D) of an Agent take constrained random walks governed by Ornstein-Uhlenbeck processes. You can change the means, variance and coherence times of these processes to control the shape of the trajectory. Default parameters are fit to real rat locomotion data from Sargolini et al. (2006): 

<img src="./readme_figs/riab_vs_sargolini.gif" width="50%" height="50%">

The default parameters can be changed to obtain different style trajectories. The following set of trajectories were generated by modifying the rotational speed parameter `Agent.rotational_velocity_std`:

```python
Agent.speed_mean = 0.08 #m/s
Agent.speed_coherence_time = 0.7
Agent.rotation_velocity_std = 120 * np.pi/180 #radians 
Agent.rotational_velocity_coherence_time = 0.08
```
<img src="./readme_figs/motion_model.png" height="200">


#### Importing trajectories
`RatInABox` supports importing external trajectory data (rather than using the in built random motion policy). Imported data can be of low temporal resolution. It will be smoothly upsampled using a cubic splines interpolation technique. We provide a 10 minute trajectory from the open-source data set of Sargolini et al. (2006) ready to import. In the following figure blue shows (low resolution) trajectory data imported into an `Agent()` and purple shows the smoothly upsampled trajectory taken by the `Agent()` during exploration. 
```python
Agent.import_trajectory(dataset='sargolini')
#or 
Agent.import_trajectory(times=array_of_times,
                        positions=array_of_positions)

```
<img src="./readme_figs/imported_trajectory.png" height="200">

#### Policy control 
By default the movement policy is an random and uncontrolled (e.g. displayed above). It is possible, however, to manually pass a "drift_velocity" to the Agent on each `update()` step. This 'closes the loop' allowing, for example, Actor-Critic systems to control the Agent policy. As a demonstration that this method can be used to control the agent's movement we set a radial drift velocity to encourage circular motion. We also use RatInABox to perform a simple model-free RL task and find a reward hidden behind a wall (the full script is given as an example script [here](./example_scripts/example_script_reinforcement_lerning.ipynb))
```python
Agent.update(drift_velocity=drift_velocity)
```
<img src="./readme_figs/motion.gif" height="200">

#### Wall repelling 
Under the random motion policy, walls in the environment mildly "repel" the agent. Coupled with the finite turning speed this replicates an effect (known as thigmotaxis and linked to anxiety) where the agent is biased to over-explore near walls and corners (as shown in these heatmaps) matching real rodent behaviour. It can be turned up or down with the `anxiety` parameter.
```python 
Αgent.thigmotaxis = 0.8 #1 = high thigmotaxis (left plot), 0 = low (right)
```
<img src="./readme_figs/wall_repel.png" height="220">


### (iii) `Neurons()` features 

#### Multiple cell types: 
We provide a list of premade `Neurons()` subclasses. These include: 

* `PlaceCells(Neurons)` 
* `GridCells(Neurons)`
* `BoundaryVectorCells(Neurons)` (can be egocentric or allocentric)
* `VelocityCells(Neurons)`
* `SpeedCells(Neurons)`
* `HeadDirectionCells(Neurons)`
* `FeedForwardLayer(Neurons)` - calculates activated weighted sum of inputs from a provide list of input `Neurons()` layers.

This last class, `FeedForwardLayer` deserves special mention. Instead of its firing rate being determined explicitly by the state of the `Agen` it summates synaptic inputs from a provided list of input layers (which can be any `Neurons` subclass). This layer is the building block for how more complex networks can be studied using `RatInABox`. 

Place cells come in multiple types (give by `params['description']`):
* `"gaussian"`: normal gaussian place cell 
* `"gaussian_threshold"`: gaussian thresholded at 1 sigma
* `"diff_of_gaussian"`: gaussian(sigma) - gaussian(1.5 sigma)
* `"top_hat"`: circular receptive field, max firing rate within, min firing rate otherwise
* `"one_hot"`: the closest palce cell to any given location is established. This and only this cell fires. 

This last place cell type, `"one_hot"` is particularly useful as it essentially rediscretises space and tabularises the state space (gridworld again). This can be used to  contrast and compare learning algorithms acting over continuous vs discrete state spaces. 

#### `PlaceCell()` geometry
Choose how you want `PlaceCells` to interact with walls in the `Environment`. We provide three types of geometries.  
<img src="./readme_figs/wall_geometry.png" height="220">

#### Spiking 
All neurons are rate based. However, at each update spikes are sampled as though neurons were Poisson neurons. These are stored in `Neurons.history['spikes']`. The max and min firing rates can be set with `Neurons.max_fr` and  `Neurons.min_fr`.
```
Neurons.plot_ratemap(spikes=True)
```
<img src="./readme_figs/spikes.png" height="180">


#### Rate maps 
`PlaceCells()`, `GridCells()` and allocentric `BoundaryVectorCells()` (among others) have firing rates which depend exclusively on the position of the agent. These rate maps can be displayed by querying their firing rate at an array of positions spanning the environment, then plotting. This process is done for you using the function `Neurons.plot_rate_map()`. 

More generally, however, cells firing is not only determined by position but potentially other factors (e.g. velocity or historical effects if the layer is part of a recurrent network). In these cases the above method of plotting rate maps will fail. A more robust way to display the receptive field is to plot a heatmap of the positions of the Agent has visited where each positions contribution to a bin is weighted by the firing rate observed at that position. Over time, as coverage become complete, the firing fields become visible.
```
Neurons.plot_rate_map() #attempted to plot analytic rate map 
Neurons.plot_rate_map(by_history=True) #plots rate map by firing-rate-weighted position heatmap
``` 
<img src="./readme_figs/rate_map.png" height="400">

#### More complex Neuron types
We encourage users to create their own subclasses of `Neurons()`. This is easy to do, see comments in the `Neurons()` class within the [code](./ratinabox.py) for explanation. For example in the case study scripts we create bespoke `ValueNeuron(Neurons)` and `PyramidalNeurons(Neurons)` classes. By forming these classes from the parent `Neurons()` class, the plotting and analysis features described above remain available to these bespoke Neuron types. 


## Example Scripts

### Example 1: Simple
Full script [here (./example_scripts/simple_example.ipynb)](./example_scripts/simple_example.ipynb). Initialise a 2D environment. Initialise an agent in the environment. Initialise some place cells. Simulate for 20 seconds. Print table of times, position and firing rates and plot trajectory and rate timeseries'. 

```python
import ratinabox
from ratinabox import *

#Initialise an Environment, an Agent and some place cells 
Env = Environment()
Ag = Agent(Env)
PCs = PlaceCells(Ag)

#Explore for 20 seconds
for i in range(int(20/Ag.dt)):
    Ag.update()
    PCs.update()

#Print outouts 
print(Ag.history['t'][:10])
print(Ag.history['pos'][:10])
print(PCs.history['firingrate'][:10])

#Plot outputs
Ag.plot_trajectory()
PCs.plot_rate_timeseries()
```

### Example 2: Extensive
In this example we go a bit further. it can be found [here (./example_scripts/extensive_example.ipynb)](./example_scripts/extensive_example.ipynb).
1. Initialise environment. A rectangular environment of size 2 x 1 meters. 
2. Add walls. Dividing the environment into two equal rooms. 
3. Add Agent. Place the Agent at coordinates (0.5,0.5). Set the speed scale of the agent to be 20 cm/s.
4. Add place cells. 100 Gaussian threshold place cells. Set the radius to 40 cm. Set their wall geometry to "line_of_sight". Set the location of the 100th place cells to be near the middle of the doorway at coordinates(1.1,0.5). Set the max firing rate of these place cells to 3 Hz and the min firing rate (e.g. baseline) of 0.1 Hz. 
5. Add boundary vector cells. 30 of them. 
6. Simulate. For 10 minutes of random motio with a timestep of dt=10 ms. 
7. Plot trajectory. Plot final 30 seconds from t=4min30 to t=5mins seconds overlayed onto a heatmap of the trajectory over the full period. 
8. Plot timeseries. For 12 randomly chosen boundary vector cells. From t_start = 0 s to t_end = 60 s. Include spikes. 
9. Plot place cells. Show a scatter plot of the centres of the place cells. 
10. Plot rate maps. For 3 randomly chosen place cells. Then, below this, plot a rate map of the same 5 place cells but as calculated using the firing-rate-weighted position historgram. Include spikes on the latter rate maps. 

Despite the complexity of the above simulation it requires only ~40 lines of code and takes ~1.5 minutes to run on a laptop (or just 5 seconds whith dt=200 ms, which is still stable).

``` python 
import ratinabox
from ratinabox import *

# 1 Initialise environment.
Env = Environment(
    params = {'aspect':2,
               'scale':1})

# 2 Add walls. 
Env.add_wall([[1,0],[1,0.35]])
Env.add_wall([[1,0.65],[1,1]])

# 3 Add Agent.
Ag = Agent(Env,
           params={'speed_mean':0.2})
Ag.pos = np.array([0.5,0.5])

# 4 Add place cells. 
PCs = PlaceCells(Ag,
                 params={'n':100,
                         'description':'gaussian_threshold',
                         'widths':0.40,
                         'wall_geometry':'line_of_sight',
                         'max_fr':5,
                         'min_fr':0.1})
PCs.place_cell_centres[99] = np.array([1.1,0.5])

# 5 Add boundary vector cells.
BVCs = BoundaryVectorCells(Ag,
                params = {'n':30,})

# 6 Simulate. 
dt = 10e-3 
T = 10*60
from tqdm import tqdm #gives time bar
for i in tqdm(range(int(T/dt))):
    Ag.update(dt=dt)
    PCs.update()
    BVCs.update()

# 7 Plot trajectory. 
fig, ax = Ag.plot_position_heatmap()
fig, ax = Ag.plot_trajectory(t_start=50,t_end=60,fig=fig,ax=ax)

# 8 Plot timeseries. 
fig, ax = BVCs.plot_rate_timeseries(t_start=0,t_end=60,chosen_neurons='12',spikes=True)

# 9 Plot place cells. 
fig, ax = PCs.plot_place_cell_locations()
# 10 Plot rate maps. 
fig, ax = PCs.plot_rate_map(chosen_neurons='3',method='analytic')
fig, ax = PCs.plot_rate_map(chosen_neurons='3',method='history',spikes=True)

# 11 Display BVC rate maps and polar receptive fields
fig, ax = BVCs.plot_rate_map(chosen_neurons='2')
fig, ax = BVCs.plot_BVC_receptive_field(chosen_neurons='2')
```

### Example 3: `RatInABox` for reinforcement learning. 
`RatInABox` is used in a reinforcement learning project. An `Agent` exploring in 2D must find a reward hidden behind a wall. In this script we explain how to define a bespoke `Neurons()` subclass (the `ValueNeuron`) which can be trained, using continuous TD learning, to learn the value function for a given policy. The policy in turn is then biased such that the `Agent` follows the gradient of the value function and "ascends" the value map, approaching reward. 

Script (and more detailed description) can be found here [./example_scripts/reinforcement_learning_example.ipynb](./example_scripts/reinforcement_learning_example.ipynb)

### Example 4: `RatInABox` for path integration
`RatInABox` is used to build a netwokr capable of learning to "path integrate" it's position in the absence of sensory input. A set of bespoke `PyramidalNeurons` encoding the `Agent`s position in a 1D `Environment` learn, using a local Hebbian learning rule, to become a ring-attractor network which can maintain position estimate in the absence of sensory drive. Inputs from velocity neurons (also learned) "push" the bump of activity around the ring attractor in correspondence with the position. 

Script (and more detailed description) can be found here [./example_scripts/path_interation_example.ipynb](./example_scripts/path_integration_example.ipynb)


## Contribute 
`RatInABox` is an open source project, and we actively encourage community contributions. These can take various forms, such as new movement policies, new cells types, new plotting functions, new geometries, bug fixes, documentation, citations of relevant work, or additional experiment notebooks. If there is a small contribution you would like to make, please feel free to open a pull request, and we can review it. If you would like to add a new `Neurons` class please pull request it into the [`contribs`](./contribs) directory. If there is a larger contribution you are considering please contact the correponding author at `tomgeorge1@btinternet.com`. 

## Cite
If you use `RatInABox` in your research or educational material, please cite the work as follows: `my wicked bibtex citation`
The research paper corresponding to the above citation can be found [here](link/to/my/paper).

## 
<img src="./readme_figs/riab.png" height="200">
