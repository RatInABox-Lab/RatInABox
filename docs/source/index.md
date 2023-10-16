# RatInABox

`RatInABox` (see [paper](https://www.biorxiv.org/content/10.1101/2022.08.10.503541v3)) is a toolkit for generating synthetic behaviour and neural data for spatially and/or velocity selective cell types in complex continuous environments.

With `RatInABox` you can: 

* **Generate realistic trajectories** for rats exploring complex 1 and 2D environments under a smooth random policy, an external control signal, or your own trajectory data.
* **Generate artificial neuronal data** for various location- or velocity-selective cells found in the brain (e.g., but not limited to, Hippocampal cell types), or build your own more complex cell types. 
* **Build and train complex multi-layer networks** of cells, powered by data generated with `RatInABox`. 

<img src="https://raw.githubusercontent.com/RatInABox-Lab/RatInABox/dev/.images/readme/ratinabox.gif" width=850>

`RatInABox` is an open source project welcoming [contributions](#contribute). If you use `RatInABox` please [cite](#cite) the paper and consider giving this repository a star ☆. It contains three classes: 

1. `Environment()`📦: The environment/maze (or "box") that the agent lives in. 1- or 2-dimensional.
2. `Agent()`      🐀: The agent (or "rat") moving around the `Environment`. 
3. `Neurons()`    🧠: A population of neurons with firing rates determined by the state (position and velocity) of the `Agent`. Make your own or use one of our premade cell types including: 
    * `PlaceCells()`
    * `GridCells()`
    * `BoundaryVectorCells()` (egocentric or allocentric)
    * `ObjectVectorCells()`
    * `VelocityCells()`
    * `SpeedCells()`
    * `FieldOfViewNeurons()` (egocentric encoding of what the `Agent` can see)
    * `RandomSpatialNeurons()`
    * `HeadDirectionCells()`
    * `FeedForwardLayer()` (a generic class analagous to a feedforward layer in a deep neural network)
    * `NeuralNetworkNeurons()` (a generic class analagous to a deep neural network)
    * `SuccessorFeatures()` 
    * ...

The top animation shows an example use case: an `Agent` randomly explores a 2D `Environment` with a wall. Three populations of `Neurons` (`PlaceCells`, `GridCells`, `BoundaryVectorCells`) fire according to the receptive fields shown. All data is saved into the history for downstream use. `RatInABox` is fully continuous in space; this means that position and neural firing rates are calculated rapidly online with float precision rather than pre-calculated over a discretised mesh. `RatInABox` is flexibly discretised in time; `dt` can be set by the user (defaulting to 10 ms) depending on requirements.


## Key features
* **Non-specific**: Trajectories can be randomly generated, imported, or adaptively controlled making `RatInABox` a powerful engine for many tasks involving continuous motion (e.g. control theory or [reinforcement learning](#policy-control)). 
* **Biological**:   Simulate large populations of spatially and/or velocity modulated cell types. Neurons can be rate based or spiking. The random motion model is fitted to match real rodent motion. 
* **Flexible**:     Simulate environments in 1D or 2D with arbitrarily wall, boundary and hole arrangements.  Combine premade or bespoke `Neurons` classes into arbitrary deep networks (examples given).
* **Fast**:         Simulating 1 minute of exploration in a 2D environment with 100 place cells (dt=10 ms) take just 2 seconds on a laptop (no GPU needed).
* **Precise**:      No more prediscretised positions, tabular state spaces, or jerky movement policies. It's all continuous. 
* **Easy**:         Sensible default parameters mean you can have realisitic simulation data to work with in <10 lines of code.
* **Visual**        Plot or animate trajectories, firing rate timeseries', spike rasters, receptive fields, heat maps, velocity histograms...using the plotting functions ([summarised here](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/list_of_plotting_fuctions.md)). 

<!-- 
## Announcement about support for OpenAI's `gymnasium` <img src="https://raw.githubusercontent.com/RatInABox-Lab/RatInABox/dev/.images/readme/gymnasium_logo.svg" width=25> API
A new wrapper contributed by [@SynapticSage](https://github.com/SynapticSage) allows `RatInABox` to natively support OpenAI's [`gymnasium`](https://gymnasium.farama.org) API for standardised and multiagent reinforment learning. This can be used to flexibly integrate `RatInABox` with other RL libraries such as Stable-Baselines3 etc. and to build non-trivial tasks with objectives and time dependent rewards. Check it out [here](https://github.com/TomGeorge1234/RatInABox/blob/dev/ratinabox/contribs/TaskEnv_example_files/TaskEnvironment_basics.md). -->


## Contribute
`RatInABox` is open source project and we actively encourage  all contributions from example bug fixes to documentation or new cell types. Feel free to make a pull request (you will need to fork the repository first) or raise and issue. 

We have a dedicated [contribs](https://github.com/RatInABox-Lab/RatInABox/tree/dev/ratinabox/contribs) directory where you can safely add awesome scripts and new `Neurons` classes etc.

Questions? Just ask! Ideally via opening an issue so others can see the answer too. 

Thanks to all contributors so far:
![GitHub Contributors Image](https://contrib.rocks/image?repo=TomGeorge1234/RatInABox)

## Cite [![](http://img.shields.io/badge/bioRxiv-10.1101/2022.08.10.503541-B31B1B.svg)](https://doi.org/10.1101/2022.08.10.503541) 

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

```{toctree}
:maxdepth: 2
:hidden:

why-riab
get-started/index
documentation
```