# RatInABox 🐀📦

RatInABox is a toolkit for simulating pseudo-realistic motion in continuous 1 and 2-dimensional environment. Essentially it simulates a rat 🐀 in a box 📦. With it you can

* Generate pseudo-realistic trajectories for rats foraging/exploring 1 and 2D environments
* Simulate spatially selective cells typically found in the Hippocampal-Entorhinal system (place cells, grid cells, boundary vector cells, and velocity cells). 

~~Gridworld~~ RatInABox represents a clean departure from pre-discretised "gridworld" models which have so far been sued for simulating hippocampal navigation. Position and neuronal firing rates are calculated online with float precision. 

RatInABox contains just three classes: 

1. `Environment()`: The environment/maze the agent lives in. 1- or 2-dimensional. Handles geometry, walls, boundary conditions etc.
2. `Agent()`: The agent (or "rat") moving around the environment. 
3. `Neurons()`: A population of neurons. Neurons have firing rates determined by the state of the agent (e.g. it's position). 

Here's an animation of the kind of simulation you can run. It shows an agent randomly exploring a 2D environment with a wall. Four populations of cells  (place cells, grid cells, boundary vector cells and velcity cells) "fire" as the agent explore. Below shows the code needed to replicate this exact simulation (13 lines!).

![Example environments](./figures/readme/animation.gif)


## Requirements
* Python 3.7+
* NumPy
* Scipy
* Matplotlib
* Jupyter (optional)

## Installation 

## Features 
Here is a list of features loosely organised into three categories. Those pertaining to (i) Environment, (ii) Agent and (iii) the Neurons. 

### (i) `Environment()` features
#### Walls 
Arbitrarily add walls to the environment to replicate any desired maze structure
```
Env.add_wall([])
```

#### Boundary conditions 
#### 1- or 2-dimensions 


### (ii) `Agent()` features
#### 

### (iii) `Neuron()` features 

