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
Arbitrarily add walls to the environment to replicate any desired maze structure. The following code shows how to add walls to make a multicompartment environment like used in Carpenter et al. (2015). 
```
Env.add_wall([[0.3,0.0],[0.3,0.5]])
```
![Walls](./figures/readme/walls.pdf)

#### Boundary conditions 
Boundary conditions can be "periodic" or "solid". Place cells and the Agent will respect boundaries accordingly.
![Boundaries](./figures/readme/boundary_conditions.pdf)

#### 1- or 2-dimensions 
Most features work in both 1 and 2 dimensions. Some don't (e.g. walls, boundary_vector_cells aren't defined in 1D)
![1D](./figures/readme/one_dimension.pdf)


### (ii) `Agent()` features
#### Wall repelling 
Walls mildly repel the agents. Coupled with the finite turning speed this creates a combined effect that the agent is encourged over explore walls and corners (as shown in ...et al.). This can of course be turned off.
```
Αg.walls_repel = True #False
```
![Walls](./figures/readme/walls_repel.pdf)

#### Policy control 
By default the movement policy is an uncontrolled "smooth" random walk where the velocity is governed by an Ornstein-Uhlenbeck process. It is possible, however, to pass a "drift velocity" to the Agent, towards which it's velocity will drift. We envisage this being use, for example, by an Actor-Critic system to control the Agent. For demonstration here we set a radial drift velocity to encourage circular motion.   

### (iii) `Neuron()` features 

