# RatInABox 🐀📦

RatInABox is a toolkit for simulating pseudo-realistic motion in  and 2-dimensional environment. It simulates a rat 🐀 in a box 📦. With it you can

* Generate pseudo-realistic trajectories for rats foraging/exploring 1 and 2D environments
* Simulate spatially selective cells typically found in the Hippocampal-Entorhinal system (place cells, grid cells, boundary vector cells, and velocity cells).

This is not gridworld. This simulator is fundamentally continuous in both time and space. 
There are three main classes: 

1. `Environment()`: The environment/maze the agent lives in. 1 or 2D. Handles geometry, collision with walls, boundary conditions.
2. `Agent()`: The agent or rat doing the exploring. Agents exist within an Environment. Agents move around the environment. 
3. `Neurons()`: Neurons have firing rates determined by the agent (e.g. it's position i). Neurons exist with an Agent. 

Here's a pretty animation of the kind of simulation you can run. It shows an agent randomly exploring a 2D environment with a wall. Four place cells, A single gaussian place cell is located in the middle of hte environmetn and fires when the agent enters its receptive field. 

![Example environments](./figures/readme/animation.gif)


## Requirements
* Python 3.7+
* NumPy
* Scipy
* Matplotlib
* Jupyter (optional)

## Installation 

## Features 

### 