# Example scripts
In the folder called [demos](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/) we provide numerous script and demos which will help when learning `RatInABox`. In approximate order of complexity, these include:
* [simple_example.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/simple_example.ipynb): a very simple tutorial for importing RiaB, initialising an Environment, Agent and some PlaceCells, running a brief simulation and outputting some data. Code copied here for convenience.
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
* [extensive_example.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/extensive_example.ipynb): a more involved tutorial. More complex enivornment, more complex cell types and more complex plots are used. 
* [list_of_plotting_functions.md](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/list_of_plotting_fuctions.md): All the types of plots available for are listed and explained. 
* [readme_figures.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/readme_figures.ipynb): (Almost) all plots/animations shown in the root readme are produced from this script (plus some minor formatting done afterwards in powerpoint).
* [paper_figures.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/paper_figures.ipynb): (Almost) all plots/animations shown in the paper are produced from this script (plus some major formatting done afterwards in powerpoint).
* [decoding_position_example.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/decoding_position_example.ipynb): Postion is decoded from neural data generated with RatInABox. Place cells, grid cell and boundary vector cells are compared. 
* [splitter_cells_example.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/splitter_cells_example.ipynb): A simple simultaion demonstrating how `Splittre` cell data could be create in a figure-8 maze.
* [reinforcement_learning_example.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/reinforcement_learning_example.ipynb): RatInABox is use to construct, train and visualise a small two-layer network capable of model free reinforcement learning in order to find a reward hidden behind a wall. 
* [actor_critic_example.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/actor_critic_example.ipynb): RatInABox is use to implement the actor critic algorithm using deep neural networks.
* [successor_features_example.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/successor_features_example.ipynb): RatInABox is use to learn and visualise successor features under random and biased motion policies.
* [path_integration_example.ipynb](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/path_integration_example.ipynb): RatInABox is use to construct, train and visualise a large multi-layer network capable of learning a "ring attractor" capable of path integrating a position estimate using only velocity inputs.
