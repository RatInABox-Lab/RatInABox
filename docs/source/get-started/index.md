# Get started 
Many [demos](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/) are provided. Reading through the [example scripts](./examples) (one simple and one extensive, duplicated at the bottom of the readme) these should be enough to get started. We also provide numerous interactive jupyter scripts as more in-depth case studies; for example one where `RatInABox` is used for [reinforcement learning](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/reinforcement_learning_example.ipynb), another for [neural decoding](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/decoding_position_example.ipynb) of position from firing rate. Jupyter scripts reproducing all figures in the [paper](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/paper_figures.ipynb) and [readme](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/readme_figures.ipynb) are also provided. All [demos](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/) can be run on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/RatInABox-Lab/RatInABox/blob/dev/demos/)

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

```{toctree}
:maxdepth: 2
:hidden:

examples
```