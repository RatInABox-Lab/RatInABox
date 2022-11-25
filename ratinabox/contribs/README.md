# Contributions 

Contributions to `RatInABox` are welcome. Pull request them into the here.

If you wish to add a new `Neurons` subclass you should follow the instructions found within the header comments of the `Neurons` class (code [here](../../ratinabox/Neurons.py)). N.b. your new `Neurons` subclass can be a direct subclass of `Neurons` itself (e.g. `MyNeuron(Neuron)`) or it can be a subclass of one of the other `Neurons` subclasses (e.g. `MyFancyPlaceCells(PlaceCells)` or ` MyFancyFeedForwardLayerWithItsOwnFancyLearningRule(FeedForwardLayer)`). 

We provide two example contrib files for you to refer to. These files can also be run standalone to show mini-demos.: 
* [PhasePrecessingPlaceCells.py](./PhasePrecessingPlaceCell.py): Place cells (1 or 2D) which also phase precess.
* [ValueNeuron.py](./ValueNeuron.py): A neuron which uses TD learning to approximate a value function as a sum of inpouts from a downstream layer. Is a subclass of `FeedForwardLayer`
* [PlaneWaveNeurons.py](./PlaneWaveNeurons.py): A layer of neurons. Each neurons rate map is a plane wave with a random orientation and phase offset.

Once you have made a contribution (`myContribution.py`), reinstall RatInABox and import using 
```python 
import ratinabox
from ratinabox.contribs.myContribution import *
```