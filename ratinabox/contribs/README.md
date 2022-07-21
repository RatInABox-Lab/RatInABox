# Contributions 

Contributions to `RatInABox` are welcome. Pull request them here.

If you wish to add a new `Neurons` subclass you should follow the instructions found within the header comments of the `Neurons` class (code [here](../../ratinabox/Neurons.py)). N.b. your new `Neurons` subclass can be a direct subclass of `Neurons` itself (e.g. `MyNeuron(Neuron)`) or it can be a subclass of one of the other `Neurons` subclasses (e.g. `MyFancyPlaceCells(PlaceCells)` or ` MyFancyFeedForwardLayerWithItsOwnFancyLearningRule(FeedForwardLayer)`). 