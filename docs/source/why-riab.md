# Why RatInABox? 
## Our beliefs and mission statement

The document outlines a set of beliefs which motivated RatInABox and a set of principles on which is was, and continues to be, built. It should be used to guide decisions about whether to accept new functionality, make changes to the styling of code or whether to take on new dependencies etc.

![why_riab](https://github.com/RatInABox-Lab/RatInABox/assets/41446693/02b28673-84d4-47e5-9bdf-b52d2384cfea)



## Our beliefs 

* **Continuous motion modelling is less common than discrete motion modelling (but not for the right reasons).** Discrete models (e.g. grid world, markov chains, random walks, tabular RL) have historically been favoured over continuous models (defined below) for modelling animal behaviour and neural representations. We believe this is primarily because continuous models are harder to code and without a large community not such a large community supporting it. We believe it is _not_ because discreteness is in any way fundamental to the animal behaviour or neural processes being modelled.

This means that... 

* **Many researchers _would_ use continuous models if the entry barrier could be lowered.** The "spin up" for generating the continuous simulation on the left was (pre-RatInABox) higher than the grid world simulation on the right. If we could save people the effort of writing the boilerplate code many would switch.

This is because...

* **The continuous approach is more accurate, can make models more insightful and can scale better.** You get what you pay for*. When modelling continuous systems its better to use continuous models. As the animation shows continuous models can more accurately replicate animal behaviour and this is important if we wish to understand the mathematical or neural processes underpinning it.

We believe that 

* **This is eminently viable.** A minimal set of priciples and well made supporting software toolkit amounts to a framework which can significantly speed up motion modelling and improve the science which relies on it. This is one of the purposes of RatInABox.

Lastly we hold the belief that: 

* **Visual clarity is important for science and should be built into scientific software.** There is no point doing scientific research if you don't then communicate it effectively. Making minimal, clear and informative figures is an effective way to do this and therefore scientific software should support this by providing extensible plotting functions to visualise the data they handle/produce. 


### What does it mean to be continuous?
Discrete and continouous (time, space, actions etc.) are alternative frameworks within which variables can exists and evolve. Discrete frameworks view time as existing only in well separated (e.g. by `dt`) moments  or space as existing in well separated (e.g. by `dx`) bins.  In contrast, time and space in continuous models can take any values. Some concrete examples: 
* Where discrete models care about updates $x_{t+1} = x_{t}+...$ continuous models care about dynamics $\frac{dx(t)}{dt}=\cdot \cdot \cdot$.
* Where discrete models care about summations $V_{t}^{\pi}=\sum_{t}\gamma^{t}R_{t}$ continuous models care about integrals $V^{\pi}(t) = \int_t^{\infty}e^{-\frac{t^{\prime-t}}{\tau}}R(t) dt^{\prime}$

For a model to be continuous discretisation size variables `dx` and `dt` either _shouldn't exist_, or, if they do, their values _shouldn't particularly matter_ i.e. they might affect compute time or the resolution of visual renderings but not the statistics of the process being simulated. In the same way that animals have no notion of `dx` or `dt` when locomoting, models shouldn't either. 

This is simple in theory but non-trivial in practice. For example, if I changed either `dx` or `dt` in the grid world animation you would immediate know because it would affect the statistics of the motion the simulation produces.

In RatInABox the agents can in principle be at _any_ location in environment (down to float precision, there is no `dx`) and although `dt` is defined it is only an anciliary variable necessary because CPUs compute discretely - fundamentally the equations of motions are all temporally homogeneous (the statistics are independent of `dt`). Details of how this work are laid out in the paper. 

## Our mission statement
Based on these beliefs , the mission of RatInABox is **to provide a simple and extensible community-driven framework for modelling motion and neural representations in continuous environemnts with Python**. 

Our goal is to **significantly lower the entry barrier such that making and visualising highly accurate simulations in continuous time and space is as easy as -- and a viable alternative to -- grid world**. 

To achieve these we will adhere to a set of core principles: 

## Our core principles
* **Continuity**: At the heart of RatInABox sits the principle of continuity. Environments are extended regions of space; a zero-dimensional Agent can moves smoothly around all regions of the space and interact with 1-dimensional walls and 0-dimensioal objects. 
* **Simplicity**: Code should be easy to use (a simple, sensible API) and easy to learn (well-written, diverse and relevant demos and tutorials). RatInABox is primarily a toolkit not a model bank.  This may warrant rejecting -- or perhaps keeping as a `contribs` or `demo` -- new albeit exciting features if they would bloat the core utility and make it harder to learn. 
* **Visualisations**: RatInABox will always maintain powerful visualisation functions which are modular, combinative and user-extensible. Since this software models motion dynamics, wherever possible animations will be supported too. Clarity comes before aesthetics but these usually correlate.


### Auxiliarly principles 
* **1D and 2D**: Wherever possible RatInABox support 1D and 2D environments. 1D environments are not just thin 2D environments, they are fundamentally 1D. 
* **Speed**: Wherever possible calculations should be vectorised, algorithms optimised and default parameters set to allow RatInABox to run fast out-of-the-box.
* **Accuracy**: Where implemented motion or cell models should be realistic, and fitted to real data.



*Of course heavily reduced models can still teach us a lot but it's a question of at what level you hope to understand a system. If you want to understand the neural underpinnings of complex behavioural trajectories then grid world is probably not the best place to start.