
## Community Development Guidelines

We welcome contributions and feedback from the open-source community!

### Discussions

[Discussions](https://github.com/pnnl/neuromancer/discussions) should be the first line of contact for new users to provide direct feedback on the library.
Post your [Ideas](https://github.com/pnnl/neuromancer/discussions/categories/ideas) for new features or examples, 
showcase your work using neuromancer in [Show and tell](https://github.com/pnnl/neuromancer/discussions/categories/show-and-tell),
or get support for usage in [Q&As](https://github.com/pnnl/neuromancer/discussions/categories/q-a),
please post them in one of our  categories.


### Contributing examples
If you have an example of using NeuroMANCER to solve an interesting problem, or of using 
NeuroMANCER in a unique way please share them in [Show and tell](https://github.com/pnnl/neuromancer/discussions/categories/show-and-tell)
discussions.
The best examples might be incorporated into our current library of examples. 
To submit an example, create a folder for your example/s in the example folder if there isn't 
currently an applicable folder and place either your executable python file or notebook file there. 
Push your code back to Github and then submit a pull request. Please make sure to note in a comment at 
the top of your code if there are additional dependencies to run your example and how to install 
those dependencies. 

### Contributing code

We welcome contributions to NeuroMANCER. Please accompany contributions with some lightweight unit tests
via pytest (see test/ folder for some examples of easy to compose unit tests using pytest). 
In addition to unit tests
a script utilizing introduced new classes or modules should be placed in the examples folder. 
To contribute a new well-developed feature please submit a pull request (PR). 
Before creating a PR, we encourage developers to discuss and document the intended feature
in [Ideas](https://github.com/pnnl/neuromancer/discussions/categories/ideas) discussion category.

### Reporting issues or bugs
If you find a bug in the code or want to request a new well-developed feature, please open an [issue](https://github.com/pnnl/neuromancer/issues).


## NeuroMANCER development plan
Here are some upcoming features we plan to develop. Please let us know if you would like to get involved and 
contribute so we may be able to coordinate on development. If there is a feature that you think would
be highly valuable but not included below, please open an issue and let us know your thoughts. 

+ Control and modelling for networked systems
+ Support for stochastic differential equations (SDEs)
+ Easy to implement modeling and control with uncertainty quantification
+ Proximal operators for dealing with equality and inequality constraints
+ Interface with CVXPYlayers
+ Online learning examples
+ Benchmark examples of DPC compared to deep RL
+ Conda and pip package distribution
+ More versatile and simplified time series dataloading
+ Discovery of governing equations from learned RHS via NODEs and SINDy
+ More domain science examples
