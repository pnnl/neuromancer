.. SLiM documentation master file, created by
   sphinx-quickstart on Sat Nov  7 06:40:51 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. _Aaron Tuor: http://sw.cs.wwu.edu/~tuora/aarontuor/
.. _Orthogonal: https://arxiv.org/abs/1612.00188
.. _Butterfly: https://github.com/HazyResearch/learning-circuits
.. _L0: https://arxiv.org/pdf/1712.01312.pdf
.. _Spectral: https://arxiv.org/pdf/1803.09327.pdf
.. _`Schur Decomposition`: https://papers.nips.cc/paper/9513-non-normal-recurrent-neural-network-nnrnn-learning-long-time-dependencies-while-improving-expressivity-with-transient-dynamics.pdf
.. _Lasso: https://leon.bottou.org/publications/pdf/compstat-2010.pdf
.. _Symplectic: https://arxiv.org/abs/1705.03341
.. _AntiSymmetric: https://arxiv.org/abs/1705.03341

.. _`Concentric Ellipses`: https://arxiv.org/pdf/1705.03341.pdf
.. _`Swiss Roll`: https://arxiv.org/pdf/1705.03341.pdf
.. _Peaks: https://arxiv.org/pdf/1705.03341.pdf
.. _Add: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf
.. _Multiply: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf
.. _XOR: https://icml.cc/Conferences/2011/papers/532_icmlpaper.pdf
.. _Ackley: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.71.2526&rep=rep1&type=pdf
.. _MNIST: https://ieeexplore.ieee.org/abstract/document/6296535


SLiM: Structured Linear Maps
============================


This package provides a suite of structured linear maps which can be used as drop-in replacements for PyTorch's nn.Linear module. Recent work viewing neural networks from a dynamical systems perspective has introduced a host of parametrizations for the basic linear maps which are subcomponents of neural networks. Such parametrizations may enhance the stability of learning, and embed models with inductive priors that encode domain application knowledge. This package is an effort to collect these all in one place with a common API to facilitate rapid exploration.

.. note:: We encourage folks to contribute any new structured linear maps as feature requests.

+---------------------------------------+---------------------------------------+
| Linear Map                            | class                                 |
+=======================================+=======================================+
| Orthogonal_                           |:any:`OrthogonalLinear`                |
+---------------------------------------+---------------------------------------+
| L0_                                   |:any:`L0Linear`                        |
+---------------------------------------+---------------------------------------+
| Butterfly_                            |:any:`ButterflyLinear`                 |
+---------------------------------------+---------------------------------------+
| Spectral_                             |:any:`SpectralLinear`                  |
+---------------------------------------+---------------------------------------+
| `Schur Decomposition`_                |:any:`SchurDecompositionLinear`        |
+---------------------------------------+---------------------------------------+
| Lasso_                                |:any:`LassoLinear`                     |
+---------------------------------------+---------------------------------------+
| Symplectic_                           |:any:`SymplecticLinear`                |
+---------------------------------------+---------------------------------------+
| AntiSymmetric_                        |:any:`SkewSymmetricLinear`             |
+---------------------------------------+---------------------------------------+

Benchmarks
----------
Sequence to Classification:

- MNIST_
- XOR_

Point to Classification:

- `Concentric Ellipses`_
- `Swiss Roll`_
- Peaks_

Sequence to Regression:

- Add_
- Multiply_

Point to Regression:

- Ackley_

Dynamics:

- Building models


.. toctree::
   :maxdepth: 2
   :caption: Docs:

   linear.rst
   rnn.rst
   benchmarks.rst
