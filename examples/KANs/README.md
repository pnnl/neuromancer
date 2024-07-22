# Kolmogorov-Arnold Networks in Neuromancer

This directory contains interactive examples that can serve as a step-by-step tutorial 
showcasing the capabilities of Kolmogorov-Arnold Networks (KANs) and finite basis KANs (FBKANs) in Neuromancer.

Examples of learning from multiscale, noisy data with KANs and FBKANs:
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/feature/fbkans/examples/KANs/p1_fbkan_vs_kan_noise_data_1d.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: A comparison of KANs and FBKANs in learning a 1D multiscale function with noise
+ <a target="_blank" href="https://colab.research.google.com/github/pnnl/neuromancer/blob/feature/fbkans/examples/KANs/p2_fbkan_vs_kan_noise_data_2d.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: A comparison of KANs and FBKANs in learning a 2D multiscale function with noise

## Kolmogorov-Arnold Networks (KANs)
Based on the Kolmogorov-Arnold representation theorem, KANs offer an alternative architecture: where traditional neural networks utilize fixed activation functions, KANs employ learnable activation functions on the edges of the network, replacing linear weight parameters with parametrized spline functions. This fundamental shift sometimes enhances model interpretability and improves computational efficiency and accuracy [1]. KANs are available on Neuromancer via `blocks.KANBlock`, which leverages the efficient-kan implementation of [2]. Moreover, users can leverage the finite basis KANs (FBKANs), a domain decomposition method for KANs proposed by Howard et al. (2024)[3] by simply setting the `num_domains` argument in `blocks.KANBlock`.

### References

[1] [Liu, Ziming, et al. (2024). KAN: Kolmogorov-Arnold Networks.](https://arxiv.org/abs/2404.19756)

[2] https://github.com/Blealtan/efficient-kan

[3] Howard, Amanda A., et al. (2024) Finite basis Kolmogorov-Arnold networks: domain decomposition for data-diven and physics-informed problems.
