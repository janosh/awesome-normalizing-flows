<img src="normalizing-flow.svg" alt="Normalizing Flow" align="right" height="120">

# Awesome Normalizing Flows &thinsp; [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A list of awesome resources for understanding and applying normalizing flows (NF). It's a relatively simple yet powerful new tool in statistics for constructing expressive probability distributions from simple base distribution using smooth bijective transformations (diffeomorphisms).

## Publications

- Dec 5, 2019 - [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) by Papamakarios et al. A thorough and very readable review article by some of the guys at DeepMind involved in the development of NF. Highly recommended.
- May 21, 2015 - [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) by Rezende et al. They show how to go beyond mean field in variational inference by using NF to increase the flexibility of the variational family and make much more complex approximate posteriors possible.
- Jun 12, 2017 - [Multiplicative Normalizing Flows for Variational Bayesian Neural Networks](https://arxiv.org/abs/1703.01961) by Louizos et al. With the goal of improving predictive accuracy and uncertainty in Bayesian neural networks, they interpret multiplicative noise in neural network parameters as auxiliary random variables and show how to model these using NF. As in variational inference, the idea is again use NF's power to augment the approximate posterior while maintaining tractability.

## Videos

- Dec 6, 2019 - Ari Seff created a [super helpful 3blue1brown-style video](https://youtube.com/watch?v=i7LjDvsLWCg) explaining the basics of normalizing flows.
- Oct 9, 2019 - ["A primer on normalizing flows" by Laurent Dinh](https://youtube.com/watch?v=P4Ta-TZPVi0), first author on both the NICE and RNVP papers and one of the first in this field.

## Blog Posts

- Oct 13, 2018 - [Flow-based Deep Generative Models by Lilian Weng](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models)
- Apr 3, 2018 - ["Normalizing Flows" by Adam Kosiorek](https://akosiorek.github.io/ml/2018/04/03/norm_flows.html)

## Code

- PyMC3 has a [very helpful notebook](https://docs.pymc.io/notebooks/normalizing_flows_overview.html) showcasing how to work with flows in practice and comparing it to their NUTS-based HMC implementation.
- Dec 9, 2019 - [A Jupyter notebook with PyTorch implementations of the most commonly used flows by Andrej Karpathy](https://github.com/karpathy/pytorch-normalizing-flows).

## Open to Suggestions!

Feel free to submit a PR to extend this list.
