# Awesome Normalizing Flows &thinsp; [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

<img src="normalizing-flow.svg" alt="Normalizing Flow" width="1000">

A list of awesome resources for understanding and applying normalizing flows (NF). It's a relatively simple yet powerful new tool in statistics for constructing expressive probability distributions from simple base distribution using a chain (flow) of trainable smooth bijective transformations (diffeomorphisms).

<small>Figure inspired by [Lilian Weng](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models). Created in TikZ. Source code found [here](https://github.com/janosh/tikz/tree/master/assets/normalizing-flow).</small>

## 📝 Publications

- 1 April, 2011 - [Iterative Gaussianization: from ICA to Random Rotations](https://arxiv.org/abs/1602.00229) by Laparra et. al.
  
  Normalizing flows in the form of Gaussianization in an iterative format. Also shows connections to information theory.

- 30 Oct, 2014 - [Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) by Laurent Dinh, David Krueger, Yoshua Bengio.

- 12 Feb, 2015 - [Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle.

- 21 May, 2015 - [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) by Danilo Rezende, Shakir Mohamed.

  They show how to go beyond mean-field variational inference by using flows to increase the flexibility of the variational family.

- 27 May, 2016 - [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) by Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio.

  They introduce the affine coupling layer, a major improvement in terms of flexibility over the additive coupling layer with unit Jacobian while keeping a single-pass forward and inverse transformation for fast sampling and density estimation, respectively.

- 15 Jun, 2016 - [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934) by Diederik Kingma et al.

- 6 Mar, 2017 - [Multiplicative Normalizing Flows for Variational Bayesian Neural Networks](https://arxiv.org/abs/1703.01961) by Christos Louizos, Max Welling.

  They introduce a new type of variational Bayesian neural network that uses flows to generate auxiliary random variables which boost the flexibility of the variational family by multiplying the means of a fully-factorized Gaussian posterior over network parameters. This turns the usual diagonal covariance Gaussian into something that allows for multimodality and non-linear dependencies between network parameters.

- 19 May, 2017 - [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057) by George Papamakarios, Theo Pavlakou, Iain Murray.

  To improve predictive accuracy and uncertainty in Bayesian neural networks, they interpret multiplicative noise in neural network parameters as auxiliary random variables and show how to model these using flows. As in variational inference, the idea is again to use the power of flows to augment the approximate posterior while maintaining tractability.

- 3 July, 2018 - [Deep Density Destructors](http://proceedings.mlr.press/v80/inouye18a.html) by Inouye & Ravikumar

  Normalizing flows but from an iterative perspective. Features a Tree-based density estimator.

- 9 Jul, 2018 - [Glow: Generative Flow with Invertible 1x1 Convolutions](http://arxiv.org/abs/1807.03039) by Kingma, Dhariwal.

  They show that flows using invertible 1x1 convolution achieve high likelihood on standard generative benchmarks and can efficiently synthesize realistic-looking, large images.

- 2 Oct, 2018 - [FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367) by Grathwohl & Chen et. al.

- 9 Apr, 2019 - [Block Neural Autoregressive Flow](https://arxiv.org/abs/1904.04676) - De Cao et. al.
  
- 30 May, 2019 - [Graph Normalizing Flows](https://arxiv.org/abs/1905.13177) by Jenny Liu et al. A new, reversible graph network for prediction and generation.

  They perform similarly to message passing neural networks on supervised tasks, but at significantly reduced memory use, allowing them to scale to larger graphs. Combined with a novel graph auto-encoder for unsupervised learning, graph normalizing flows are a generative model for graph structures.

- 10 Jun, 2019 - [Neural Spline Flows](https://arxiv.org/abs/1906.04032) by Conor Durkan et. al.
  
  Uses monotonic ration splines as a coupling layer. This is currently one of the state of the art.

- 25 Aug, 2019 (_Revised Aug 6, 2020_) - [Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/abs/1908.09257) by Kobyzev et al.

  Another very thorough and very readable review article going through the basics of NFs as well as some of the state-of-the-art. Also highly recommended.

- 5 Dec, 2019 - [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) by Papamakarios et al.

  A thorough and very readable review article by some of the guys at DeepMind involved in the development of flows. Highly recommended.

- 15 Jan, 2020 (_Revised Apr 13, 2020_) - [Invertible Generative Modeling using Linear Rational Splines](https://arxiv.org/abs/2001.05168) by Dolatabadi et. al.

  A successor to the Neural spline flows which features an easy-to-compute inverse. 

- Mar 4, 2020 - [Gaussianization Flows](https://arxiv.org/abs/2003.01941) by Meng et. al.

  Uses a repeated composition of trainable kernel layers and orthogonal transformations. Very competitive versus some of the SOTA like Real-NVP, Glow and FFJORD.

## 📺 Videos

- Sep 24, 2019 - [Graph Normalizing Flows](https://youtu.be/frMPP30QQgY): Jenny Liu (University of Toronto, Vector Institute) introduces a new graph generating model for use e.g. in drug discovery, where training on molecules that are known to bind/dissolve/etc. may help to generate more similarly effective molecules.

- 9 Oct, 2019 - [A primer on normalizing flows](https://youtu.be/P4Ta-TZPVi0): [Laurent Dinh](https://laurent-dinh.github.io) (Google Brain), first author on both the NICE and RNVP papers and one of the first in this field, gives an introductory talk at "Machine Learning for Physics and the Physics of Learning 2019".

- 6 Dec, 2019 - [What are normalizing flows?](https://youtu.be/i7LjDvsLWCg): [Ari Seff](https://cs.princeton.edu/~aseff) (Princeton) created a super clear 3blue1brown-style video explaining the basics of normalizing flows.

## 🌐 Blog Posts

- Jan 17, 2018 - [Normalizing Flows Tutorial](https://blog.evjang.com/2018/01/nf1.html) by Eric Jang. Part 1: Distributions and Determinants. [Part 2: Modern Normalizing Flows](https://blog.evjang.com/2018/01/nf2.html). Features lots of great graphics.

- Apr 3, 2018 - [Normalizing Flows](https://akosiorek.github.io/ml/2018/04/03/norm_flows) by Adam Kosiorek.

- Oct 13, 2018 - [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models) by Lilian Weng.

## 🧑‍💻 Code

- Jul 11, 2017 - [`normalizing_flows_overview.ipynb`](https://docs.pymc.io/notebooks/normalizing_flows_overview.html) by PyMC3.

  A very helpful notebook showcasing how to work with flows in practice and comparing it to PyMC3's NUTS-based HMC kernel.

- 1 Sept, 2018 - [`pytorch-flows`](https://github.com/ikostrikov/pytorch-flows) by Ilya Kostrikov.

  PyTorch implementations of density estimation algorithms: MAF, RNVP, Glow.

- 30 Dec, 2018 - [`normalizing_flows`](https://github.com/kamenbliznashki/normalizing_flows) by Kamen Bliznashki.

  Pytorch implementations of density estimation algorithms: BNAF, Glow, MAF, RealNVP, planar flows.

- 9 Dec, 2019 - [`pytorch-normalizing-flows`](https://github.com/karpathy/pytorch-normalizing-flows) by Andrej Karpathy.

  A Jupyter notebook with PyTorch implementations of the most commonly used flows: NICE, RNVP, MAF, Glow, NSF.

- [`nflows`](https://github.com/bayesiains/nflows) by [bayesiains](https://homepages.inf.ed.ac.uk/imurray2/group/)
  A suite of most of the SOTA methods using PyTorch. From an ML group in Edinburgh. They created the current SOTA spline flows. Almost as complete as you'll find from a single repo.

- [`Normalizing Flows - Introduction (Part 1)`](https://pyro.ai/examples/normalizing_flows_i.html) by [pyro.ai](http://pyro.ai/) 
  
  A tutorial about how to use the `pyro-ppl` library (based on PyTorch) to use Normalizing flows. They have some of the SOTA methods including NSF and MAF.

- [`destructive-deep-learning`](https://github.com/davidinouye/destructive-deep-learning) by [David Inouye](https://www.davidinouye.com/)
  
  An entire suite of iterative methods to normalizing flows. Includes the Tree-based method as well as Gaussianization methods.

- [`Neural Transport`](https://pyro.ai/numpyro/examples/neutra.html) by [numpyro](http://num.pyro.ai/en/stable/)
  
  Features an example of how Normalizing flows can be used to get more robust posteriors from Monte Carlo methods. Uses the `numpyro` library which is a PPL with JAX as the backend. The NF implementations include the basic ones like IAF and BNAF.

- [`NuX`](https://github.com/Information-Fusion-Lab-Umass/NuX) by Information Fusion Labs (UMass)
  
  A library that has normalizing flows using JAX as the backend. Has some of the SOTA methods. They also feature a surjective flow via quantization.

- [`jax-flows`](https://github.com/ChrisWaites/jax-flows) by [Chris Waites](https://chriswaites.com/#/)
  
  Another library that has normalizing flows using JAX as the backend. Has some of the SOTA methods.

- [`Density Estimation with Neural ODEs`](https://torchdyn.readthedocs.io/en/latest/tutorials/07a_continuous_normalizing_flows.html) and [`Density Estimation with FFJORDs`](https://torchdyn.readthedocs.io/en/latest/tutorials/07b_ffjord.html) by [torchdyn](https://torchdyn.readthedocs.io/en/latest/index.html)
  
  An example for how to use FFJORD as a continuous normalizing flow (CNF). Based off of a PyTorch suite (`torchdyn`) which has continuous neural architectures.


## 🎉 Open to Suggestions!

See something that's missing from this list? [PRs welcome!](https://github.com/janosh/awesome-normalizing-flows/edit/master/readme.md)
