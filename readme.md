# Awesome Normalizing Flows

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) ![Pull Requests Welcome](https://img.shields.io/badge/Pull%20Requests-welcome-brightgreen.svg) [![License](https://img.shields.io/github/license/janosh/awesome-normalizing-flows?label=License)](license) ![GitHub last commit](https://img.shields.io/github/last-commit/janosh/awesome-normalizing-flows?label=Last+Commit)

A list of awesome resources for understanding and applying normalizing flows (NF): a relatively simple yet powerful new tool in statistics for constructing expressive probability distributions from simple base distributions using a chain (flow) of trainable smooth bijective transformations (diffeomorphisms).

<img src="assets/normalizing-flow.svg" alt="Normalizing Flow" width="1000">

<sup>_Figure inspired by [Lilian Weng](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models). Created in TikZ. [View source](https://github.com/janosh/tikz/tree/master/assets/normalizing-flow)._</sup>

<br>

## <img src="assets/toc.svg" alt="Contents" height="18px"> &nbsp;Table of Contents

1. [📝 Publications](#-publications)
   1. [🛠️ Applications](#️-applications)
2. [📺 Videos](#-videos)
3. [🌐 Blog Posts](#-blog-posts)
4. [📦 Packages](#-packages)
   1. [<img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Packages](#-pytorch-packages)
   2. [<img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Packages](#-tensorflow-packages)
   3. [<img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Packages](#-jax-packages)
5. [🧑‍💻 Code](#-code)
   1. [<img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Repos](#-pytorch-repos)
   2. [<img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Repos](#-jax-repos)
   3. [<img src="assets/ellipsis.svg" alt="Others" height="15px"> &nbsp;Others](#-others)
6. [🎉 Open to Suggestions!](#-open-to-suggestions)

<br>

## 📝 Publications

1. Apr 1, 2011 - [Iterative Gaussianization: from ICA to Random Rotations](https://arxiv.org/abs/1602.00229) by Laparra et. al.

   > Normalizing flows in the form of Gaussianization in an iterative format. Also shows connections to information theory.

2. Oct 30, 2014 - [Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) by Laurent Dinh, David Krueger, Yoshua Bengio.

   > Introduces the additive coupling layer (NICE) and shows how to use it for image generation and inpainting.

3. Feb 12, 2015 - [Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle.

   > Introduces MADE, a feed-forward network that uses carefully constructed binary masks on its weights to control the precise flow of information through the network. The masks ensure that each output unit receives signals only from input units that come before it in some arbitrary order. Yet all outputs can be computed in a single pass.<br>
   > A popular and efficient method to bestow flows with autoregressivity is to construct them from MADE nets.
   > <br><img src="assets/made.svg" alt="MADE"><br><sup>_Figure created in TikZ. [View source](https://github.com/janosh/tikz/tree/master/assets/made)._</sup>

4. May 21, 2015 - [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) by Danilo Rezende, Shakir Mohamed.

   > They show how to go beyond mean-field variational inference by using flows to increase the flexibility of the variational family.

5. May 27, 2016 - [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) by Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio.

   > They introduce the affine coupling layer (RNVP), a major improvement in terms of flexibility over the additive coupling layer (NICE) with unit Jacobian while keeping a single-pass forward and inverse transformation for fast sampling and density estimation, respectively.

6. Jun 15, 2016 - [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934) by Diederik Kingma et al.

7. Mar 6, 2017 - [Multiplicative Normalizing Flows for Variational Bayesian Neural Networks](https://arxiv.org/abs/1703.01961) by Christos Louizos, Max Welling.

   > They introduce a new type of variational Bayesian neural network that uses flows to generate auxiliary random variables which boost the flexibility of the variational family by multiplying the means of a fully-factorized Gaussian posterior over network parameters. This turns the usual diagonal covariance Gaussian into something that allows for multimodality and non-linear dependencies between network parameters.

8. May 19, 2017 - [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057) by George Papamakarios, Theo Pavlakou, Iain Murray.

   > Introduces MAF, a stack of autoregressive models forming a normalizing flow suitable for fast density estimation but slow at sampling. Analogous to Inverse Autoregressive Flow (IAF) except the forward and inverse passes are exchanged. Generalization of RNVP.

9. Mar 15, 2018 - [Sylvester Normalizing Flow for Variational Inference](https://arxiv.org/abs/1803.05649) by Rianne van den Berg, Leonard Hasenclever, Jakub M. Tomczak, Max Welling.

   > Introduces Sylvester normalizing flows which remove the single-unit bottleneck from planar flows for increased flexibility in the variational posterior.

10. April 3, 2018 - [Neural Autoregressive Flows](https://arxiv.org/abs/1804.00779) by Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville.

    > Unifies and generalize autoregressive and normalizing flow approaches, replacing the (conditionally) affine univariate transformations of MAF/IAF with a more general class of invertible univariate transformations expressed as monotonic neural networks. Also demonstrates that the proposed neural autoregressive flows (NAF) are universal approximators for continuous probability distributions. ([Author's Code](https://github.com/CW-Huang/NAF))

11. July 3, 2018 - [Deep Density Destructors](http://proceedings.mlr.press/v80/inouye18a.html) by Inouye & Ravikumar

    > Normalizing flows but from an iterative perspective. Features a Tree-based density estimator.

12. Jul 9, 2018 - [Glow: Generative Flow with Invertible 1x1 Convolutions](http://arxiv.org/abs/1807.03039) by Kingma, Dhariwal.

    > They show that flows using invertible 1x1 convolution achieve high likelihood on standard generative benchmarks and can efficiently synthesize realistic-looking, large images.

13. Oct 2, 2018 - [FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367) by Grathwohl & Chen et. al.

    > Uses Neural ODEs as a solver to produce continuous-time normalizing flows (CNF)

14. Nov 6, 2018 - [FloWaveNet : A Generative Flow for Raw Audio](https://arxiv.org/abs/1811.02155) by Kim et. al.

    > A flow-based generative model for raw audo synthesis. ([Author's Code](https://github.com/ksw0306/FloWaveNet))

15. Apr 9, 2019 - [Block Neural Autoregressive Flow](https://arxiv.org/abs/1904.04676) - De Cao et. al.

16. May 17, 2019 - [Integer Discrete Flows and Lossless Compression](https://arxiv.org/abs/1905.07376) by Hoogeboom et. al.

    > A normalizing flow to be used for ordinal discrete data. They introduce a flexible transformation layer called integer discrete coupling.

17. May 30, 2019 - [Graph Normalizing Flows](https://arxiv.org/abs/1905.13177) by Jenny Liu et al. A new, reversible graph network for prediction and generation.

    > They perform similarly to message passing neural networks on supervised tasks, but at significantly reduced memory use, allowing them to scale to larger graphs. Combined with a novel graph auto-encoder for unsupervised learning, graph normalizing flows are a generative model for graph structures.

18. Jul 21, 2019 - [Noise Regularization for Conditional Density Estimation](https://arxiv.org/abs/1907.08982) by Rothfuss et. al.

    > Normalizing flows for conditional density estimation. This paper proposes noise regularization to reduce overfitting. ([Blog](https://siboehm.com/articles/19/normalizing-flow-network) | )

19. Aug 25, 2019 - [Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/abs/1908.09257) by Kobyzev et al.

    > Another very thorough and very readable review article going through the basics of NFs as well as some of the state-of-the-art. Also highly recommended.

20. Jun 10, 2019 - [Neural Spline Flows](https://arxiv.org/abs/1906.04032) by Conor Durkan et. al.

    > Uses monotonic ration splines as a coupling layer. This is currently one of the state of the art.

21. Dec 5, 2019 - [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) by Papamakarios et al.

    > A thorough and very readable review article by some of the guys at DeepMind involved in the development of flows. Highly recommended.

22. Jan 15, 2020 - [Invertible Generative Modeling using Linear Rational Splines](https://arxiv.org/abs/2001.05168) by Dolatabadi et. al.

    > A successor to the Neural spline flows which features an easy-to-compute inverse.

23. Jan 17, 2020 - [Training Normalizing Flows with the Information Bottleneck for Competitive Generative Classification](https://arxiv.org/abs/2001.06448) by Ardizzone et. al.

    > They introduce a class of conditional normalizing flows with an information bottleneck objective. ([Author's Code](https://github.com/VLL-HD/exact_information_bottleneck))

24. Feb 16, 2020 - [Stochastic Normalizing Flows](https://arxiv.org/abs/2002.06707) by Hao Wu, Jonas Köhler, Frank Noé.

    > Introduces SNF, an arbitrary sequence of deterministic invertible functions (the flow) and stochastic processes such as MCMC or Langevin Dynamics. The aim is to increase expressiveness of the chosen deterministic invertible function, while the trainable flow improves sampling efficiency over pure MCMC ([Tweet](https://twitter.com/FrankNoeBerlin/status/1229734899034329103?s=19)).

25. Feb 21, 2020 - [Stochastic Normalizing Flows](https://arxiv.org/abs/2002.09547) by Liam Hodgkinson, Chris van der Heide, Fred Roosta, Michael W. Mahoney.

    > Name clash for a very different technique from the above SNF: an extension of continuous normalizing flows using stochastic differential equations (SDE). Treats Brownian motion in the SDE as a latent variable and approximates it by a flow. Aims to enable efficient training of neural SDEs which can be used for constructing efficient Markov chains.

26. Feb 24, 2020 - [Modeling Continuous Stochastic Processes with Dynamic Normalizing Flows](https://arxiv.org/abs/2002.10516) by Deng et. al.

    > They propose a normalizing flow using differential deformation of the Wiener process. Applied to time series. ([Tweet](https://twitter.com/r_giaquinto/status/1309648804824723464?s=09))

27. Feb 27, 2020 - [Gradient Boosted Normalizing Flows](https://arxiv.org/abs/2002.11896) by Giaquinto & Banerjee

    > Augment traditional normalizing flows with gradient boosting. They show that training multiple models can achieve good results and it's not necessary to have more complex distributions. ([Author's Code](https://github.com/robert-giaquinto/gradient-boosted-normalizing-flows))

28. Mar 4, 2020 - [Gaussianization Flows](https://arxiv.org/abs/2003.01941) by Meng et. al.

    > Uses a repeated composition of trainable kernel layers and orthogonal transformations. Very competitive versus some of the SOTA like Real-NVP, Glow and FFJORD. ([Author's Code](https://github.com/chenlin9/Gaussianization_Flows))

29. Mar 31, 2020 - [Flows for simultaneous manifold learning and density estimation](https://arxiv.org/abs/2003.13913) by Brehmer & Cranmer.

    > Normalizing flows that learn the data manifold and probability density function on that manifold. ([Tweet](https://twitter.com/kylecranmer/status/1250129080395223040?lang=es) | [Author's Code](https://github.com/johannbrehmer/manifold-flow))

30. April 8, 2020 - [Normalizing Flows with Multi-Scale Autoregressive Priors](https://arxiv.org/abs/2004.03891) by Mahajan & Bhattacharyya et. al.

    > Improves the representational power of flow-based models by introducing channel-wise dependencies in their latent space through multi-scale autoregressive priors (mAR) ([Author's Code](https://github.com/visinf/mar-scf))

31. Jun 3, 2020 - [Equivariant Flows: exact likelihood generative learning for symmetric densities](https://arxiv.org/abs/2006.02425) by Jonas Köhler, Leon Klein, Frank Noé.

    > Shows that distributions generated by equivariant NFs faithfully reproduce symmetries in the underlying density. Proposes building blocks for flows which preserve typical symmetries in physical/chemical many-body systems. Shows that symmetry-preserving flows can provide better generalization and sampling efficiency.

32. Jun 15, 2020 - [Why Normalizing Flows Fail to Detect Out-of-Distribution Data](https://proceedings.neurips.cc//paper/2020/hash/ecb9fe2fbb99c31f567e9823e884dbec-Abstract.html) by Kirichenko et. al.

    > This study how traditional normalizing flow models can suffer from out-of-distribution data. They offer a solution to combat this issue by modifying the coupling layers. ([Tweet](https://twitter.com/polkirichenko/status/1272715634544119809) | [Author's Code](https://github.com/PolinaKirichenko/flows_ood))

33. July 15, 2020 - [AdvFlow: Inconspicuous Black-box Adversarial Attacks using Normalizing Flows](https://arxiv.org/abs/2007.07435) by Dolatabadi etl. al.

    > An adversarial attack method on image classifiers that use normalizing flows. ([Author's Code](https://github.com/hmdolatabadi/AdvFlow))

34. Sept 21, 2020 - [Haar Wavelet based Block Autoregressive Flows for Trajectories](https://arxiv.org/abs/2009.09878) by Bhattacharyya et. al.
    > Introduce a Haar wavelet-based block autoregressive model.

<br>

### 🛠️ Applications

1. Aug 14, 2018 - [Analyzing Inverse Problems with Invertible Neural Networks](https://arxiv.org/abs/1808.04730) by Ardizzone et. al.

   > Normalizing flows for inverse problems.

2. Mar 9, 2019 - [NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport](https://arxiv.org/abs/1903.03704) by Hoffman et. al.

   > Uses normalizing flows in conjunction with Monte Carlo estimation to have more expressive distributions and better posterior estimation.

3. Jun 25, 2020 - [SRFlow: Learning the Super-Resolution Space with Normalizing Flow](https://arxiv.org/abs/2006.14200) by Lugmayr et. al.

   > Uses normalizing flows for super-resolution.

4. Jul 15, 2020 - [Faster Uncertainty Quantification for Inverse Problems with Conditional Normalizing Flows](https://arxiv.org/abs/2007.07985) by Siahkoohi et. al.

   > Uses conditional normalizing flows for inverse problems. ([Video](https://www.youtube.com/watch?v=nPvZIKaRBkI&feature=youtu.be))

5. Oct 13, 2020 - [Targeted free energy estimation via learned mappings](https://aip.scitation.org/doi/10.1063/5.0018903)
   > Normalizing flows used to estimate free energy differences.

<br>

## 📺 Videos

1. Oct 4, 2018 - [Sylvester Normalizing Flow for Variational Inference](https://youtu.be/VeYyUcIDVHI) by Rianne van den Berg.

   > Introduces Sylvester normalizing flows which remove the single-unit bottleneck from planar flows for increased flexibility in the variational posterior.

2. Mar 24, 2019 - [PixelCNN, Wavenet, Normalizing Flows - Santiago Pascual - UPC Barcelona 2018](https://www.youtube.com/watch?v=7XRpVKpbxq8&feature=youtu.be)

3. Sep 24, 2019 - [Graph Normalizing Flows](https://youtu.be/frMPP30QQgY) by Jenny Liu (University of Toronto, Vector Institute).

   > Introduces a new graph generating model for use e.g. in drug discovery, where training on molecules that are known to bind/dissolve/etc. may help to generate novel, similarly effective molecules.

4. Oct 9, 2019 - [A primer on normalizing flows](https://youtu.be/P4Ta-TZPVi0) by [Laurent Dinh](https://laurent-dinh.github.io) (Google Brain).

   > The first author on both the NICE and RNVP papers and one of the first in this field gives an introductory talk at "Machine Learning for Physics and the Physics Learning of, 2019".

5. Dec 6, 2019 - [What are normalizing flows?](https://youtu.be/i7LjDvsLWCg) by [Ari Seff](https://cs.princeton.edu/~aseff) (Princeton).

   > A great 3blue1brown-style video explaining the basics of normalizing flows.

6. [Flow Models](https://sites.google.com/view/berkeley-cs294-158-sp20/home#h.p_E-C2dsllTu6x) by [CS294-158-SP20 Deep, Unsupervised Spring Learning,, 2020](https://sites.google.com/view/berkeley-cs294-158-sp20/home) (Berkeley)

   > A really thorough explanation of normalizing flows. Also includes some sample code.

<br>

## 🌐 Blog Posts

1. Jan 17, 2018 - [Normalizing Flows Tutorial](https://blog.evjang.com/2018/01/nf1.html) by Eric Jang.

   > [Part 1](https://blog.evjang.com/2018/01/nf1.html): Distributions and Determinants. [Part 2](https://blog.evjang.com/2018/01/nf2.html): Modern Normalizing Flows. Lots of great graphics.

1. Apr 3, 2018 - [Normalizing Flows](https://akosiorek.github.io/ml/2018/04/03/norm_flows) by Adam Kosiorek.

1. Oct 13, 2018 - [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models) by Lilian Weng.

<br>

## 📦 Packages

<br>

### <img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Packages

1. Feb 9, 2020 - [`nflows`](https://github.com/bayesiains/nflows) by [bayesiains](https://homepages.inf.ed.ac.uk/imurray2/group)

   > A suite of most of the SOTA methods using PyTorch. From an ML group in Edinburgh. They created the current SOTA spline flows. Almost as complete as you'll find from a single repo.

<br>

### <img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Packages

1. [Jun 22, 2018](https://github.com/tensorflow/probability/commit/67a55b2ef5e94da203991ed102c9e41d085a840a#diff-b89d05520b0fa976a2c5c918c1114b17) - [TensorFlow Probability](https://tensorflow.org/probability)

   > Offers RNVP, MAF and other autoregressive models.

<br>

### <img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Packages

1. [Mar 9, 2020](https://github.com/Information-Fusion-Lab-Umass/NuX/commit/24e9d6fe28b63ac23469982c0cddd3e0d4ed5ffd) - [`NuX`](https://github.com/Information-Fusion-Lab-Umass/NuX) by Information Fusion Labs (UMass)

   > A library that offers normalizing flows using JAX as the backend. Has some SOTA methods. They also feature a surjective flow via quantization.

2. [Mar 23, 2020](https://github.com/ChrisWaites/jax-flows/commit/9d5af75d3bf89499b279f83e7ae9a637246cd6f1) - [`jax-flows`](https://github.com/ChrisWaites/jax-flows) by [Chris Waites](https://chriswaites.com/#)

   > Another library that has normalizing flows using JAX as the backend. Has some of the SOTA methods.

<br>

## 🧑‍💻 Code

<br>

### <img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Repos

1. Sep 1, 2018 - [`pytorch-flows`](https://github.com/ikostrikov/pytorch-flows) by Ilya Kostrikov.

   > PyTorch implementations of density estimation algorithms: MAF, RNVP, Glow.

2. Dec 30, 2018 - [`normalizing_flows`](https://github.com/kamenbliznashki/normalizing_flows) by Kamen Bliznashki.

   > Pytorch implementations of density estimation algorithms: BNAF, Glow, MAF, RealNVP, planar flows.

3. Feb 6, 2019 - [`pytorch_flows`](https://github.com/acids-ircam/pytorch_flows) by [acids-ircam](https://github.com/acids-ircam)

   > A great repo with some basic PyTorch implementations of normalizing flows from scratch.

4. Dec 9, 2019 - [`pytorch-normalizing-flows`](https://github.com/karpathy/pytorch-normalizing-flows) by Andrej Karpathy.

   > A Jupyter notebook with PyTorch implementations of the most commonly used flows: NICE, RNVP, MAF, Glow, NSF.

5. [Jul 3, 2020](https://github.com/DiffEqML/torchdyn/commit/349fbae0576a38bdb138633cfea4483706e11dd0#diff-2442d08f4c0753b452c9d2a8cd8a1a41) - [Density Estimation with Neural ODEs](https://torchdyn.readthedocs.io/en/latest/tutorials/07a_continuous_normalizing_flows.html) and [Density Estimation with FFJORDs](https://torchdyn.readthedocs.io/en/latest/tutorials/07b_ffjord.html) by [`torchdyn`](https://torchdyn.readthedocs.io)

   > Example of how to use FFJORD as a continuous normalizing flow (CNF). Based on the PyTorch suite `torchdyn` which offers continuous neural architectures.

6. July 19, 2020 - [`Normalizing Flows - Introduction (Part 1)`](https://pyro.ai/examples/normalizing_flows_i) by [pyro.ai](http://pyro.ai)

   > A tutorial about how to use the `pyro-ppl` library (based on PyTorch) to use Normalizing flows. They provide some SOTA methods including NSF and MAF. [Parts 2 and 3 coming later](https://github.com/pyro-ppl/pyro/issues/1992).

<br>

### <img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Repos

1. [Jul 19, 2019](https://github.com/pyro-ppl/numpyro/commit/781baf51752e5919c358362c2ef745a3df55f709#diff-177a79ac143e3f5df6fabf6604c28559) - [`Neural Transport`](https://pyro.ai/numpyro/examples/neutra) by [numpyro](http://num.pyro.ai/en/stable)

   > Features an example of how Normalizing flows can be used to get more robust posteriors from Monte Carlo methods. Uses the `numpyro` library which is a PPL with JAX as the backend. The NF implementations include the basic ones like IAF and BNAF.

<br>

### <img src="assets/ellipsis.svg" alt="Others" height="18px"> &nbsp;Others

1. Mar 21, 2017 - ['NormFlows'](https://github.com/andymiller/NormFlows)

   > Simple didactic example using [`autograd`](https://github.com/HIPS/autograd), so pretty low-level.

2. Jul 11, 2017 - [`normalizing_flows_overview.ipynb`](https://docs.pymc.io/notebooks/normalizing_flows_overview.html) by PyMC3.

   > A very helpful notebook showcasing how to work with flows in practice and comparing it to PyMC3's NUTS-based HMC kernel. Based on [Theano](https://github.com/Theano/Theano).

3. Jun 11, 2018 - [`destructive-deep-learning`](https://github.com/davidinouye/destructive-deep-learning) by [David Inouye](https://davidinouye.com)

   > Code base for the paper [Deep Density Destructors](http://proceedings.mlr.press/v80/inouye18a.html) by Inouye & Ravikumar (2018). An entire suite of iterative methods including tree-based as well as Gaussianization methods which are similar to normalizing flows except they converge iteratively instead of fully parametrized. That is, they still use bijective transforms, compute the Jacobian, check the likelihood and you can still sample and get probability density estimates. The only difference is you repeat the following two steps until convergence:
   >
   > 1. compute one layer or block layer (e.g. Marginal Gaussianization + PCA rotation)
   > 2. check for convergence (e.g log-likelihood using the change-of-variables formula)
   >
   > Table 1 in the paper has a good comparison with traditional NFs.

<br>

## 🎉 Open to Suggestions!

See something that's missing from this list? [PRs welcome!](https://github.com/janosh/awesome-normalizing-flows/edit/master/readme.md)

If you're unsure if a paper or resource belongs in this list, feel free to [open an issue](https://github.com/janosh/awesome-normalizing-flows/issues/new) and start a discussion. This repo is meant to be a community effort. So don't hesitate to voice an opinion.
