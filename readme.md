# Awesome Normalizing Flows

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Pull Requests Welcome](https://img.shields.io/badge/Pull%20Requests-welcome-brightgreen.svg)](https://github.com/janosh/awesome-normalizing-flows/pulls)
[![Markdown linting](https://github.com/janosh/awesome-normalizing-flows/workflows/Linter/badge.svg)](https://github.com/janosh/awesome-normalizing-flows/actions)

A list of awesome resources for understanding and applying normalizing flows (NF): a relatively simple yet powerful new tool in statistics for constructing expressive probability distributions from simple base distributions using a chain (flow) of trainable smooth bijective transformations (diffeomorphisms).

<img src="assets/normalizing-flow.svg" alt="Normalizing Flow" width="1000">

<sup>_Figure inspired by [Lilian Weng](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models). Created in TikZ. [View source](https://github.com/janosh/tikz/tree/main/assets/normalizing-flow)._</sup>

<br>

## <img src="assets/toc.svg" alt="Contents" height="18px"> &nbsp;Table of Contents

1. [📝 Publications](#-publications)
   1. [🛠️ Applications](#️-applications)
1. [📺 Videos](#-videos)
1. [📦 Packages](#-packages)
   1. [<img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Packages](#-pytorch-packages)
   1. [<img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Packages](#-tensorflow-packages)
   1. [<img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Packages](#-jax-packages)
   1. [<img src="assets/julia.svg" alt="Julia" height="15px"> &nbsp;Julia Packages](#-julia-packages)
1. [🧑‍💻 Code](#-code)
   1. [<img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Repos](#-pytorch-repos)
   1. [<img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Repos](#-jax-repos)
   1. [<img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Repos](#-tensorflow-repos)
   1. [<img src="assets/ellipsis.svg" alt="Others" height="15px"> &nbsp;Others](#-others)
1. [🌐 Blog Posts](#-blog-posts)
1. [🎉 Open to Suggestions!](#-open-to-suggestions)

<br>

## 📝 Publications

1. Apr 1, 2011 - [Iterative Gaussianization: from ICA to Random Rotations](https://arxiv.org/abs/1602.00229) by Laparra et. al.

   > Normalizing flows in the form of Gaussianization in an iterative format. Also shows connections to information theory.

2. Oct 30, 2014 - [Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) by Laurent Dinh, David Krueger, Yoshua Bengio.

   > Introduces the additive coupling layer (NICE) and shows how to use it for image generation and inpainting.

3. Feb 12, 2015 - [Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle.

   > Introduces MADE, a feed-forward network that uses carefully constructed binary masks on its weights to control the precise flow of information through the network. The masks ensure that each output unit receives signals only from input units that come before it in some arbitrary order. Yet all outputs can be computed in a single pass.<br>
   > A popular and efficient method to bestow flows with autoregressivity is to construct them from MADE nets.
   > <br><img src="assets/made.svg" alt="MADE"><br><sup>_Figure created in TikZ. [View source](https://github.com/janosh/tikz/tree/main/assets/made)._</sup>

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

11. July 3, 2018 - [Deep Density Destructors](https://proceedings.mlr.press/v80/inouye18a.html) by Inouye & Ravikumar

    > Normalizing flows but from an iterative perspective. Features a Tree-based density estimator.

12. Jul 9, 2018 - [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039) by Kingma, Dhariwal.

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

33. July 6, 2020 - [SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows](https://arxiv.org/abs/2007.02731) by Didrik Nielsen et. al.

    > They present a generalized framework that encompasses both Flows (deterministic maps) and VAEs (stochastic maps). By seeing deterministic maps `x = f(z)` as limiting cases of stochastic maps `x ~ p(x|z)`, the ELBO is reinterpreted as a change of variables formula for the stochastic maps. Moreover, they present a few examples of surjective layers using stochastic maps, which can be composed together with flow layers. ([Video](https://youtu.be/bXp8fk4MRXQ) | [Author's Code](https://github.com/didriknielsen/survae_flows))

34. July 15, 2020 - [AdvFlow: Inconspicuous Black-box Adversarial Attacks using Normalizing Flows](https://arxiv.org/abs/2007.07435) by Dolatabadi et. al.

    > An adversarial attack method on image classifiers that use normalizing flows. ([Author's Code](https://github.com/hmdolatabadi/AdvFlow))

35. Sept 21, 2020 - [Haar Wavelet based Block Autoregressive Flows for Trajectories](https://arxiv.org/abs/2009.09878) by Bhattacharyya et. al.
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

   > Uses conditional normalizing flows for inverse problems. ([Video](https://youtu.be/nPvZIKaRBkI))

5. Oct 13, 2020 - [Targeted free energy estimation via learned mappings](https://aip.scitation.org/doi/10.1063/5.0018903)
   > Normalizing flows used to estimate free energy differences.

6. Nov 2, 2020 - [On the Sentence Embeddings from Pre-trained Language Models](https://aclweb.org/anthology/2020.emnlp-main.733) by Li et al.

   > Proposes to use flows to transform anisotropic sentence embedding distributions from BERT to a smooth and isotropic Gaussian, learned through unsupervised objective. Demonstrates performance gains over SOTA sentence embeddings on semantic textual similarity tasks. Code available at <https://github.com/bohanli/BERT-flow>.

7. Dec 6, 2020 - [Normalizing Kalman Filters for Multivariate Time Series Analysis](https://proceedings.neurips.cc/paper/2020/hash/1f47cef5e38c952f94c5d61726027439-Abstract.html)

   > Augments state space models with normalizing flows and thereby mitigates imprecisions stemming from idealized assumptions. Aimed at forecasting real-world data and handling varying levels of missing data. (Also available at [Amazon Science](https://amazon.science/publications/normalizing-kalman-filters-for-multivariate-time-series-analysis).)

<br>

## 📺 Videos

1. Oct 4, 2018 - [Sylvester Normalizing Flow for Variational Inference](https://youtu.be/VeYyUcIDVHI) by Rianne van den Berg.

   > Introduces Sylvester normalizing flows which remove the single-unit bottleneck from planar flows for increased flexibility in the variational posterior.

2. Mar 24, 2019 - [PixelCNN, Wavenet, Normalizing Flows - Santiago Pascual - UPC Barcelona 2018](https://youtu.be/7XRpVKpbxq8)

3. Sep 24, 2019 - [Graph Normalizing Flows](https://youtu.be/frMPP30QQgY) by Jenny Liu (University of Toronto, Vector Institute).

   > Introduces a new graph generating model for use e.g. in drug discovery, where training on molecules that are known to bind/dissolve/etc. may help to generate novel, similarly effective molecules.

4. Oct 9, 2019 - [A primer on normalizing flows](https://youtu.be/P4Ta-TZPVi0) by [Laurent Dinh](https://laurent-dinh.github.io) (Google Brain).

   > The first author on both the NICE and RNVP papers and one of the first in this field gives an introductory talk at "Machine Learning for Physics and the Physics Learning of, 2019".

5. Dec 6, 2019 - [What are normalizing flows?](https://youtu.be/i7LjDvsLWCg) by [Ari Seff](https://cs.princeton.edu/~aseff) (Princeton).

   > A great 3blue1brown-style video explaining the basics of normalizing flows.

6. [Flow Models](https://sites.google.com/view/berkeley-cs294-158-sp20/home#h.p_E-C2dsllTu6x) by [CS294-158-SP20 Deep, Unsupervised Spring Learning,, 2020](https://sites.google.com/view/berkeley-cs294-158-sp20/home) (Berkeley)

   > A really thorough explanation of normalizing flows. Also includes some sample code.

7. Nov 23, 2020 - [Introduction to Normalizing Flows](https://youtu.be/u3vVyFVU_lI) by [Marcus Brubaker](https://mbrubake.github.io) (York University, Toronto)

   > A great introduction to normalizing flows by one of the creators of [Stan](https://mc-stan.org) presented at ECCV 2020. The tutorial also provides an excellent review of various practical implementations.

8. January 16, 2021 - [Normalizing Flows - Motivations, The Big Idea & Essential Foundations](https://youtu.be/IuXU2dBOJyw) by [Kapil Sachdeva](https://github.com/ksachdeva)

   > A comprehensive tutorial on flows explaining the challenges addressed by this class of algorithm. Provides intuition on how to address those challenges, and explains the underlying mathematics using a simple step by step approach.

<br>

## 📦 Packages

<br>

### <img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Packages

1. Sep 7, 2018 - [`FrEIA`](https://github.com/VLL-HD/FrEIA) by [VLL Heidelberg](https://hci.iwr.uni-heidelberg.de/vislearn) ![GitHub repo stars](https://img.shields.io/github/stars/VLL-HD/FrEIA)

   > The Framework for Easily Invertible Architectures (FrEIA) is based on RNVP flows. Easy to setup, it allows to define complex Invertible Neural Networks (INNs) from simple invertible building blocks.

2. Feb 9, 2020 - [`nflows`](https://github.com/bayesiains/nflows) by [Bayesiains](https://homepages.inf.ed.ac.uk/imurray2/group) ![GitHub repo stars](https://img.shields.io/github/stars/bayesiains/nflows)

   > A suite of most of the SOTA methods using PyTorch. From an ML group in Edinburgh. They created the current SOTA spline flows. Almost as complete as you'll find from a single repo.

<br>

### <img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Packages

1. [Jun 22, 2018](https://github.com/tensorflow/probability/commit/67a55b2ef5e94da203991ed102c9e41d085a840a#diff-b89d05520b0fa976a2c5c918c1114b17) - [TensorFlow Probability](https://tensorflow.org/probability) ![GitHub repo stars](https://img.shields.io/github/stars/tensorflow/probability)

   > Large first-party library that offers RNVP, MAF among other autoregressive models plus a collection of composable bijectors.

<br>

### <img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Packages

1. [Mar 9, 2020](https://github.com/Information-Fusion-Lab-Umass/NuX/commit/24e9d6fe28b63ac23469982c0cddd3e0d4ed5ffd) - [`NuX`](https://github.com/Information-Fusion-Lab-Umass/NuX) by Information Fusion Labs (UMass) ![GitHub repo stars](https://img.shields.io/github/stars/Information-Fusion-Lab-Umass/NuX)

   > A library that offers normalizing flows using JAX as the backend. Has some SOTA methods. They also feature a surjective flow via quantization.

2. [Mar 23, 2020](https://github.com/ChrisWaites/jax-flows/commit/9d5af75d3bf89499b279f83e7ae9a637246cd6f1) - [`jax-flows`](https://github.com/ChrisWaites/jax-flows) by [Chris Waites](https://chriswaites.com) ![GitHub repo stars](https://img.shields.io/github/stars/ChrisWaites/jax-flows)

   > Another library that has normalizing flows using JAX as the backend. Has some of the SOTA methods.

3. Jun 17, 2021 - [`pzflow`](https://github.com/jfcrenshaw/pzflow) by [John Franklin Crenshaw](https://jfcrenshaw.github.io) ![GitHub repo stars](https://img.shields.io/github/stars/jfcrenshaw/pzflow)

   > A package that focuses on probabilistic modeling of tabular data, with a focus on sampling and posterior calculation.

<br>

### <img src="assets/julia.svg" alt="Julia" height="15px"> &nbsp;Julia Packages

1. [Feb 7, 2020](https://github.com/slimgroup/InvertibleNetworks.jl/commit/e72da19f78085d270940722606922ae318dfe53d) - [`InvertibleNetworks.jl`](https://github.com/slimgroup/InvertibleNetworks.jl) by [SLIM](https://slim.gatech.edu) ![GitHub repo stars](https://img.shields.io/github/stars/slimgroup/InvertibleNetworks.jl)

   > A Flux compatible library implementing invertible neural networks and normalizing flows using memory-efficient backpropagation. Uses manually implemented gradients to take advantage of the invertibility of building blocks, which allows for scaling to large-scale problem sizes.

<br>

## 🧑‍💻 Code

<br>

### <img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Repos

1. Sep 1, 2018 - [`pytorch-flows`](https://github.com/ikostrikov/pytorch-flows) by Ilya Kostrikov. ![GitHub repo stars](https://img.shields.io/github/stars/ikostrikov/pytorch-flows)

   > PyTorch implementations of density estimation algorithms: MAF, RNVP, Glow.

2. Dec 30, 2018 - [`normalizing_flows`](https://github.com/kamenbliznashki/normalizing_flows) by Kamen Bliznashki. ![GitHub repo stars](https://img.shields.io/github/stars/kamenbliznashki/normalizing_flows)

   > Pytorch implementations of density estimation algorithms: BNAF, Glow, MAF, RealNVP, planar flows.

3. Feb 6, 2019 - [`pytorch_flows`](https://github.com/acids-ircam/pytorch_flows) by [acids-ircam](https://github.com/acids-ircam) ![GitHub repo stars](https://img.shields.io/github/stars/acids-ircam/pytorch_flows)

   > A great repo with some basic PyTorch implementations of normalizing flows from scratch.

4. Dec 9, 2019 - [`pytorch-normalizing-flows`](https://github.com/karpathy/pytorch-normalizing-flows) by Andrej Karpathy. ![GitHub repo stars](https://img.shields.io/github/stars/karpathy/pytorch-normalizing-flows)

   > A Jupyter notebook with PyTorch implementations of the most commonly used flows: NICE, RNVP, MAF, Glow, NSF.

5. [Jul 3, 2020](https://github.com/DiffEqML/torchdyn/commit/349fbae0576a38bdb138633cfea4483706e11dd0#diff-2442d08f4c0753b452c9d2a8cd8a1a41) - [Density Estimation with Neural ODEs](https://torchdyn.readthedocs.io/en/latest/tutorials/07a_continuous_normalizing_flows.html) and [Density Estimation with FFJORDs](https://torchdyn.readthedocs.io/en/latest/tutorials/07b_ffjord.html) by [`torchdyn`](https://torchdyn.readthedocs.io) ![GitHub repo stars](https://img.shields.io/github/stars/DiffEqML/torchdyn)

   > Example of how to use FFJORD as a continuous normalizing flow (CNF). Based on the PyTorch suite `torchdyn` which offers continuous neural architectures.

6. July 19, 2020 - [`Normalizing Flows - Introduction (Part 1)`](https://pyro.ai/examples/normalizing_flows_i) by [pyro.ai](https://pyro.ai) ![GitHub repo stars](https://img.shields.io/github/stars/pyro-ppl/pyro)

   > A tutorial about how to use the `pyro-ppl` library (based on PyTorch) to use Normalizing flows. They provide some SOTA methods including NSF and MAF. [Parts 2 and 3 coming later](https://github.com/pyro-ppl/pyro/issues/1992).
   
7. Aug 21, 2021 - [NICE: Non-linear Independent Components Estimation](https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/NICE_Non_linear_Independent_Components_Estimation) by Maxime Vandegar. ![GitHub repo stars](https://img.shields.io/github/stars/MaximeVandegar/Papers-in-100-Lines-of-Code)

   > PyTorch implementation that reproduces results from the paper NICE in about 100 lines of code.
<br>

### <img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Repos

1. [Jul 19, 2019](https://github.com/pyro-ppl/numpyro/commit/781baf51752e5919c358362c2ef745a3df55f709#diff-177a79ac143e3f5df6fabf6604c28559) - [`Neural Transport`](https://pyro.ai/numpyro/examples/neutra) by [numpyro](https://num.pyro.ai) ![GitHub repo stars](https://img.shields.io/github/stars/pyro-ppl/numpyro)

   > Features an example of how Normalizing flows can be used to get more robust posteriors from Monte Carlo methods. Uses the `numpyro` library which is a PPL with JAX as the backend. The NF implementations include the basic ones like IAF and BNAF.

<br>

### <img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Repos

1. Jun 12, 2020 - [Variational Inference using Normalizing Flows (VINF)](https://github.com/pierresegonne/VINF) by Pierre Segonne ![GitHub repo stars](https://img.shields.io/github/stars/pierresegonne/VINF)

   > This repository provides a hands-on TensorFlow implementation of Normalizing Flows as presented in the [paper](https://arxiv.org/pdf/1505.05770.pdf) introducing the concept (D. Rezende & S. Mohamed).

2. Nov 2, 2020 - [BERT-flow](https://github.com/bohanli/BERT-flow) by Bohan Li ![GitHub repo stars](https://img.shields.io/github/stars/bohanli/BERT-flow)

   > TensorFlow implementation of "On the Sentence Embeddings from Pre-trained Language Models" (EMNLP 2020)

<br>

### <img src="assets/ellipsis.svg" alt="Others" height="18px"> &nbsp;Others

1. Mar 21, 2017 - ['NormFlows'](https://github.com/andymiller/NormFlows) ![GitHub repo stars](https://img.shields.io/github/stars/andymiller/NormFlows)

   > Simple didactic example using [`autograd`](https://github.com/HIPS/autograd), so pretty low-level.

2. Jul 11, 2017 - [`normalizing_flows_overview.ipynb`](https://docs.pymc.io/notebooks/normalizing_flows_overview.html) by PyMC3. ![GitHub repo stars](https://img.shields.io/github/stars/pymc-devs/pymc3)

   > A very helpful notebook showcasing how to work with flows in practice and comparing it to PyMC3's NUTS-based HMC kernel. Based on [Theano](https://github.com/Theano/Theano).

3. Jun 11, 2018 - [`destructive-deep-learning`](https://github.com/davidinouye/destructive-deep-learning) by [David Inouye](https://davidinouye.com) ![GitHub repo stars](https://img.shields.io/github/stars/davidinouye/destructive-deep-learning)

   > Code base for the paper [Deep Density Destructors](https://proceedings.mlr.press/v80/inouye18a.html) by Inouye & Ravikumar (2018). An entire suite of iterative methods including tree-based as well as Gaussianization methods which are similar to normalizing flows except they converge iteratively instead of fully parametrized. That is, they still use bijective transforms, compute the Jacobian, check the likelihood and you can still sample and get probability density estimates. The only difference is you repeat the following two steps until convergence:
   >
   > 1. compute one layer or block layer (e.g. Marginal Gaussianization + PCA rotation)
   > 2. check for convergence (e.g log-likelihood using the change-of-variables formula)
   >
   > Table 1 in the paper has a good comparison with traditional NFs.

<br>

## 🌐 Blog Posts

1. Jan 17, 2018 - [Normalizing Flows Tutorial](https://blog.evjang.com/2018/01/nf1.html) by Eric Jang.

   > [Part 1](https://blog.evjang.com/2018/01/nf1.html): Distributions and Determinants. [Part 2](https://blog.evjang.com/2018/01/nf2.html): Modern Normalizing Flows. Lots of great graphics.

1. Apr 3, 2018 - [Normalizing Flows](https://akosiorek.github.io/ml/2018/04/03/norm_flows) by Adam Kosiorek.

1. May 19, 2018 - [Change of Variables: A Precursor to Normalizing Flows](http://ruishu.io/2018/05/19/change-of-variables) by Rui Shu

1. Oct 13, 2018 - [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models) by Lilian Weng.

1. Oct 21, 2018 - [Change of Variables for Normalizing Flows](https://nealjean.com/ml/change-of-variables) by Neal Jean

<br>

## 🎉 Open to Suggestions

See something that's missing from this list? [PRs welcome!](https://github.com/janosh/awesome-normalizing-flows/edit/main/readme.md)

Papers should be peer-reviewed and published in a journal. If you're unsure if a paper or resource belongs in this list, feel free to [open an issue](https://github.com/janosh/awesome-normalizing-flows/issues/new) or [start a discussion](https://github.com/janosh/awesome-normalizing-flows/discussions). This repo is meant to be a community effort. So don't hesitate to voice an opinion.
