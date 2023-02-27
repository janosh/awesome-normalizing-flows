<h1 align="center">
Awesome Normalizing Flows
</h1>

<h4 align="center">

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Pull Requests Welcome](https://img.shields.io/badge/Pull%20Requests-welcome-brightgreen.svg?logo=github)](#-contributing)
[![Link Check](https://github.com/janosh/awesome-normalizing-flows/actions/workflows/link-check.yml/badge.svg)](https://github.com/janosh/awesome-normalizing-flows/actions/workflows/link-check.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/awesome-normalizing-flows/main.svg)](https://results.pre-commit.ci/latest/github/janosh/awesome-normalizing-flows/main)
</h4>

A list of awesome resources for understanding and applying normalizing flows (NF): a relatively simple yet powerful new tool in statistics for constructing expressive probability distributions from simple base distributions using a chain (flow) of trainable smooth bijective transformations (diffeomorphisms).

<a href="https://github.com/janosh/tikz/tree/main/assets/normalizing-flow">
   <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/janosh/tikz/main/assets/normalizing-flow/normalizing-flow-white.svg">
      <img alt="Diagram of the slow (sequential) forward pass of a Masked Autoregressive Flow (MAF) layer" src="https://raw.githubusercontent.com/janosh/tikz/main/assets/normalizing-flow/normalizing-flow.svg">
   </picture>
</a>

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
1. [🧑‍💻 Repos](#-repos)
   1. [<img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Repos](#-pytorch-repos)
   1. [<img src="assets/jax.svg" alt="JAX" height="15px"> &nbsp;JAX Repos](#-jax-repos)
   1. [<img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Repos](#-tensorflow-repos)
   1. [<img src="assets/other.svg" alt="Other" height="15px"> &nbsp;Other Repos](#-other-repos)
1. [🌐 Blog Posts](#-blog-posts)
1. [🚧 Contributing](#-contributing)

<br>

## 📝 Publications

1. 2023-01-03 - [FInC Flow: Fast and Invertible k×k Convolutions for Normalizing Flows](https://arxiv.org/abs/2301.09266) by Kallapa, Nagar et al.<br>
   propose a k×k convolutional layer and Deep Normalizing Flow architecture which i) has a fast parallel inversion algorithm with running time O(nk^2) (n is height and width of the input image and k is kernel size), ii) masks the minimal amount of learnable parameters in a layer. iii) gives better forward pass and sampling times comparable to other k×k convolution-based models on real-world benchmarks. We provide an implementation of the proposed parallel algorithm for sampling using our invertible convolutions on GPUs. [[Code](https://github.com/aditya-v-kallappa/FInCFlow)]

1. 2022-08-18 - [ManiFlow: Implicitly Representing Manifolds with Normalizing Flows](https://arxiv.org/abs/2208.08932) by Postels, Danelljan et al.<br>
   The invertibility constraint of NFs imposes limitations on data distributions that reside on lower dimensional manifolds embedded in higher dimensional space. This is often bypassed by adding noise to the data which impacts generated sample quality. This work generates samples from the original data distribution given full knowledge of perturbed distribution and noise model. They establish NFs trained on perturbed data implicitly represent the manifold in regions of maximum likelihood, then propose an optimization objective that recovers the most likely point on the manifold given a sample from the perturbed distribution.

1. 2022-06-03 - [Graphical Normalizing Flows](https://arxiv.org/abs/2006.02548) by Wehenkel, Louppe<br>
   This work revisits coupling and autoregressive transformations as probabilistic graphical models showing they reduce to Bayesian networks with a pre-defined topology. From this new perspective, the authors propose the graphical normalizing flow, a new invertible transformation with either a prescribed or a learnable graphical structure. This model provides a promising way to inject domain knowledge into normalizing flows while preserving both the interpretability of Bayesian networks and the representation capacity of normalizing flows. [[Code](https://github.com/AWehenkel/Graphical-Normalizing-Flows)]

1. 2022-05-16 - [Multi-scale Attention Flow for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2205.07493) by Feng, Xu et al.<br>
   Proposes a novel non-autoregressive deep learning model, called Multi-scale Attention Normalizing Flow(MANF), where one integrates multi-scale attention and relative position information and the multivariate data distribution is represented by the conditioned normalizing flow.

1. 2022-01-14 - [E(n) Equivariant Normalizing Flows](https://arxiv.org/abs/2105.09016) by Satorras, Hoogeboom et al.<br>
   Introduces equivariant graph neural networks into the normalizing flow framework which combine to give invertible equivariant functions. Demonstrates their flow beats prior equivariant models and allows sampling of molecular configurations with positions, atom types and charges.

1. 2021-07-03 - [CInC Flow: Characterizable Invertible 3x3 Convolution](https://arxiv.org/abs/2107.01358) by Nagar, Dufraisse et al.<br>
   Seeks to improve expensive convolutions. They investigate the conditions for when 3x3 convolutions are invertible under which conditions (e.g. padding) and saw successful speedups. Furthermore, they developed a more expressive, invertible _Quad coupling_ layer. [[Code](https://github.com/Naagar/Normalizing_Flow_3x3_inv)]

1. 2021-04-14 - [Orthogonalizing Convolutional Layers with the Cayley Transform](https://arxiv.org/abs/2104.07167) by Trockman, Kolter<br>
   Parametrizes the multichannel convolution to be orthogonal via the Cayley transform (skew-symmetric convolutions in the Fourier domain). This enables the inverse to be computed efficiently. [[Code](https://github.com/locuslab/orthogonal-convolutions)]

1. 2021-04-14 - [Improving Normalizing Flows via Better Orthogonal Parameterizations](https://invertibleworkshop.github.io/INNF_2019/accepted_papers/pdfs/INNF_2019_paper_30.pdf) by Goliński, Lezcano-Casado et al.<br>
   Parametrizes the 1x1 convolution via the exponential map and the Cayley map. They demonstrate an improved optimization for the Sylvester normalizing flows.

1. 2020-09-28 - [Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows](https://arxiv.org/abs/2002.06103) by Rasul, Sheikh et al.<br>
   Models the multi-variate temporal dynamics of time series via an autoregressive deep learning model, where the data distribution is represented by a conditioned normalizing flow. [[OpenReview.net](https://openreview.net/forum?id=WiGQBFuVRv)] [[Code](https://github.com/zalandoresearch/pytorch-ts)]

1. 2020-09-21 - [Haar Wavelet based Block Autoregressive Flows for Trajectories](https://arxiv.org/abs/2009.09878) by Bhattacharyya, Straehle et al.<br>
   Introduce a Haar wavelet-based block autoregressive model.

1. 2020-07-15 - [AdvFlow: Inconspicuous Black-box Adversarial Attacks using Normalizing Flows](https://arxiv.org/abs/2007.07435) by Dolatabadi, Erfani et al.<br>
   An adversarial attack method on image classifiers that use normalizing flows. [[Code](https://github.com/hmdolatabadi/AdvFlow)]

1. 2020-07-06 - [SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows](https://arxiv.org/abs/2007.02731) by Nielsen, Jaini et al.<br>
   They present a generalized framework that encompasses both Flows (deterministic maps) and VAEs (stochastic maps). By seeing deterministic maps `x = f(z)` as limiting cases of stochastic maps `x ~ p(x|z)`, the ELBO is reinterpreted as a change of variables formula for the stochastic maps. Moreover, they present a few examples of surjective layers using stochastic maps, which can be composed together with flow layers. [[Video](https://youtu.be/bXp8fk4MRXQ)] [[Code](https://github.com/didriknielsen/survae_flows)]

1. 2020-06-15 - [Why Normalizing Flows Fail to Detect Out-of-Distribution Data](https://arxiv.org/abs/2006.08545) by Kirichenko, Izmailov et al.<br>
   This study how traditional normalizing flow models can suffer from out-of-distribution data. They offer a solution to combat this issue by modifying the coupling layers. [[Tweet](https://twitter.com/polkirichenko/status/1272715634544119809)] [[Code](https://github.com/PolinaKirichenko/flows_ood)]

1. 2020-06-03 - [Equivariant Flows: exact likelihood generative learning for symmetric densities](https://arxiv.org/abs/2006.02425) by Köhler, Klein et al.<br>
   Shows that distributions generated by equivariant NFs faithfully reproduce symmetries in the underlying density. Proposes building blocks for flows which preserve typical symmetries in physical/chemical many-body systems. Shows that symmetry-preserving flows can provide better generalization and sampling efficiency.

1. 2020-06-02 - [The Convolution Exponential and Generalized Sylvester Flows](https://arxiv.org/abs/2006.01910) by Hoogeboom, Satorras et al.<br>
   Introduces exponential convolution to add the spatial dependencies in linear layers as an improvement of the 1x1 convolutions. It uses matrix exponentials to create cheap and invertible layers. They also use this new architecture to create _convolutional Sylvester flows_ and _graph convolutional exponentials_. [[Code](https://github.com/ehoogeboom/convolution_exponential_and_sylvester)]

1. 2020-05-11 - [iUNets: Fully invertible U-Nets with Learnable Upand Downsampling](https://arxiv.org/abs/2005.05220) by Etmann, Ke et al.<br>
   Extends the classical UNet to be fully invertible by enabling invertible, orthogonal upsampling and downsampling layers. It is rather efficient so it should be able to enable stable training of deeper and larger networks.

1. 2020-04-08 - [Normalizing Flows with Multi-Scale Autoregressive Priors](https://arxiv.org/abs/2004.03891) by Mahajan, Bhattacharyya et al.<br>
   Improves the representational power of flow-based models by introducing channel-wise dependencies in their latent space through multi-scale autoregressive priors (mAR). [[Code](https://github.com/visinf/mar-scf)]

1. 2020-03-31 - [Flows for simultaneous manifold learning and density estimation](https://arxiv.org/abs/2003.13913) by Brehmer, Cranmer<br>
   Normalizing flows that learn the data manifold and probability density function on that manifold. [[Tweet](https://twitter.com/kylecranmer/status/1250129080395223040)] [[Code](https://github.com/johannbrehmer/manifold-flow)]

1. 2020-03-04 - [Gaussianization Flows](https://arxiv.org/abs/2003.01941) by Meng, Song et al.<br>
   Uses a repeated composition of trainable kernel layers and orthogonal transformations. Very competitive versus some of the SOTA like Real-NVP, Glow and FFJORD. [[Code](https://github.com/chenlin9/Gaussianization_Flows)]

1. 2020-02-27 - [Gradient Boosted Normalizing Flows](https://arxiv.org/abs/2002.11896) by Giaquinto, Banerjee<br>
   Augment traditional normalizing flows with gradient boosting. They show that training multiple models can achieve good results and it's not necessary to have more complex distributions. [[Code](https://github.com/robert-giaquinto/gradient-boosted-normalizing-flows)]

1. 2020-02-24 - [Modeling Continuous Stochastic Processes with Dynamic Normalizing Flows](https://arxiv.org/abs/2002.10516) by Deng, Chang et al.<br>
   They propose a normalizing flow using differential deformation of the Wiener process. Applied to time series. [[Tweet](https://twitter.com/r_giaquinto/status/1309648804824723464)]

1. 2020-02-21 - [Stochastic Normalizing Flows](https://arxiv.org/abs/2002.09547) by Hodgkinson, Heide et al.<br>
   Name clash for a very different technique from the above SNF: an extension of continuous normalizing flows using stochastic differential equations (SDE). Treats Brownian motion in the SDE as a latent variable and approximates it by a flow. Aims to enable efficient training of neural SDEs which can be used for constructing efficient Markov chains.

1. 2020-02-16 - [Stochastic Normalizing Flows (SNF)](https://arxiv.org/abs/2002.06707) by Wu, Köhler et al.<br>
   Introduces SNF, an arbitrary sequence of deterministic invertible functions (the flow) and stochastic processes such as MCMC or Langevin Dynamics. The aim is to increase expressiveness of the chosen deterministic invertible function, while the trainable flow improves sampling efficiency over pure MCMC [[Tweet](https://twitter.com/FrankNoeBerlin/status/1229734899034329103)).]

1. 2020-01-17 - [Training Normalizing Flows with the Information Bottleneck for Competitive Generative Classification](https://arxiv.org/abs/2001.06448) by Ardizzone, Mackowiak et al.<br>
   They introduce a class of conditional normalizing flows with an information bottleneck objective. [[Code](https://github.com/VLL-HD/exact_information_bottleneck)]

1. 2020-01-15 - [Invertible Generative Modeling using Linear Rational Splines](https://arxiv.org/abs/2001.05168) by Dolatabadi, Erfani et al.<br>
   A successor to the Neural spline flows which features an easy-to-compute inverse.

1. 2019-12-05 - [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) by Papamakarios, Nalisnick et al.<br>
   A thorough and very readable review article by some of the guys at DeepMind involved in the development of flows. Highly recommended.

1. 2019-09-14 - [Unconstrained Monotonic Neural Networks](https://arxiv.org/abs/1908.05164) by Wehenkel, Louppe<br>
   UMNN relaxes the constraints on weights and activation functions of monotonic neural networks by setting the derivative of the transformation as the output of an unconstrained neural network. The transformation itself is computed by numerical integration (Clenshaw-Curtis quadrature) of the derivative. [[Code](https://github.com/AWehenkel/UMNN)]

1. 2019-08-25 - [Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/abs/1908.09257) by Kobyzev, Prince et al.<br>
   Another very thorough and very readable review article going through the basics of NFs as well as some of the state-of-the-art. Also highly recommended.

1. 2019-07-21 - [Noise Regularization for Conditional Density Estimation](https://arxiv.org/abs/1907.08982) by Rothfuss, Ferreira et al.<br>
   Normalizing flows for conditional density estimation. This paper proposes noise regularization to reduce overfitting. [[Blog](https://siboehm.com/articles/19/normalizing-flow-network)]

1. 2019-07-18 - [MintNet: Building Invertible Neural Networks with Masked Convolutions](https://arxiv.org/abs/1907.07945) by Song, Meng et al.<br>
   Creates an autoregressive-like coupling layer via masked convolutions which is fast and efficient to evaluate. [[Code](https://github.com/ermongroup/mintnet)]

1. 2019-07-18 - [Densely connected normalizing flows](https://arxiv.org/abs/2106.04627) by Grcić, Grubišić et al.<br>
   Creates a nested coupling structure to add more expressivity to standard coupling layers. They also utilize slicing/factorization for dimensionality reduction and Nystromer for the coupling layer conditioning network. They achieved SOTA results for normalizing flow models. [[Code](https://github.com/matejgrcic/DenseFlow)]

1. 2019-06-15 - [Invertible Convolutional Flow](https://proceedings.neurips.cc/paper/2019/hash/b1f62fa99de9f27a048344d55c5ef7a6-Abstract.html) by Karami, Schuurmans et al.<br>
   Introduces convolutional layers that are circular and symmetric. The layer is invertible and cheap to evaluate. They also showcase how one can design non-linear elementwise bijectors that induce special properties via constraining the loss function. [[Code](https://github.com/Karami-m/Invertible-Convolutional-Flow)]

1. 2019-06-15 - [Invertible Convolutional Networks](https://invertibleworkshop.github.io/INNF_2019/accepted_papers/pdfs/INNF_2019_paper_26.pdf) by Finzi, Izmailov et al.<br>
   Showcases how standard convolutional layers can be made invertible via Fourier transformations. They also introduce better activations which might be better suited to normalizing flows, e.g. SneakyRELU

1. 2019-06-10 - [Neural Spline Flows](https://arxiv.org/abs/1906.04032) by Durkan, Bekasov et al.<br>
   Uses monotonic ration splines as a coupling layer. This is currently one of the state of the art.

1. 2019-05-30 - [Graph Normalizing Flows](https://arxiv.org/abs/1905.13177) by Liu, Kumar et al.<br>
   A new, reversible graph network for prediction and generation. They perform similarly to message passing neural networks on supervised tasks, but at significantly reduced memory use, allowing them to scale to larger graphs. Combined with a novel graph auto-encoder for unsupervised learning, graph normalizing flows are a generative model for graph structures.

1. 2019-05-24 - [Fast Flow Reconstruction via Robust Invertible n x n Convolution](https://arxiv.org/abs/1905.10170) by Truong, Luu et al.<br>
   Seeks to overcome the limitation of 1x1 convolutions and proposes invertible nxn convolutions via a clever convolutional _affine_ function.

1. 2019-05-17 - [Integer Discrete Flows and Lossless Compression](https://arxiv.org/abs/1905.07376) by Hoogeboom, Peters et al.<br>
   A normalizing flow to be used for ordinal discrete data. They introduce a flexible transformation layer called integer discrete coupling.

1. 2019-04-09 - [Block Neural Autoregressive Flow](https://arxiv.org/abs/1904.04676)) by Cao, Titov et al.<br>
   Introduces (B-NAF), a more efficient probability density approximator. Claims to be competitive with other flows across datasets while using orders of magnitude fewer parameters.

1. 2019-04-09 - [Block Neural Autoregressive Flow](https://arxiv.org/abs/1904.04676) by Wehenkel, Louppe<br>
   As an alternative to hand-crafted bijections, Huang et al. (2018) proposed NAF, a universal approximator for density functions. Their flow is a neural net whose parameters are predicted by another NN. The latter grows quadratically with the size of the former which is inefficient. We propose block neural autoregressive flow (B-NAF), a much more compact universal approximator of density functions, where we model a bijection directly using a single feed-forward network. Invertibility is ensured by carefully designing affine transformations with block matrices that make the flow autoregressive and monotone. We compare B-NAF to NAF and show our flow is competitive across datasets while using orders of magnitude fewer parameters. [[Code](https://github.com/nicola-decao/BNAF)]

1. 2019-02-19 - [MaCow: Masked Convolutional Generative Flow](https://arxiv.org/abs/1902.04208) by Ma, Kong et al.<br>
   Introduces a masked convolutional generative flow (MaCow) layer using a small kernel to capture local connectivity. They showed some improvement over the GLOW model while being fast and stable.

1. 2019-01-30 - [Emerging Convolutions for Generative Normalizing Flows](https://arxiv.org/abs/1901.11137) by Hoogeboom, Berg et al.<br>
   Introduces autoregressive-like convolutional layers that operate on the channel **and** spatial axes. This improved upon the performance of image datasets compared to the standard 1x1 Convolutions. The trade-off is that the inverse operator is quite expensive however the authors provide a fast C++ implementation. [[Code](https://github.com/ehoogeboom/emerging)]

1. 2018-11-06 - [FloWaveNet : A Generative Flow for Raw Audio](https://arxiv.org/abs/1811.02155) by Kim, Lee et al.<br>
   A flow-based generative model for raw audo synthesis. [[Code](https://github.com/ksw0306/FloWaveNet)]

1. 2018-10-02 - [FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367) by Grathwohl, Chen et al.<br>
   Uses Neural ODEs as a solver to produce continuous-time normalizing flows (CNF).

1. 2018-07-09 - [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039) by Kingma, Dhariwal<br>
   They show that flows using invertible 1x1 convolution achieve high likelihood on standard generative benchmarks and can efficiently synthesize realistic-looking, large images.

1. 2018-07-03 - [Deep Density Destructors](https://proceedings.mlr.press/v80/inouye18a.html) by Inouye, Ravikumar<br>
   Normalizing flows but from an iterative perspective. Features a Tree-based density estimator.

1. 2018-04-03 - [Neural Autoregressive Flows](https://arxiv.org/abs/1804.00779) by Huang, Krueger et al.<br>
   Unifies and generalize autoregressive and normalizing flow approaches, replacing the (conditionally) affine univariate transformations of MAF/IAF with a more general class of invertible univariate transformations expressed as monotonic neural networks. Also demonstrates that the proposed neural autoregressive flows (NAF) are universal approximators for continuous probability distributions. [[Code](https://github.com/CW-Huang/NAF)]

1. 2018-03-15 - [Sylvester Normalizing Flow for Variational Inference](https://arxiv.org/abs/1803.05649) by Berg, Hasenclever et al.<br>
   Introduces Sylvester normalizing flows which remove the single-unit bottleneck from planar flows for increased flexibility in the variational posterior.

1. 2017-11-17 - [Convolutional Normalizing Flows](https://arxiv.org/abs/1711.02255) by Zheng, Yang et al.<br>
   Introduces normalizing flows that take advantage of convolutions (based on convolution over the dimensions of random input vector) to improve the posterior in the variational inference framework. This also reduced the number of parameters due to the convolutions.

1. 2017-05-19 - [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057) by Papamakarios, Pavlakou et al.<br>
   Introduces MAF, a stack of autoregressive models forming a normalizing flow suitable for fast density estimation but slow at sampling. Analogous to Inverse Autoregressive Flow (IAF) except the forward and inverse passes are exchanged. Generalization of RNVP.

   <a href="https://github.com/janosh/tikz/tree/main/assets/maf">
     <picture>
       <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/janosh/tikz/main/assets/maf/maf-white.svg">
       <img alt="Diagram of the slow (sequential) forward pass of a Masked Autoregressive Flow (MAF) layer" src="https://raw.githubusercontent.com/janosh/tikz/main/assets/maf/maf.svg">
     </picture>
   </a>

1. 2017-03-06 - [Multiplicative Normalizing Flows for Variational Bayesian Neural Networks](https://arxiv.org/abs/1703.01961) by Louizos, Welling<br>
   They introduce a new type of variational Bayesian neural network that uses flows to generate auxiliary random variables which boost the flexibility of the variational family by multiplying the means of a fully-factorized Gaussian posterior over network parameters. This turns the usual diagonal covariance Gaussian into something that allows for multimodality and non-linear dependencies between network parameters.

1. 2016-06-15 - [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934) by Kingma, Salimans et al.<br>
   Introduces inverse autoregressive flow (IAF), a new type of flow which scales well to high-dimensional latent spaces. [[Code](https://github.com/openai/iaf)]

1. 2016-05-27 - [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) by Dinh, Sohl-Dickstein et al.<br>
   They introduce the affine coupling layer (RNVP), a major improvement in terms of flexibility over the additive coupling layer (NICE) with unit Jacobian while keeping a single-pass forward and inverse transformation for fast sampling and density estimation, respectively.

   <a href="https://github.com/janosh/tikz/tree/main/assets/rnvp">
     <picture>
       <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/janosh/tikz/main/assets/rnvp/rnvp-white.svg">
       <img alt="Diagram of real-valued non-volume preserving (RNVP) coupling layer" src="https://raw.githubusercontent.com/janosh/tikz/main/assets/rnvp/rnvp.svg">
     </picture>
   </a>

1. 2015-05-21 - [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) by Rezende, Mohamed<br>
   They show how to go beyond mean-field variational inference by using flows to increase the flexibility of the variational family.

1. 2015-02-12 - [Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Germain, Gregor et al.<br>
   Introduces MADE, a feed-forward network that uses carefully constructed binary masks on its weights to control the precise flow of information through the network. The masks ensure that each output unit receives signals only from input units that come before it in some arbitrary order. Yet all outputs can be computed in a single pass.

   A popular and efficient way to make flows autoregressive is to construct them from MADE nets.

   <a href="https://github.com/janosh/tikz/tree/main/assets/made">
     <picture>
       <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/janosh/tikz/main/assets/made/made-white.svg">
       <img alt="Masked Autoencoder for Distribution Estimation" src="https://raw.githubusercontent.com/janosh/tikz/main/assets/made/made.svg">
     </picture>
   </a>

1. 2014-10-30 - [Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) by Dinh, Krueger et al.<br>
   Introduces the additive coupling layer (NICE) and shows how to use it for image generation and inpainting.

1. 2011-04-01 - [Iterative Gaussianization: from ICA to Random Rotations](https://arxiv.org/abs/1602.00229) by Laparra, Camps-Valls et al.<br>
   Normalizing flows in the form of Gaussianization in an iterative format. Also shows connections to information theory.

<br>

## 🛠️ Applications

1. 2020-12-06 - [Normalizing Kalman Filters for Multivariate Time Series Analysis](https://assets.amazon.science/ea/0c/88b7bdd54eae8c08983fa4cc3e06/normalizing-kalman-filters-for-multivariate-time-series-analysis.pdf) by Bézenac, Rangapuram et al.<br>
   Augments state space models with normalizing flows and thereby mitigates imprecisions stemming from idealized assumptions. Aimed at forecasting real-world data and handling varying levels of missing data. (Also available at [Amazon Science](https://amazon.science/publications/normalizing-kalman-filters-for-multivariate-time-series-analysis).)

1. 2020-11-02 - [On the Sentence Embeddings from Pre-trained Language Models](https://aclweb.org/anthology/2020.emnlp-main.733) by Li, Zhou et al.<br>
   Proposes to use flows to transform anisotropic sentence embedding distributions from BERT to a smooth and isotropic Gaussian, learned through unsupervised objective. Demonstrates performance gains over SOTA sentence embeddings on semantic textual similarity tasks. Code available at <https://github.com/bohanli/BERT-flow>.

1. 2020-10-13 - [Targeted free energy estimation via learned mappings](https://aip.scitation.org/doi/10.1063/5.0018903) by Wirnsberger, Ballard et al.<br>
   Normalizing flows used to estimate free energy differences.

1. 2020-07-15 - [Faster Uncertainty Quantification for Inverse Problems with Conditional Normalizing Flows](https://arxiv.org/abs/2007.07985) by Siahkoohi, Rizzuti et al.<br>
   Uses conditional normalizing flows for inverse problems. [[Video](https://youtu.be/nPvZIKaRBkI)]

1. 2020-06-25 - [SRFlow: Learning the Super-Resolution Space with Normalizing Flow](https://arxiv.org/abs/2006.14200) by Lugmayr, Danelljan et al.<br>
   Uses normalizing flows for super-resolution.

1. 2019-03-09 - [NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport](https://arxiv.org/abs/1903.03704) by Hoffman, Sountsov et al.<br>
   Uses normalizing flows in conjunction with Monte Carlo estimation to have more expressive distributions and better posterior estimation.

1. 2018-08-14 - [Analyzing Inverse Problems with Invertible Neural Networks](https://arxiv.org/abs/1808.04730) by Ardizzone, Kruse et al.<br>
   Normalizing flows for inverse problems.

1. 2018-04-09 - [Latent Space Policies for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1804.02808) by Haarnoja, Hartikainen et al.<br>
   Uses normalizing flows, specifically RealNVPs, as policies for reinforcement learning and also applies them for the hierarchical reinforcement learning setting.

<br>

## 📺 Videos

1. 2021-01-16 - [Normalizing Flows - Motivations, The Big Idea & Essential Foundations](https://youtu.be/IuXU2dBOJyw) by [Kapil Sachdeva](https://github.com/ksachdeva)<br>
   A comprehensive tutorial on flows explaining the challenges addressed by this class of algorithm. Provides intuition on how to address those challenges, and explains the underlying mathematics using a simple step by step approach.

1. 2020-12-07 - [Normalizing Flows](https://youtu.be/7TOvhz93G9o) by [Marc Deisenroth](https://mml-book.github.io/slopes-expectations.html)<br>
   Part of a NeurIPS 2020 tutorial series titled "There and Back Again: A Tale of Slopes and Expectations". Link to [full series](https://youtube.com/playlist?list=PL93aLKqThq4h7UpgeNhkOtEeCnX3DMseS).

1. 2020-11-23 - [Introduction to Normalizing Flows](https://youtu.be/u3vVyFVU_lI) by [Marcus Brubaker](https://mbrubake.github.io)<br>
   A great introduction to normalizing flows by one of the creators of [Stan](https://mc-stan.org) presented at ECCV 2020. The tutorial also provides an excellent review of various practical implementations.

1. 2020-02-06 - [Flow Models](https://youtu.be/JBb5sSC0JoY) by [Pieter Abbeel](https://sites.google.com/view/berkeley-cs294-158-sp20/home)<br>
   A really thorough explanation of normalizing flows. Also includes some sample code.

1. 2019-12-06 - [What are normalizing flows?](https://youtu.be/i7LjDvsLWCg) by [Ari Seff](https://scholar.google.com/citations?user=IxBGctYAAAAJ)<br>
   A great 3blue1brown-style video explaining the basics of normalizing flows.

1. 2019-10-09 - [A primer on normalizing flows](https://youtu.be/P4Ta-TZPVi0) by [Laurent Dinh](https://laurent-dinh.github.io)<br>
   The first author on both the NICE and RNVP papers and one of the first in this field gives an introductory talk at "Machine Learning for Physics and the Physics Learning of, 2019".

1. 2019-09-24 - [Graph Normalizing Flows](https://youtu.be/frMPP30QQgY) by Jenny Liu<br>
   Introduces a new graph generating model for use e.g. in drug discovery, where training on molecules that are known to bind/dissolve/etc. may help to generate novel, similarly effective molecules.

1. 2018-10-04 - [Sylvester Normalizing Flow for Variational Inference](https://youtu.be/VeYyUcIDVHI) by Rianne van den Berg<br>
   Introduces Sylvester normalizing flows which remove the single-unit bottleneck from planar flows for increased flexibility in the variational posterior.

<br>

## 📦 Packages

<br>

### <img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Packages

1. 2022-05-21 - [Zuko](https://github.com/francois-rozet/zuko) by [François Rozet](https://francois-rozet.github.io)
&ensp;
<img src="https://img.shields.io/github/stars/francois-rozet/zuko" alt="GitHub repo stars" valign="middle" /><br>
   Zuko is a Python package that implements normalizing flows in PyTorch. It relies heavily on PyTorch's built-in distributions and transformations, which makes the implementation concise, easy to understand and extend. The API is fully documented with references to the original papers.

Zuko is used in [LAMPE](https://github.com/francois-rozet/lampe) to enable Likelihood-free AMortized Posterior Estimation with PyTorch.

1. 2021-01-25 - [Jammy Flows](https://github.com/thoglu/jammy_flows) by [Thorsten Glüsenkamp](https://github.com/thoglu)
&ensp;
<img src="https://img.shields.io/github/stars/thoglu/jammy_flows" alt="GitHub repo stars" valign="middle" /><br>
   A package that models joint (conditional) PDFs on tensor products of manifolds (Euclidean, sphere, interval, simplex) - like inverse autoregressive flows, but connects manifolds, models conditional PDFs, and allows for arbitrary couplings instead of affine ones. Includes a few SOTA flows like Gaussianization flows.

1. 2020-12-07 - [flowtorch](https://github.com/facebookincubator/flowtorch) by [Facebook / Meta](https://opensource.fb.com)
&ensp;
<img src="https://img.shields.io/github/stars/facebookincubator/flowtorch" alt="GitHub repo stars" valign="middle" /><br>
   [FlowTorch Docs](https://flowtorch.ai) is a PyTorch library for learning and sampling from complex probability distributions using a class of methods called Normalizing Flows.

1. 2020-02-09 - [nflows](https://github.com/bayesiains/nflows) by [Bayesiains](https://homepages.inf.ed.ac.uk/imurray2/group)
&ensp;
<img src="https://img.shields.io/github/stars/bayesiains/nflows" alt="GitHub repo stars" valign="middle" /><br>
   A suite of most of the SOTA methods using PyTorch. From an ML group in Edinburgh. They created the current SOTA spline flows. Almost as complete as you'll find from a single repo.

1. 2020-01-28 - [normflows](https://github.com/VincentStimper/normalizing-flows) by [Vincent Stimper](https://github.com/VincentStimper)
&ensp;
<img src="https://img.shields.io/github/stars/VincentStimper/normalizing-flows" alt="GitHub repo stars" valign="middle" /><br>
   The library provides most of the common normalizing flow architectures. It also includes stochastic layers, flows on tori and spheres, and other tools that are particularly useful for applications to the physical sciences.

1. 2018-09-07 - [FrEIA](https://github.com/VLL-HD/FrEIA) by [VLL Heidelberg](https://hci.iwr.uni-heidelberg.de/vislearn)
&ensp;
<img src="https://img.shields.io/github/stars/VLL-HD/FrEIA" alt="GitHub repo stars" valign="middle" /><br>
   The Framework for Easily Invertible Architectures (FrEIA) is based on RNVP flows. Easy to setup, it allows to define complex Invertible Neural Networks (INNs) from simple invertible building blocks.

<br>

### <img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Packages

1. 2018-06-22 - [TensorFlow Probability](https://github.com/tensorflow/probability) by [Google](https://tensorflow.org/probability)
&ensp;
<img src="https://img.shields.io/github/stars/tensorflow/probability" alt="GitHub repo stars" valign="middle" /><br>
   Large first-party library that offers RNVP, MAF among other autoregressive models plus a collection of composable bijectors.

<br>

### <img src="assets/jax.svg" alt="JAX" height="20px"> &nbsp;JAX Packages

1. 2021-06-17 - [pzflow](https://github.com/jfcrenshaw/pzflow) by [John Franklin Crenshaw](https://jfcrenshaw.github.io)
&ensp;
<img src="https://img.shields.io/github/stars/jfcrenshaw/pzflow" alt="GitHub repo stars" valign="middle" /><br>
   A package that focuses on probabilistic modeling of tabular data, with a focus on sampling and posterior calculation.

1. 2021-04-12 - [Distrax](https://github.com/deepmind/distrax) by [DeepMind](https://deepmind.com)
&ensp;
<img src="https://img.shields.io/github/stars/deepmind/distrax" alt="GitHub repo stars" valign="middle" /><br>
   Distrax is a lightweight library of probability distributions and bijectors. It acts as a JAX-native re-implementation of a subset of TensorFlow Probability (TFP), with some new features and emphasis on extensibility.

1. 2020-03-23 - [jax-flows](https://github.com/ChrisWaites/jax-flows) by [Chris Waites](https://chriswaites.com)
&ensp;
<img src="https://img.shields.io/github/stars/ChrisWaites/jax-flows" alt="GitHub repo stars" valign="middle" /><br>
   Another library that has normalizing flows using JAX as the backend. Has some of the SOTA methods.

1. 2020-03-09 - [NuX](https://github.com/Information-Fusion-Lab-Umass/NuX) by Information Fusion Labs (UMass)
&ensp;
<img src="https://img.shields.io/github/stars/Information-Fusion-Lab-Umass/NuX" alt="GitHub repo stars" valign="middle" /><br>
   A library that offers normalizing flows using JAX as the backend. Has some SOTA methods. They also feature a surjective flow via quantization.

<br>

### <img src="assets/julia.svg" alt="Julia" height="20px"> &nbsp;Julia Packages

1. 2020-02-07 - [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl) by [SLIM](https://slim.gatech.edu)
&ensp;
<img src="https://img.shields.io/github/stars/slimgroup/InvertibleNetworks.jl" alt="GitHub repo stars" valign="middle" /><br>
   A Flux compatible library implementing invertible neural networks and normalizing flows using memory-efficient backpropagation. Uses manually implemented gradients to take advantage of the invertibility of building blocks, which allows for scaling to large-scale problem sizes.

<br>

## 🧑‍💻 Repos

<br>

### <img src="assets/pytorch.svg" alt="PyTorch" height="20px"> &nbsp;PyTorch Repos

1. 2021-09-27 - [DeeProb-kit](https://github.com/deeprob-org/deeprob-kit) by [Lorenzo Loconte](https://github.com/loreloc)
&ensp;
<img src="https://img.shields.io/github/stars/deeprob-org/deeprob-kit" alt="GitHub repo stars" valign="middle" /><br>
   A general-purpose Python library providing a collection of deep probabilistic models (DPMs) which are easy to use and extend.
Implements flows such as MAF, RealNVP and NICE.

1. 2021-08-21 - [NICE: Non-linear Independent Components Estimation](https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/NICE_Non_linear_Independent_Components_Estimation) by Maxime Vandegar
&ensp;
<img src="https://img.shields.io/github/stars/MaximeVandegar/Papers-in-100-Lines-of-Code" alt="GitHub repo stars" valign="middle" /><br>
   PyTorch implementation that reproduces results from the paper NICE in about 100 lines of code.

1. 2020-07-19 - [Normalizing Flows - Introduction (Part 1)](https://pyro.ai/examples/normalizing_flows_i) by [pyro.ai](https://pyro.ai)<br>
   A tutorial about how to use the `pyro-ppl` library (based on PyTorch) to use Normalizing flows. They provide some SOTA methods including NSF and MAF. [Parts 2 and 3 coming later](https://github.com/pyro-ppl/pyro/issues/1992).

1. 2020-07-03 - [Density Estimation with Neural ODEs and Density Estimation with FFJORDs](https://git.io/JiWaG) by [torchdyn](https://torchdyn.readthedocs.io)<br>
   Example of how to use FFJORD as a continuous normalizing flow (CNF). Based on the PyTorch suite `torchdyn` which offers continuous neural architectures.

1. 2020-05-26 - [StyleFlow](https://github.com/RameenAbdal/StyleFlow) by [Rameen Abdal](https://twitter.com/AbdalRameen)
&ensp;
<img src="https://img.shields.io/github/stars/RameenAbdal/StyleFlow" alt="GitHub repo stars" valign="middle" /><br>
   Attribute-conditioned Exploration of StyleGAN-generated Images using Conditional Continuous Normalizing Flows. [[Docs](https://rameenabdal.github.io/StyleFlow)]

1. 2020-02-04 - [Graphical Normalizing Flows](https://github.com/AWehenkel/Graphical-Normalizing-Flows) by [Antoine Wehenkel](https://awehenkel.github.io)
&ensp;
<img src="https://img.shields.io/github/stars/AWehenkel/Graphical-Normalizing-Flows" alt="GitHub repo stars" valign="middle" /><br>
   Official implementation of "Graphical Normalizing Flows" and the experiments presented in the paper.

1. 2019-12-09 - [pytorch-normalizing-flows](https://github.com/karpathy/pytorch-normalizing-flows) by Andrej Karpathy
&ensp;
<img src="https://img.shields.io/github/stars/karpathy/pytorch-normalizing-flows" alt="GitHub repo stars" valign="middle" /><br>
   A Jupyter notebook with PyTorch implementations of the most commonly used flows: NICE, RNVP, MAF, Glow, NSF.

1. 2019-09-19 - [Unconstrained Monotonic Neural Networks (UMNN)](https://github.com/AWehenkel/UMNN) by Antoine Wehenkel
&ensp;
<img src="https://img.shields.io/github/stars/AWehenkel/UMNN" alt="GitHub repo stars" valign="middle" /><br>
   Official implementation of "Unconstrained Monotonic Neural Networks" and the experiments presented in the paper.

1. 2019-02-06 - [pytorch_flows](https://github.com/acids-ircam/pytorch_flows) by [acids-ircam](https://github.com/acids-ircam)
&ensp;
<img src="https://img.shields.io/github/stars/acids-ircam/pytorch_flows" alt="GitHub repo stars" valign="middle" /><br>
   A great repo with some basic PyTorch implementations of normalizing flows from scratch.

1. 2018-12-30 - [normalizing_flows](https://github.com/kamenbliznashki/normalizing_flows) by Kamen Bliznashki
&ensp;
<img src="https://img.shields.io/github/stars/kamenbliznashki/normalizing_flows" alt="GitHub repo stars" valign="middle" /><br>
   Pytorch implementations of density estimation algorithms: BNAF, Glow, MAF, RealNVP, planar flows.

1. 2018-09-01 - [pytorch-flows](https://github.com/ikostrikov/pytorch-flows) by Ilya Kostrikov
&ensp;
<img src="https://img.shields.io/github/stars/ikostrikov/pytorch-flows" alt="GitHub repo stars" valign="middle" /><br>
   PyTorch implementations of density estimation algorithms: MAF, RNVP, Glow.

<br>

### <img src="assets/tensorflow.svg" alt="TensorFlow" height="20px"> &nbsp;TensorFlow Repos

1. 2020-11-02 - [Variational Inference using Normalizing Flows (VINF)](https://github.com/pierresegonne/VINF) by Pierre Segonne
&ensp;
<img src="https://img.shields.io/github/stars/pierresegonne/VINF" alt="GitHub repo stars" valign="middle" /><br>
   This repository provides a hands-on TensorFlow implementation of Normalizing Flows as presented in the [paper](https://arxiv.org/pdf/1505.05770.pdf) introducing the concept (D. Rezende & S. Mohamed).

1. 2020-01-29 - [Normalizing Flows](https://github.com/LukasRinder/normalizing-flows) by [Lukas Rinder](https://github.com/LukasRinder)
&ensp;
<img src="https://img.shields.io/github/stars/LukasRinder/normalizing-flows" alt="GitHub repo stars" valign="middle" /><br>
   Implementation of normalizing flows (Planar Flow, Radial Flow, Real NVP, Masked Autoregressive Flow (MAF), Inverse Autoregressive Flow (IAF), Neural Spline Flow) in TensorFlow 2 including a small tutorial.

1. 2019-07-19 - [BERT-flow](https://github.com/bohanli/BERT-flow) by Bohan Li
&ensp;
<img src="https://img.shields.io/github/stars/bohanli/BERT-flow" alt="GitHub repo stars" valign="middle" /><br>
   TensorFlow implementation of "On the Sentence Embeddings from Pre-trained Language Models" (EMNLP 2020).

<br>

### <img src="assets/jax.svg" alt="JAX" height="20px"> &nbsp;JAX Repos

1. 2020-06-12 - [Neural Transport](https://pyro.ai/numpyro/examples/neutra) by [numpyro](https://num.pyro.ai)<br>
   Features an example of how Normalizing flows can be used to get more robust posteriors from Monte Carlo methods. Uses the `numpyro` library which is a PPL with JAX as the backend. The NF implementations include the basic ones like IAF and BNAF.

<br>

### <img src="assets/julia.svg" alt="Julia" height="20px"> &nbsp;Julia Repos

1. 2021-11-10 - [ICNF.jl](https://github.com/impICNF/ICNF.jl) by [Hossein Pourbozorg](https://github.com/prbzrg)
&ensp;
<img src="https://img.shields.io/github/stars/impICNF/ICNF.jl" alt="GitHub repo stars" valign="middle" /><br>
   Implementations of Infinitesimal Continuous Normalizing Flows in Julia. [[Docs](https://impicnf.github.io/ICNF.jl/dev)]

<br>

### <img src="assets/other.svg" alt="Other" height="20px"> &nbsp;Other Repos

1. 2018-06-11 - [Destructive Deep Learning (ddl)](https://github.com/davidinouye/destructive-deep-learning) by [David Inouye](https://davidinouye.com)
&ensp;
<img src="https://img.shields.io/github/stars/davidinouye/destructive-deep-learning" alt="GitHub repo stars" valign="middle" /><br>
   Code base for the paper [Deep Density Destructors](https://proceedings.mlr.press/v80/inouye18a.html) by Inouye & Ravikumar (2018). An entire suite of iterative methods including tree-based as well as Gaussianization methods which are similar to normalizing flows except they converge iteratively instead of fully parametrized. That is, they still use bijective transforms, compute the Jacobian, check the likelihood and you can still sample and get probability density estimates. The only difference is you repeat the following two steps until convergence:

1. compute one layer or block layer (e.g. Marginal Gaussianization + PCA rotation)
1. check for convergence (e.g log-likelihood using the change-of-variables formula)

Table 1 in the paper has a good comparison with traditional NFs.

1. 2017-07-11 - [Normalizing Flows Overview](https://docs.pymc.io/en/v3/pymc-examples/examples/variational_inference/normalizing_flows_overview.html) by PyMC3<br>
   A very helpful notebook showcasing how to work with flows in practice and comparing it to PyMC3's NUTS-based HMC kernel. Based on [Theano](https://github.com/Theano/Theano).

1. 2017-03-21 - [NormFlows](https://github.com/andymiller/NormFlows) by Andy Miller
&ensp;
<img src="https://img.shields.io/github/stars/andymiller/NormFlows" alt="GitHub repo stars" valign="middle" /><br>
   Simple didactic example using [`autograd`](https://github.com/HIPS/autograd), so pretty low-level.

<br>

## 🌐 Blog Posts

1. 2020-08-19 - [Chapter on flows from the book 'Deep Learning for Molecules and Materials'](https://dmol.pub/dl/flows) by Andrew White<br>
   A nice introduction starting with the change of variables formula (aka flow equation), going on to cover some common bijectors and finishing with a code example showing how to fit the double-moon distribution with TensorFlow Probability.

1. 2018-10-21 - [Change of Variables for Normalizing Flows](https://nealjean.com/ml/change-of-variables) by Neal Jean<br>
   Short and simple explanation of change of variables theorem i.t.o. probability mass conservation.

1. 2018-10-13 - [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models) by Lilian Weng<br>
   Covers change of variables, NICE, RNVP, MADE, Glow, MAF, IAF, WaveNet, PixelRNN.

1. 2018-04-03 - [Normalizing Flows](https://akosiorek.github.io/ml/2018/04/03/norm_flows) by Adam Kosiorek<br>
   Introduction to flows covering change of variables, planar flow, radial flow, RNVP and autoregressive flows like MAF, IAF and Parallel WaveNet.

1. 2018-01-17 - [Normalizing Flows Tutorial](https://blog.evjang.com/2018/01/nf1.html) by Eric Jang<br>
   [Part 1](https://blog.evjang.com/2018/01/nf1.html): Distributions and Determinants. [Part 2](https://blog.evjang.com/2018/01/nf2.html): Modern Normalizing Flows. Lots of great graphics.

<br>

## 🚧 Contributing

See something that's missing from this list? PRs welcome! A good place to find new items for the Repos section is the [Normalizing Flows topic on GitHub](https://github.com/topics/normalizing-flows).

Note: Don't edit the readme directly (it's auto-generated). Add your submission
to the appropriate [`data/*.yml`](https://github.com/janosh/awesome-normalizing-flows/edit/main/data) file.

Papers should be peer-reviewed and published in a journal. If you're unsure if a paper or resource belongs in this list, feel free to [open an issue](https://github.com/janosh/awesome-normalizing-flows/issues/new) or [start a discussion](https://github.com/janosh/awesome-normalizing-flows/discussions). This repo is meant to be a community effort. Don't hesitate to voice an opinion.
