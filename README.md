## Keywords
* mean field theory
* central limit theorem
* dynamical isometry
* Jacobian matrix
* dynamical system
* fixed points
* eigenvalues
* singular values

## Key people
* [Samuel S. Schoenholz](https://samschoenholz.wordpress.com/), Google Brain
	- focused on using notions from statistical physics to better understand neural networks.
	- Ph. D. in Physics working with Andrea Liu at the University of Pennsylvania, focused on understanding the behavior of disordered solids and glassy liquids from their structure. Central to the approach has been the use of machine learning to identify local structural motifs that are particularly susceptible to rearrangement.

* Jeffrey Pennington, Google Brain
	- a postdoctoral fellow at Stanford University, as a member of the Stanford Artificial Intelligence Laboratory in the Natural Language Processing (NLP) group. He received his Ph.D. in theoretical particle physics from Stanford University while working at the SLAC National Accelerator Laboratory.
	- Jeffrey’s research interests are multidisciplinary, ranging from the development of calculational techniques in perturbative quantum field theory to the vector representation of words and phrases in NLP to the study of trainability and expressivity in deep learning. Recently, his work has focused on building a set of theoretical tools with which to study deep neural networks. Leveraging techniques from random matrix theory and free probability, Jeffrey has investigated the geometry of neural network loss surfaces and the learning dynamics of very deep neural networks. He has also developed a new framework to begin harnessing the power of random matrix theory in applications with nonlinear dependencies, like deep learning.
	- [Theories of Deep Learning (STATS 385): Harnessing the Power of Random Matrix Theory to Study and Improve Deep Learning](https://stats385.github.io/pennington_lecture), Stanford University, Fall 2017

## Papers

* Mean Field Analysis of Deep Neural Networks | [arXiv:1903.04440](https://arxiv.org/abs/1903.04440)
	- asymptotic behavior of MLP under large network size and large number of training iterations 
	- characterization of the evolution of parameters in terms of their initialization
	- the limit is a system of integro-differential equations

* Mean-field Analysis of Batch Normalization | [arXiv:1903.02606](https://arxiv.org/abs/1903.02606)
	- analytically quantify the impact of BatchNorm on the geometry of the loss landscape for multi-layer networks consisting of fully-connected and convolutional layers.
	- it has a flattening effect on the loss landscape, as quantified by the maximum eigenvalue of the Fisher Information Matrix, enabling using larger learning rate
	- quantitative characterization of the maximal allowable learning rate to ensure convergence
	- suggest that networks with smaller values of the BatchNorm parameter achieve lower loss after the same number of epochs of training

* A Mean Field Theory of Batch Normalization | [arXiv:1902.08129](https://arxiv.org/abs/1902.08129)
	- provide a precise characterization of signal propagation and gradient backpropagation in wide batch-normalized fully-connected feedforward networks at initialization.
	- BN causes that gradient signals grow exponentially in depth and these exploding gradients cannot be eliminated by tuning the initial weight variances or by adjusting the nonlinear activation function.
	- gradient explosion can be reduced by tuning the network close to the linear regime.

* Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent | [arXiv:1902.06720](https://arxiv.org/abs/1902.06720)
	- for wide neural networks the learning dynamics simplify considerably and that, in the infinite width limit, they are governed by a linear model obtained from the first-order Taylor expansion of the network around its initial parameters
	- find excellent empirical agreement between the predictions of the original network and those of the linearized version even for finite practically-sized networks. This agreement is robust across different architectures, optimization methods, and loss functions.

* Dynamical Isometry and a Mean Field Theory of LSTMs and GRUs | [arXiv:1901.08987](https://arxiv.org/abs/1901.08987)
	- develop a mean field theory of signal propagation in LSTMs and GRUs that enables us to calculate the time scales for signal propagation as well as the spectral properties of the state-to-state Jacobians.
	- derive a novel initialization scheme that eliminates or reduces training instabilities, enabling successful training while a standard initialization either fails completely or is orders of magnitude slower.
	- observe a beneficial effect on generalization performance using this new initialization.

* Information Geometry of Orthogonal Initializations and Training  | [arXiv:1810.03785](https://arxiv.org/abs/1810.03785)
	- show a novel connection between the maximum curvature of the optimization landscape (gradient smoothness) as measured by the Fisher information matrix and the maximum singular value of the input-output Jacobian. 
	- partially explains why neural networks that are more isometric can train much faster.
	- experimentally investigate the benefits of maintaining orthogonality throughout training
	- critical orthogonal initializations do not trivially give rise to a mean field limit of pre-activations for each layer.

* Dynamical Isometry is Achieved in Residual Networks in a Universal Way for any Activation Function | [arXiv:1809.08848](https://arxiv.org/abs/1809.08848)
	- demonstrate that in residual neural networks (ResNets) dynamical isometry is achievable irrespectively of the activation function used
	- initialization acts as a confounding factor between the choice of activation function and the rate of learning, which can be resolved in ResNet by ensuring the same level of dynamical isometry at initialization.

* Mean Field Analysis of Neural Networks: A Central Limit Theorem | [arXiv:1808.09372](https://arxiv.org/abs/1808.09372)
	- asymptotic regime of simultaneously (A) large network sizes and (B) large numbers of stochastic gradient descent training iterations.
	- rigorously prove that the neural network satisfies a central limit theorem
	- describes the neural network's fluctuations around its mean-field limit
	- The fluctuations have a Gaussian distribution and satisfy a stochastic partial differential equation.

* Dynamical Isometry and a Mean Field Theory of RNNs: Gating Enables Signal Propagation in Recurrent Neural Networks | [arXiv:1806.05394](https://arxiv.org/abs/1806.05394)
	- develop a theory for signal propagation in recurrent networks after random initialization using a combination of mean field theory and random matrix theory.
	- Our theory allows us to define a maximum timescale over which RNNs can remember an input, predicting the trainability.
	- gated recurrent networks feature a much broader, more robust, trainable region than vanilla RNNs
	- develop a closed-form critical initialization scheme that achieves dynamical isometry in both vanilla RNNs and minimalRNNs.

* Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks | [arXiv:1806.05393](https://arxiv.org/abs/1806.05393)
	- it is possible to train vanilla CNNs with ten thousand layers or more simply by using an appropriate initialization scheme
	- develop a mean field theory for signal propagation and characterize the conditions for dynamical isometry for CNN
	- These conditions require that the convolution operator be an orthogonal transformation in the sense that it is norm-preserving.
	- present an algorithm for generating such random initial orthogonal convolution kernels.

* Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach | [arXiv:1806.01316 ](https://arxiv.org/abs/1806.01316)
	- reveals novel statistics of Fisher information matrix (FIM) that are universal among a wide class of DNNs
	- investigate the asymptotic statistics of the FIM's eigenvalues and reveal that most of them are close to zero while the maximum takes a huge value
	- implies that the eigenvalue distribution has a long tail
	- Because the landscape of the parameter space is defined by the FIM, it is locally flat in most dimensions, but strongly distorted in others
	- small eigenvalues that induce flatness can be connected to a norm-based capacity measure of generalization ability
	- maximum eigenvalue that induces the distortion enables us to quantitatively estimate an appropriately sized learning rate for gradient methods to converge

* The Emergence of Spectral Universality in Deep Networks | [arXiv:1802.09979](https://arxiv.org/abs/1802.09979)
	- build a full theoretical understanding of the spectra of Jacobians at initialization
	- leverage powerful tools from free probability theory to provide a detailed analytic understanding of how a deep network's Jacobian spectrum depends on various hyperparameters including the nonlinearity, the weight and bias distributions, and the depth
	- For a variety of nonlinearities, our work reveals the emergence of new universal limiting spectral distributions that remain concentrated around one even as the depth goes to infinity.

* Mean Field Residual Networks: On the Edge of Chaos | [arXiv:1712.08969](https://arxiv.org/abs/1712.08969)
	- study randomly initialized residual networks using mean field theory and the theory of difference equations
	- Classical feedforward neural networks, such as those with tanh activations, exhibit exponential behavior on the average when propagating inputs forward or gradients backward. The exponential forward dynamics causes rapid collapsing of the input space geometry, while the exponential backward dynamics causes drastic vanishing or exploding gradients.
	- In contrast, by adding skip connections, the network will, depending on the nonlinearity, adopt subexponential forward and backward dynamics, and in many cases in fact polynomial.
	- In terms of the "edge of chaos" hypothesis, these subexponential and polynomial laws allow residual networks to "hover over the boundary between stability and chaos," thus preserving the geometry of the input space and the gradient information flow.
	- common initializations such as the Xavier or the He schemes are not optimal for residual networks, because the optimal initialization variances depend on the depth.

* Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice | [arXiv:1711.04735](https://arxiv.org/abs/1711.04735)
	- extend the results obtained previously from linear DNN to the nonlinear setting
	- explore the dependence of the singular value distribution on the depth of the network, the weight initialization, and the choice of nonlinearity.
	- ReLU networks are incapable of dynamical isometry
	- sigmoidal networks can achieve isometry, but only with orthogonal weight initialization.
	- demonstrate empirically that deep nonlinear networks achieving dynamical isometry learn orders of magnitude faster than networks that do not.
	- show that properly-initialized deep sigmoidal networks consistently outperform deep ReLU networks

* Deep Information Propagation | [arXiv:1611.01232](https://arxiv.org/abs/1611.01232)
	- study the behavior of untrained neural networks whose weights and biases are randomly distributed using mean field theory
	- show the existence of depth scales that naturally limit the maximum depth of signal propagation through these random networks
	- arbitrarily deep networks may be trained only sufficiently close to criticality.
	- the presence of dropout destroys the order-to-chaos critical point and therefore strongly limits the maximum trainable depth for random networks.
	- develop a mean field theory for backpropagation and we show that the ordered and chaotic phases correspond to regions of vanishing and exploding gradient respectively.
