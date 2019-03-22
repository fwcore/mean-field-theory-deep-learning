# Mean-field theory and dynamical isometry of neural networks

## Speaker: Feng Wang

Abstract: Initialization, activation function, and batch normalization are known to have a strong impact on the training and the generalization of deep neural networks. Heuristic arguments have provided insightful pictures on their underlying mechanisms, however, a fundamental understanding remains elusive, leaving training and design neural networks strongly relying on experience. Recently, researchers from Google Brain start to build a theoretical framework to understand the impact on initialization, activation function, and batch normalization, as well as the architectures, such as CNN, ResNet, Gated RNN. The proposed mean-field theory focuses on the information flow through each layer by investigating the preservation of covariance matrices of pre-activations/gradients of each layer in the limit that (1) each layer is wide and (2) network is deep.

In this talk, I will first briefly outline the theoretical framework and show the theoretical predictions, especially the predicted initialization strategies and the counterintuitive results on batch normalization. Then I will lead to dive into the theory and show its assumptions and possible pitfalls. We will emphasize the ideas behind the theory rather than the mathematical derivation.
