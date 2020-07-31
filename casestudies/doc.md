
# 1. Introduction

Data scientists often choose the uniform distribution
or the normal distribution as the latent variable distribution
when they build representetive models of datasets.
For example,
the studies of the GANs [1] and the VAEs[2] used the normal distribution.

As the approximate function implimented by neural networks 
is usually continuous,
the topological structure of the latent variable distribution
is preserved after the transformation 
from the latent variable to the feature ones.
Given that the observed variables are distributed on a torus 
and that networks, for example the GANs, are trained with the latent variables
sampled from the normal distribution,
the structure of the distribution projected through the trained network
does not meet with the torus,
even though residual error is enough small.
Imagine another example
where the feature variables follow a mixtured distribution,
of whihch clusters separate each other,
trained variational auto encoder can encode the feature
on the latent variable with enough precision,
however,
the decoded distribution consists of 
a connected set 
since the latent variable is topologically equal with the ball.
This means that the topology of the given dataset is destroyed
through the projction of the trained networks.

In this short text,
we study the topological mismatch with the SAE[3],
which is enhanced based on the WAE[4] 
owing to the sinkhorn algorithm.


# 2. Specifications


# 3. Case studies

# 3-1. Case study #1:

This case study builds representitive models
of a two-dimensional torus
by using the autoencoder with the latent variables
sampled from the two dimentional uniformal distribution.
We show a consequence caused by the topological mismatch 
between the observable and the latent variables.

Models are trained by using the hyperparameters shown in the table 3.1.1.
The figure 3.1.1 shows the learning curves of the following 
training performance:
- Representive error, `mean((Y-Yhat)^2)`, where `Y and Yhat` are the original observed varibales and the represented ones, respectively.
- Discrepancy between the referenced distribution of the latent variables and the ones projected by the encoder of trained model. Note that the discrepancy is measured by the two norm wasserstein distance.

The learning curves tell us that the training has converged at the end of the training.

The figure 3.1.2 shows an example of the projection of a trained model chosen randomly among the trained models.
The figure 3.1.3 (a) shows the images projected through the encoder of the model.
In this case, the image of the observed variables is approximated by an analytical function.
The figure 3.1.3 (b) is the case of the decoder.
We found that 
- the projected samples match well with the original samples and the distribution of the latent variables looks like the uniform distribution,
- ,the hole mapped from the observable variable almost disappears from the image of the encoder
- and the image of the decoder is topologically identified with the disk, even though the region around the hole is stretched.

The last findings says that the decoder as a map from the latent variable to the observable variable
cannot preserve the topological structure.
