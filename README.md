# Overview

This repository is a PyTorch implementation of Variational Continual Learning [1]](#1). As an additional modification in the latter experiments, the KL-divergence objective between successive weights from the original paper has been replaced by one based on moment matching. This is theoretically motivated from Wasserstein-distance minimisation. The proof is as follows:

In my setup, the marginals of the joint distribution $\gamma$ over each set of weights ($\theta \sim q, \theta' \sim q_{t-1}$) is Gaussian:
\begin{align*}
    \int_\theta \gamma(\theta, \theta') \, d\theta &= q_{t-1}(\theta') = \mathcal{N}(\theta'; \mu_{t-1}, \sigma_{t-1}^2) \\ 
    \int_\theta' \gamma(\theta, \theta') \, d\theta' &= q(\theta) = \mathcal{N}(\theta; \mu, \sigma^2)
\end{align*}

\begin{align*}
    &\mathbb{E}_{\theta, \theta' \sim \gamma} \left[\sum_{n=1}^{N_t} -\log p(y_t^{(n)}|\theta, \mathbf{x}_t^{(n)})  + \lambda(\theta - \theta')^2 \right] \\
    &= \iint \sum_{n=1}^{N_t} -\log p(y_t^{(n)}|\theta, \mathbf{x}_t^{(n)}) \gamma(\theta, \theta') \, d\theta' d\theta +  \lambda \mathbb{E}_{\theta, \theta' \sim \gamma} \left[\theta^2 + \theta'^2 - 2\theta \theta' \right] \\ 
    &= \int \sum_{n=1}^{N_t} -\log p(y_t^{(n)}|\theta, \mathbf{x}_t^{(n)})q(\theta) \, d\theta + \lambda \left(\mathbb{E}_{\theta, \theta' \sim \gamma} [\theta^2] + \mathbb{E}_{\theta, \theta' \sim \gamma} [\theta'^2] - 2\mathbb{E}_{\theta, \theta' \sim \gamma}[\theta\theta'] \right) \\
    &= \mathbb{E}_q\left[ \sum_{n=1}^{N_t} -\log p(y_t^{(n)}|\theta, \mathbf{x}_t^{(n)}) \right] + \lambda \left( (\mu^2 + \sigma^2) + (\mu_{t-1}^2 + \sigma_{t-1}^2) - 2(\rho_{\theta\theta'}\sigma\sigma_{t-1} + \mu \mu_{t-1})   \right) 
\end{align*}

Where the last term follows from $\rho_{XY} = \frac{\mathbb{C}\text{ov}(X, Y)}{\sigma_X\sigma_Y} = \frac{\mathbb{E}[(X - \mathbb{E[X]})(Y - \mathbb{E}[Y])]}{\sigma_X\sigma_Y}$ for two random variables $X, Y$.

Finally, since both distributions are from the same family, for the optimisation objective of aligning them, I take the case when the correlation is 1, giving the following optimisation objective after completing the square:



## References
<a id="1">[1]</a> Nguyen, Cuong V., et al. "Variational continual learning." arXiv preprint arXiv:1710.10628 (2017).
