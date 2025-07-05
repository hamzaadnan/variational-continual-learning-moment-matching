# Overview

This repository is a PyTorch implementation of Variational Continual Learning [1]](#1). As an additional modification in the latter experiments, the KL-divergence objective between successive weights from the original paper has been replaced by one based on moment matching. This is theoretically motivated from Wasserstein-distance minimisation. The proof is as follows:

In my setup, the marginals of the joint distribution γ over each set of weights (θ ~ q, θ′ ~ q_{t−1}) is Gaussian:

∫_θ γ(θ, θ′) dθ     = q_{t−1}(θ′) = 𝒩(θ′; μ_{t−1}, σ_{t−1}²)  
∫_{θ′} γ(θ, θ′) dθ′ = q(θ)       = 𝒩(θ; μ, σ²)

We compute the following expectation under γ:

E_{θ, θ′ ~ γ} [ Σ_{n=1}^{N_t} −log p(y_t^{(n)} | θ, x_t^{(n)}) + λ(θ − θ′)² ]

Expanding this:

= ∬ Σ_{n=1}^{N_t} −log p(y_t^{(n)} | θ, x_t^{(n)}) γ(θ, θ′) dθ dθ′  
  + λ E_{θ, θ′ ~ γ} [θ² + θ′² − 2θθ′]

= ∫ Σ_{n=1}^{N_t} −log p(y_t^{(n)} | θ, x_t^{(n)}) q(θ) dθ  
  + λ ( E_γ[θ²] + E_γ[θ′²] − 2E_γ[θθ′] )

= E_q[ Σ_{n=1}^{N_t} −log p(y_t^{(n)} | θ, x_t^{(n)}) ]  
  + λ [ μ² + σ² + μ_{t−1}² + σ_{t−1}² − 2(ρ_{θθ′} σ σ_{t−1} + μ μ_{t−1}) ]

The last term uses the identity:

ρ_{XY} = Cov(X, Y) / (σ_X σ_Y) = E[(X − E[X])(Y − E[Y])] / (σ_X σ_Y)

Finally, since both distributions are from the same family, for the optimisation objective of aligning them, we take the case when the correlation is 1, giving the final objective after completing the square:

E_q[ Σ_{n=1}^{N_t} −log p(y_t^{(n)} | θ, x_t^{(n)}) ]  
+ λ [ (μ − μ_{t−1})² + (σ − σ_{t−1})² ]



## References
<a id="1">[1]</a> Nguyen, Cuong V., et al. "Variational continual learning." arXiv preprint arXiv:1710.10628 (2017).
