# Overview

This repository is a PyTorch implementation of Variational Continual Learning [[1]](#1). As an additional modification in the latter experiments, the KL-divergence objective between successive weights from the original paper has been replaced by one based on moment matching. This is theoretically motivated from Wasserstein-distance minimisation. The proof can be found at the start of `main.ipynb`.


## References
<a id="1">[1]</a> Nguyen, Cuong V., et al. "Variational continual learning." arXiv preprint arXiv:1710.10628 (2017).
