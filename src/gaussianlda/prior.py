import numpy as np
from scipy.linalg import cholesky


class Wishart(object):
    """
    Simply data structure to hold the parameters to the inverse Wishart prior.

    """
    def __init__(self, word_vecs, kappa=0.1):
        vec_size = word_vecs.shape[1]

        self.kappa = kappa
        # Initialize nu to num vector dimensions
        self.nu = vec_size
        # Set sigma_0
        self.sigma = np.eye(vec_size, dtype=np.float64) * vec_size * 3.
        self.chol_sigma = cholesky(self.sigma)

        self.mu = np.mean(word_vecs, axis=0)
