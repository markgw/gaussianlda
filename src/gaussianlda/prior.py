import numpy as np
from scipy.linalg import cholesky


class Wishart(object):
    """
    Simply data structure to hold the parameters to the normal-inverse-Wishart prior.

    The distribution is described here:
       https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution

    The parameters are:

      - kappa: Concentration parameter for the normal prior on the mean of the gaussian.
               A higher value gives more weight to the prior (see mu) and less to the
               observations. Called lambda on the wikipedia description.
      - mu:    Prior belief about the mean of the gaussian.
               Called mu_0 in the wikipedia description.
      - nu:    Prior belief about the degrees of freedom of the inverse Wishart prior on the
               covariance of the gaussian. The posterior degrees of freedom increases with the
               number of samples.
               Also called nu in the wikipedia description.
      - sigma: Prior belief about the scale matrix of the inverse Wishart distribution on
               the covariance matrix.
               Called Psi_0 in the wikipedia description.

    A default for sigma is often recommended to be eye(D), i.e. a DxD identity matrix (where
    D is the number of dimensions). If you specify scale_sigma, sigma is set to the identity
    matrix mutiplied by this scalar. Set scale_sigma=1. to get this recommended default.

    The default sigma, however, if you don't give scale_sigma, is scaled by 3*D, which is
    what the Gaussian LDA authors use.

    """
    def __init__(self, word_vecs, kappa=0.1, scale_sigma=None):
        vec_size = word_vecs.shape[1]

        # Parameters for the prior on the mean
        self.kappa = kappa
        self.mu = np.mean(word_vecs, axis=0)

        # Parameters for the prior on the covariance
        # Initialize nu to num vector dimensions
        # It must be at least D-1
        self.nu = vec_size
        # Set sigma_0
        if scale_sigma is None:
            # Use the Gaussian LDA default behaviour
            scale_sigma = vec_size * 3.
        self.sigma = np.eye(vec_size, dtype=np.float64) * scale_sigma
        self.chol_sigma = cholesky(self.sigma)
