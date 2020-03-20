import json
import os
import pickle
import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from gaussianlda.prior import Wishart


class GaussianLDA:
    """
    Trained model.

    First train using the GaussianLDATrainer or GaussianLDAAliasTrainer.
    Then load using this class to get a GaussianLDA with the saved parameters for performing
    inference on new data without updating the parameters.

    """
    def __init__(self, vocab_embeddings, vocab, num_tables, alpha, kappa, table_counts, table_means, log_determinants,
                 sum_squared_table_customers, table_cholesky_ltriangular_mat):
        # Vocab is used for outputting topics
        self.vocab = vocab

        # Dirichlet hyperparam
        self.alpha = alpha

        # dataVectors
        self.vocab_embeddings = vocab_embeddings
        self.embedding_size = vocab_embeddings.shape[1]
        # numIterations
        # K, num tables
        self.num_tables = num_tables
        # Number of customers observed at each table
        self.table_counts = table_counts
        # Mean vector associated with each table
        # This is the bayesian mean (i.e has the prior part too)
        self.table_means = table_means
        # log-determinant of covariance matrix for each table.
        # Since 0.5 * logDet is required in (see logMultivariateTDensity), that value is kept.
        self.log_determinants = log_determinants
        # Stores the squared sum of the vectors of customers at a given table
        self.sum_squared_table_customers = sum_squared_table_customers
        # Cholesky Lower Triangular Decomposition of covariance matrix associated with each table.
        self.table_cholesky_ltriangular_mat = table_cholesky_ltriangular_mat

        # Normal inverse wishart prior
        self.prior = Wishart(self.vocab_embeddings, kappa=kappa)

        # Cache k_0\mu_0\mu_0^T, only compute it once
        # Used in calculate_table_params()
        self.k0mu0mu0T = self.prior.kappa * np.outer(self.prior.mu, self.prior.mu)

        # Since we ignore the document's contributions to the global parameters when sampling,
        # we can precompute a whole load of parts of the likelihood calculation.
        # Table counts are not updated for the document in question, as it's assumed to make
        # a tiny contribution compared to the whole training corpus.
        k_n = self.prior.kappa + self.table_counts
        nu_n = self.prior.nu + self.table_counts
        self.scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.)))
        self.nu = self.prior.nu + self.table_counts - self.embedding_size + 1.

    @staticmethod
    def load(path):
        # Load JSON hyperparams
        with open(os.path.join(path, "params.json"), "r") as f:
            hyperparams = json.load(f)

        # Load numpy arrays for model parameters
        arrs = {}
        for name in [
            "table_counts", "table_means", "log_determinants", "sum_squared_table_customers",
            "table_cholesky_ltriangular_mat", "vocab_embeddings",
        ]:
            with open(os.path.join(path, "{}.pkl".format(name)), "rb") as f:
                arrs[name] = pickle.load(f)

        # Initialize a model
        model = GaussianLDA(
            arrs["vocab_embeddings"], hyperparams["vocab"], hyperparams["num_tables"], hyperparams["alpha"],
            hyperparams["kappa"],
            arrs["table_counts"], arrs["table_means"], arrs["log_determinants"], arrs["sum_squared_table_customers"],
            arrs["table_cholesky_ltriangular_mat"],
        )
        return model

    def sample(self, doc, num_iterations):
        """
        Run Gibbs sampler on a single document without updating global parameters.

        for num_iters:
            for each customer
                remove him from his old_table and update the table params.
                if old_table is empty:
                    remove table
                Calculate prior and likelihood for this customer sitting at each table
                sample for a table index
        """
        table_assignments = list(np.random.randint(self.num_tables, size=len(doc)))
        doc_table_counts = np.bincount(table_assignments, minlength=self.num_tables)

        for iteration in range(num_iterations):
            for w, cust_id in enumerate(doc):
                x = self.vocab_embeddings[cust_id]

                # Remove custId from his old_table
                old_table_id = table_assignments[w]
                table_assignments[w] = -1  # Doesn't really make any difference, as only counts are used
                doc_table_counts[old_table_id] -= 1

                # Now calculate the prior and likelihood for the customer to sit in each table and sample
                # Go over each table
                counts = doc_table_counts[:] + self.alpha
                # Now calculate the likelihood for each table
                log_lls = self.log_multivariate_tdensity_tables(x)
                # Add log prior in the posterior vector
                log_posteriors = np.log(counts) + log_lls
                # To prevent overflow, subtract by log(p_max).
                # This is because when we will be normalizing after exponentiating,
                # each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
                # the log p_max cancels put and prevents overflow in the exponentiating phase.
                posterior = np.exp(log_posteriors - log_posteriors.max())
                posterior /= posterior.sum()
                # Now sample an index from this posterior vector.
                new_table_id = np.random.choice(self.num_tables, p=posterior)

                # Now have a new assignment: add its counts
                doc_table_counts[new_table_id] += 1
                table_assignments[w] = new_table_id
        return table_assignments

    def log_multivariate_tdensity_tables(self, x):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.
        This version computes the likelihood for all tables in parallel.

        """
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver first calculate (x-mu)
        x_minus_mu = x[None, :] - self.table_means
        # Now scale the lower tringular matrix
        ltriangular_chol = self.scaleTdistrn[:, None, None] * self.table_cholesky_ltriangular_mat
        # We can't do solve_triangular for all matrices at once in scipy
        val = np.zeros(self.num_tables, dtype=np.float64)
        for table in range(self.num_tables):
            table_solved = solve_triangular(ltriangular_chol[table], x_minus_mu[table])
            # Now take xTx (dot product)
            val[table] = (table_solved ** 2.).sum()

        logprob = gammaln((self.nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(self.nu / 2.) +
                          self.embedding_size / 2. * (np.log(self.nu) + np.log(math.pi)) +
                          self.log_determinants +
                          (self.nu + self.embedding_size) / 2. * np.log(1. + val / self.nu)
                  )
        return logprob
