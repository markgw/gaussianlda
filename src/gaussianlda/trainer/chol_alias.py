"""
Implementation of the collapsed gibbs sampler for LDA where the distribution
of each table is a multivariate gaussian with unknown mean and covariances.

Closely based on the authors' implementation in Java:
  https://github.com/rajarshd/Gaussian_LDA

This implementation uses Numpy/Scipy.

This implementation always uses Cholesky decomposition for faster updates to the
covariance matrix. This makes the code simpler and easier to debug than the
version in chol.py, where Cholesky decomposition can be disabled.

Additionally, this implementation includes Vose aliasing to speed up the computation
of the posterior distribution for sparse doc-topic distributions. It is therefore
a full implementation of the training routine given in the paper, with both
speed-up tricks.

"""
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
from multiprocessing import Process
import random

import numpy as np
import warnings

from gaussianlda.mp_utils import GaussianLock, SharedArray, TwoSidedLock
from numpy.core.umath import isinf
from numpy.linalg import slogdet
from scipy.linalg import get_lapack_funcs, solve_triangular
from scipy.special import gammaln

from gaussianlda.prior import Wishart
from gaussianlda.utils import get_logger, get_progress_bar, chol_rank1_downdate, chol_rank1_update, BatchedRands, \
    BatchedRandInts
from gaussianlda.perplexity import calculate_avg_ll


class GaussianLDAAliasTrainer:
    def __init__(self, corpus, vocab_embeddings, vocab, num_tables, alpha=None, kappa=0.1, log=None, save_path=None,
                 show_topics=None, mh_steps=2, num_words_for_formatting=None,
                 replicate_das=False, show_progress=True,
                 initializer=None):
        """

        :param corpus:
        :param vocab_embeddings:
        :param vocab:
        :param num_tables:
        :param alpha: Dirichlet concentration. Defaults to 1/num_tables
        :param kappa:
        :param log:
        :param save_path:
        :param show_topics:
        :param mh_steps:
        :param num_words_for_formatting: By default, each topic is formatted by computing the probability of
            every word in the vocabulary under that topic. This can take a long time for a large vocabulary.
            If given, this limits the number considered to the first
            N in the vocabulary (which makes sense if the vocabulary is ordered with most common words first).
        :param replicate_das: Use normalization of distributions and acceptance probability calculation as
            in the original GLDA paper.
            Use the normalization of probability distributions used by Das, Zaheer and Dyer's
            original implementation when computing the sampling probability to choose whether to use the document
            posterior or language model part of the topic posterior. If False, do not normalize in this way, but use
            an alternative, which looks to me like it's more correct mathematically.
            Use the acceptance probability calculation exactly as in Das et al.'s Java
            implementation. Default behaviour is to use a corrected version of the calculation based on my
            reading of the background literature. For exact comparison to the Java implementation,
            the original formula should be used.
        :param initializer: By default, topics are randomly initialized. Use this to apply a different
            initialization scheme. If given, should be a function that takes two arguments - doc num and
            doc ids - and returns a list of topic IDs to initialize that document to.
        """
        if log is None:
            log = get_logger("GLDA")
        self.log = log
        self.show_progress = show_progress
        # Vocab is used for outputting topics
        self.vocab = vocab
        self.show_topics = show_topics
        self.save_path = save_path

        # MH sampling steps
        self.mh_steps = mh_steps

        # Dirichlet hyperparam
        if alpha is None:
            alpha = 1. / num_tables
        self.alpha = alpha

        self.replicate_das = replicate_das

        # dataVectors
        self.vocab_embeddings = vocab_embeddings
        self.embedding_size = vocab_embeddings.shape[1]
        self.num_terms = vocab_embeddings.shape[0]
        # List of list of ints
        self.corpus = corpus
        # numIterations
        # K, num tables
        self.num_tables = num_tables
        # N, num docs
        self.num_documents = len(corpus)
        # In the current iteration, map of table_id's to number of customers. Table id starts from 0
        # Use shared memory
        self.table_counts = SharedArray.create(self.num_tables, "int")
        # K x N array.tableCounts[i][j] represents how many words of document j are present in topic i.
        self.table_counts_per_doc = np.zeros((self.num_tables, self.num_documents), dtype=np.int32)
        # Stores the table (topic) assignment of each customer in each iteration
        # tableAssignments[i][j] gives the table assignment of customer j of the ith document.
        self.table_assignments = []
        # The following 4 parameters are arraylist (list) and not maps (dict) because,
        # if they are K tables, they are continuously numbered from 0 to K-1 and hence we can directly index them
        # Mean vector associated with each table in the current iteration.
        # This is the bayesian mean (i.e has the prior part too)
        # Use shared memory
        self.table_means = SharedArray.create((self.num_tables, self.embedding_size), "float")
        # log-determinant of covariance matrix for each table.
        # Since 0.5 * logDet is required in (see logMultivariateTDensity), therefore that value is kept.
        # Use shared memory
        self.log_determinants = SharedArray.create(self.num_tables, "float")
        # Stores the squared sum of the vectors of customers at a given table
        self.sum_squared_table_customers = np.zeros((self.num_tables, self.embedding_size, self.embedding_size), dtype=np.float64)

        # Cholesky Lower Triangular Decomposition of covariance matrix associated with each table.
        # Use shared memory
        self.table_cholesky_ltriangular_mat = SharedArray.create(
            (self.num_tables, self.embedding_size, self.embedding_size), "float"
        )

        # Normal inverse wishart prior
        self.prior = Wishart(self.vocab_embeddings, kappa=kappa)

        # Pre-compute the outer product of each vector with itself
        self.sqaured_embeddings = np.zeros((self.vocab_embeddings.shape[0], self.embedding_size, self.embedding_size), dtype=np.float64)
        for v_id in range(self.vocab_embeddings.shape[0]):
            self.sqaured_embeddings[v_id] = np.outer(self.vocab_embeddings[v_id], self.vocab_embeddings[v_id])

        # Cache k_0\mu_0\mu_0^T, only compute it once
        # Used in calculate_table_params()
        self.k0mu0mu0T = self.prior.kappa * np.outer(self.prior.mu, self.prior.mu)

        # We use this a lot
        self.log_pi = np.log(np.pi)

        self.num_words_for_formatting = num_words_for_formatting

        self.aliases = VoseAliases.create(self.num_terms, self.num_tables)

        # Speed up random sampling by drawing batches of random numbers at once
        self.rng = BatchedRands()

        self.log.info("Initializing assignments")
        self.initializer = initializer
        self.initialize()

    def initialize(self):
        """
        Initialize the gibbs sampler state.

        I start with log N tables and randomly initialize customers to those tables.

        """
        # First check the prior degrees of freedom.
        # It has to be >= num_dimension
        if self.prior.nu < self.embedding_size:
            self.log.warn("The initial degrees of freedom of the prior is less than the dimension!. "
                          "Setting it to the number of dimensions: {}".format(self.embedding_size))
            self.prior.nu = self.embedding_size

        deg_of_freedom = self.prior.nu - self.embedding_size + 1
        # Now calculate the covariance matrix of the multivariate T-distribution
        coeff = (self.prior.kappa + 1.) / (self.prior.kappa * deg_of_freedom)
        sigma_T = self.prior.sigma * coeff
        # This features in the original code, but doesn't get used
        # Or is it just to check that the invert doesn't fail?
        #sigma_Tinv = inv(sigma_T)
        sigma_TDet_sign, sigma_TDet = slogdet(sigma_T)
        if sigma_TDet_sign != 1:
            raise ValueError("sign of log determinant of initial sigma is {}".format(sigma_TDet_sign))

        # Storing zeros in sumTableCustomers and later will keep on adding each customer.
        self.sum_squared_table_customers[:] = 0
        # Means are set to the prior and then updated as we add each assignment
        self.table_means.np[:] = self.prior.mu

        # Initialize the cholesky decomp of each table, with no counts yet
        for table in range(self.num_tables):
            self.table_cholesky_ltriangular_mat.np[table] = self.prior.chol_sigma.copy()

        # Speed up random samples by drawing batches
        rng = BatchedRandInts(self.num_tables, batch_size=10000)

        # Randomly assign customers to tables
        self.table_assignments = []
        pbar = get_progress_bar(len(self.corpus), title="Initializing", show_progress=self.show_progress)
        for doc_num, doc in enumerate(pbar(self.corpus)):
            if self.initializer is None:
                # Default, random initialization
                tables = list(rng.integers(len(doc)))
            else:
                # Custom initializer provided
                # Use it to get topics for this document
                tables = self.initializer(doc_num, doc)

            self.table_assignments.append(tables)
            for (word, table) in zip(doc, tables):
                self.table_counts.np[table] += 1
                self.table_counts_per_doc[table, doc_num] += 1
                # update the sumTableCustomers
                self.sum_squared_table_customers[table] += self.sqaured_embeddings[word]

                self.update_table_params(table, word)

        # Output initial perplexity
        self.log.info("Computing average LL")
        ave_ll = calculate_avg_ll(
            get_progress_bar(len(self.corpus))(self.corpus), self.table_assignments, self.vocab_embeddings,
            self.table_means.np, self.table_cholesky_ltriangular_mat.np,
            self.prior, self.table_counts_per_doc
        )
        self.log.info("Average LL after initialization: {:.3e}".format(ave_ll))

    def update_table_params(self, table_id, cust_id, is_removed=False):
        count = self.table_counts.np[table_id]
        k_n = self.prior.kappa + count
        nu_n = self.prior.nu + count
        scaleTdistrn = (k_n + 1.) / (k_n * (float(nu_n) - self.embedding_size + 1.))

        if is_removed:
            # Now use the rank1 downdate to calculate the cholesky decomposition of the updated covariance matrix
            # The update equation is
            #   \Sigma_(N+1) =\Sigma_(N) - (k_0 + N+1) / (k_0 + N)(X_{n} - \mu_{n-1})(X_{n} - \mu_{n-1}) ^ T
            # Therefore x = sqrt((k_0 + N - 1) / (k_0 + N)) (X_{n} - \mu_{n})
            # Note here \mu_n will be the mean before updating.
            # After updating sigma_n, we will update \mu_n.

            # calculate (X_{n} - \mu_{n-1})
            # This uses the old mean, not yet updated
            x = (self.vocab_embeddings[cust_id] - self.table_means.np[table_id]) * np.sqrt((k_n + 1.) / k_n)
            # The Chol rank1 downdate modifies the array in place
            with self.table_cholesky_ltriangular_mat.lock:
                chol_rank1_downdate(self.table_cholesky_ltriangular_mat.np[table_id], x)

            # Update the mean
            new_mean = self.table_means.np[table_id] * (k_n + 1.)
            new_mean -= self.vocab_embeddings[cust_id]
            new_mean /= k_n
            with self.table_means.lock:
                self.table_means.np[table_id] = new_mean
        else:
            # New customer is added
            new_mean = self.table_means.np[table_id] * (k_n - 1.)
            new_mean += self.vocab_embeddings[cust_id]
            new_mean /= k_n
            with self.table_means.lock:
                self.table_means.np[table_id] = new_mean

            # We need to recompute det(Sig) and (v_{d,i} - mu) . Sig^-1 . (v_{d,i} - mu)
            # v_{d,i} is the word vector being added

            # The rank1 update equation is
            #  \Sigma_{n+1} = \Sigma_{n} + (k_0 + n + 1) / (k_0 + n) * (x_{n+1} - \mu_{n+1})(x_{n+1} - \mu_{n+1}) ^ T
            # calculate (X_{n} - \mu_{n-1})
            # This time we update the mean first and use the new mean
            x = (self.vocab_embeddings[cust_id] - self.table_means.np[table_id]) * np.sqrt(k_n / (k_n - 1.))
            # The update modifies the decomp array in place
            with self.table_cholesky_ltriangular_mat.lock:
                chol_rank1_update(self.table_cholesky_ltriangular_mat.np[table_id], x)

        # Calculate the 0.5 * log(det) + D / 2 * scaleTdistrn
        # The scaleTdistrn is because the posterior predictive distribution sends in a scaled value of \Sigma
        with self.log_determinants.lock:
            self.log_determinants.np[table_id] = \
                np.sum(np.log(np.diagonal(self.table_cholesky_ltriangular_mat.np[table_id]))) \
                + self.embedding_size * np.log(scaleTdistrn) / 2.

    def log_multivariate_tdensity(self, x, table_id):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.

        """
        if x.ndim > 1:
            logprobs = np.zeros(x.shape[0], dtype=np.float64)
            for i in range(x.shape[0]):
                logprobs[i] = self.log_multivariate_tdensity(x[i], table_id)
            return logprobs

        count = self.table_counts.np[table_id]
        k_n = self.prior.kappa + count
        nu_plus_d = self.prior.nu + count + 1.
        nu = nu_plus_d - self.embedding_size
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * nu))
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver
        # first calculate (x-mu)
        x_minus_mu = x - self.table_means.np[table_id]
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn * self.table_cholesky_ltriangular_mat.np[table_id]
        # Disabling the finiteness check speeds things up
        #solved = solve_triangular(ltriangular_chol, x_minus_mu, check_finite=False, lower=True)
        solved = _fast_solve_triangular(ltriangular_chol, x_minus_mu)
        # Now take xTx (dot product)
        val = (solved ** 2.).sum(-1)

        logprob = gammaln(nu_plus_d / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + self.log_pi) +
                          self.log_determinants.np[table_id] +
                          nu_plus_d / 2. * np.log(1. + val / nu)
                  )
        return logprob

    def log_multivariate_tdensity_tables(self, x):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.
        This version computes the likelihood for all tables in parallel.

        """
        count = self.table_counts.np
        k_n = self.prior.kappa + count
        nu_plus_d = self.prior.nu + count + 1.
        nu = nu_plus_d - self.embedding_size
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * nu))
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver first calculate (x-mu)
        x_minus_mu = x[None, :] - self.table_means.np
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn[:, None, None] * self.table_cholesky_ltriangular_mat.np
        # We can't do solve_triangular for all matrices at once in scipy
        val = np.zeros(self.num_tables, dtype=np.float64)
        for table in range(self.num_tables):
            #table_solved = solve_triangular(ltriangular_chol[table], x_minus_mu[table], check_finite=False, lower=True)
            table_solved = _fast_solve_triangular(ltriangular_chol[table], x_minus_mu[table])
            # Now take xTx (dot product)
            val[table] = (table_solved ** 2.).sum()

        logprob = gammaln(nu_plus_d / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + self.log_pi) +
                          self.log_determinants.np +
                          nu_plus_d / 2. * np.log(1. + val / nu)
                  )
        return logprob

    def sample(self, num_iterations):
        """
        for num_iters:
            for each customer
                remove him from his old_table and update the table params.
                if old_table is empty:
                    remove table
                Calculate prior and likelihood for this customer sitting at each table
                sample for a table index
                if new_table is equal to old_table
                    don't have to update the parameters
                else update params of the old table.
        """
        if self.show_topics is not None:
            print("Topics after initialization")
            print(self.format_topics())
            # Compute the overall usage of topics across the training corpus
            topic_props = self.table_counts_per_doc.sum(axis=1).astype(np.float64)
            topic_props /= topic_props.sum()
            print("Words using topics: {}".format(
                ", ".join("{}={:.1f}%".format(i, prop) for i, prop in enumerate(topic_props*100.))))
            topic_doc_props = (self.table_counts_per_doc > 0).astype(np.float64).sum(axis=1)
            topic_doc_props /= self.num_documents
            print("Docs using topics: {}".format(
                ", ".join("{}={:.1f}%".format(i, prop) for i, prop in enumerate(topic_doc_props*100.))))

        # Batch random samples to speed up sampling
        rng = BatchedRands()

        # Cache log density calculations to avoid repeated (expensive) computations
        densities = LogDensityCache(self, self.num_tables)

        with VoseAliasUpdater(
                self.aliases, self.vocab_embeddings,
                self.prior.kappa, self.prior.nu,
                self.table_counts, self.table_means, self.table_cholesky_ltriangular_mat,
                self.log_determinants, das_normalization=self.replicate_das,
        ) as alias_updater:
            for iteration in range(num_iterations):
                stats = SamplingDiagnostics()
                self.log.info("Iteration {}".format(iteration))

                alias_updater.unpause()
                pbar = get_progress_bar(len(self.corpus), title="Sampling", show_progress=self.show_progress)
                for d, doc in enumerate(pbar(self.corpus)):
                    if self.show_topics is not None and self.show_topics > 0 and d % self.show_topics == 0:
                        print("Topics after {:,} docs".format(d))
                        print(self.format_topics())

                    for w, cust_id in enumerate(doc):
                        x = self.vocab_embeddings[cust_id]

                        # Remove custId from his old_table
                        old_table_id = self.table_assignments[d][w]
                        self.table_assignments[d][w] = -1  # Doesn't really make any difference, as only counts are used
                        with self.table_counts.lock:
                            self.table_counts.np[old_table_id] -= 1
                        self.table_counts_per_doc[old_table_id, d] -= 1
                        # Update vector means etc
                        self.sum_squared_table_customers[old_table_id] -= self.sqaured_embeddings[cust_id]

                        # Topic 'old_table_id' now has one member fewer
                        # Just update params for this customer
                        self.update_table_params(old_table_id, cust_id, is_removed=True)
                        densities.clear(old_table_id)

                        # Under the alias method, we only do the full likelihood computation for topics
                        # that already have a non-zero count in the current document
                        non_zero_tables = np.where(self.table_counts_per_doc[:, d] > 0)[0]
                        if len(non_zero_tables) == 0:
                            # If there's only one word in a doc, there are no topics to compute the full posterior for
                            no_non_zero = True
                        else:
                            no_non_zero = False
                            # We only compute the posterior for these topics
                            log_priors = np.log(self.table_counts_per_doc[non_zero_tables, d])
                            log_likelihoods = np.zeros(len(non_zero_tables), dtype=np.float32)
                            for nz_table, table in enumerate(non_zero_tables):
                                log_likelihoods[nz_table] = densities.logprob(cust_id, x, table)
                            log_posterior = log_priors + log_likelihoods

                            # To prevent overflow, subtract by log(p_max)
                            max_log_posterior = log_posterior.max()
                            scaled_posterior = log_posterior - max_log_posterior
                            if self.replicate_das:
                                # Not doing this now, but following what the Java impl does, however odd that seems
                                psum = np.sum(np.exp(scaled_posterior))
                            else:
                                # Java impl subtracts max before computing psum, but this seems to be wrong
                                # We still subtract first, but then multiply by the max prob afterwards
                                psum = np.exp(np.log(np.sum(np.exp(scaled_posterior))) + max_log_posterior)
                            # Now just use the scaled log posterior in the same way as in the Java impl
                            # They have a bin-search method for sampling from the cumulative dist,
                            # but we simply normalize and use Numpy to sample
                            unnormed_posterior = np.exp(scaled_posterior)
                            normed_posterior = unnormed_posterior / unnormed_posterior.sum()
                            normed_posterior_cum = np.cumsum(normed_posterior)

                        # Don't let the alias parameters get updated in the middle of the sampling
                        self.aliases.lock.acquire_read(cust_id)
                        # If replicate_das is set, psum and likelihood_sum are scaled by different factors,
                        #  which I'm pretty sure is wrong
                        # I have checked that we are getting very similar values for select_pr to the Java impl
                        select_pr = psum / (psum + self.alpha*self.aliases.likelihood_sum.np[cust_id])

                        # MHV to draw new topic
                        # Take a number of Metropolis-Hastings samples
                        current_sample = old_table_id
                        # Calculate the true likelihood of this word under the current sample,
                        # for calculating acceptance prob
                        current_sample_log_prob = densities.logprob(cust_id, x, current_sample)
                        for r in range(self.mh_steps):
                            # 1. Flip a coin
                            if not no_non_zero and rng.random() < select_pr:
                                # Choose from the exactly computed posterior dist, only allowing
                                # topics already sampled in the doc
                                temp = rng.choice_cum(normed_posterior_cum)
                                new_sample = non_zero_tables[temp]
                                stats.log_select_pr(True, select_pr)
                            else:
                                # Choose from the alias, allowing any topic but using slightly
                                # out-of-date likelihoods
                                new_sample = self.aliases.sample_vose(cust_id)
                                stats.log_select_pr(False, select_pr)

                            if new_sample != current_sample:
                                # 2. Find acceptance probability
                                new_sample_log_prob = densities.logprob(cust_id, x, new_sample)
                                # This can sometimes generate an overflow warning from Numpy
                                # We don't care, though: in that case acceptance > 1., so we always accept
                                with np.errstate(over="ignore"):
                                    if not self.replicate_das:
                                        # From my reading of:
                                        # Li et al. (2014): Reducing the sampling complexity of topic models
                                        # the acceptance probability should be as follows:
                                        acceptance = \
                                            (self.table_counts_per_doc[new_sample, d] + self.alpha) / \
                                            (self.table_counts_per_doc[current_sample, d] + self.alpha) * \
                                            np.exp(new_sample_log_prob - current_sample_log_prob) * \
                                            (self.table_counts_per_doc[current_sample, d]*np.exp(current_sample_log_prob) +
                                             self.alpha*np.exp(self.aliases.log_likelihoods.np[cust_id, current_sample])) / \
                                            (self.table_counts_per_doc[new_sample, d]*np.exp(new_sample_log_prob) +
                                             self.alpha*np.exp(self.aliases.log_likelihoods.np[cust_id, new_sample]))
                                        # The difference is the Java impl doesn't exp the log likelihood in the last
                                        # fraction, i.e. it uses a log prob instead of a prob
                                    else:
                                        # The Java implementation, however, does this
                                        # Note that log_likelihoods[cust_id] stores the log of what is
                                        #  in w in the Java impl, so we exp it here to make this identical
                                        # I have checked that this is behaving in a way very similar to the
                                        #  Java version, producing the same sort of values and getting more extreme
                                        #  values as the first iteration goes on
                                        acceptance = \
                                            (self.table_counts_per_doc[new_sample, d] + self.alpha) / \
                                            (self.table_counts_per_doc[current_sample, d] + self.alpha) * \
                                            np.exp(new_sample_log_prob - current_sample_log_prob) * \
                                            (self.table_counts_per_doc[current_sample, d]*current_sample_log_prob +
                                             self.alpha*np.exp(self.aliases.log_likelihoods.np[cust_id, current_sample])) / \
                                            (self.table_counts_per_doc[new_sample, d]*new_sample_log_prob +
                                             self.alpha*np.exp(self.aliases.log_likelihoods.np[cust_id, new_sample]))

                                # 3. Compare against uniform[0,1]
                                # If the acceptance prob > 1, we always accept: this means the new sample
                                # has a higher probability than the old
                                if isinf(acceptance) or acceptance >= 1. or np.random.sample() < acceptance:
                                    # No need to sample if acceptance >= 1
                                    # If the acceptance prob < 1, sample whether to accept or not, such that
                                    # the more likely the new sample is compared to the old, the more likely we
                                    # are to keep it
                                    current_sample = new_sample
                                    current_sample_log_prob = new_sample_log_prob
                                    stats.log_acceptance(True, acceptance)
                                else:
                                    stats.log_acceptance(False, acceptance)
                                # NOTE: There seems to be a small error in the Java implementation here
                                # On the last MH step, it doesn't make any difference whether we accept the
                                # sample or not - we always end up using it
                        self.aliases.lock.release_read()

                        if current_sample == old_table_id:
                            stats.log_sampled_same()
                        else:
                            stats.log_sampled_different()

                        # Now have a new assignment: add its counts
                        self.table_assignments[d][w] = current_sample
                        with self.table_counts.lock:
                            self.table_counts.np[current_sample] += 1
                        self.table_counts_per_doc[current_sample, d] += 1
                        self.sum_squared_table_customers[current_sample] += self.sqaured_embeddings[cust_id]

                        self.update_table_params(current_sample, cust_id)
                        densities.clear(current_sample)

                # Pause the alias updater until we start the next iteration
                alias_updater.pause()

                # Compute and output average LL
                self.log.info("Computing average LL")
                ave_ll = calculate_avg_ll(
                    get_progress_bar(len(self.corpus))(self.corpus), self.table_assignments, self.vocab_embeddings,
                    self.table_means.np, self.table_cholesky_ltriangular_mat.np,
                    self.prior, self.table_counts_per_doc
                )
                self.log.info("Average LL: {:.3e}".format(ave_ll))

                # Output some useful stats about sampling
                if stats.acceptance_used():
                    self.log.info("Acceptance rate = {:.2f}%, mean acceptance: {:.2e} ({:,} samples draw)".format(
                        stats.acceptance_rate()*100., stats.mean_acceptance(), stats.acceptance_samples()))
                else:
                    self.log.info("No new samples drawn")
                self.log.info("Prior select rate = {:.2f}%, mean select_pr = {:.2f}".format(
                    stats.select_pr_rate() * 100., stats.mean_select_pr()
                ))
                self.log.info("Chose new sample: {:.2f}%".format(stats.sample_change_rate() * 100.))

                if self.show_topics is not None:
                    print("Topics after iteration {}".format(iteration))
                    print(self.format_topics())
                    # Compute the overall usage of topics across the training corpus
                    topic_props = self.table_counts_per_doc.sum(axis=1).astype(np.float64)
                    topic_props /= topic_props.sum()
                    print("Words using topics: {}".format(
                        ", ".join("{}={:.1f}%".format(i, prop) for i, prop in enumerate(topic_props*100.))))
                    topic_doc_props = (self.table_counts_per_doc > 0).astype(np.float64).sum(axis=1)
                    topic_doc_props /= self.num_documents
                    print("Docs using topics: {}".format(
                        ", ".join("{}={:.1f}%".format(i, prop) for i, prop in enumerate(topic_doc_props*100.))))

                if self.save_path is not None:
                    self.log.info("Saving model")
                    self.save()

    def format_topics(self, num_words=10, topics=None):
        if topics is None:
            topics = list(range(self.num_tables))

        if self.num_words_for_formatting is not None:
            # Limit to the first N words to consider for inclusion in a topic's representation
            embeddings = self.vocab_embeddings[:self.num_words_for_formatting]
        else:
            embeddings = self.vocab_embeddings

        topic_fmt = []
        for topic in topics:
            if self.table_counts.np[topic] == 0:
                # This topic is never used, so should be considered to have been abandoned by the sampler (for now)
                topic_fmt.append("{}: unused")
            else:
                # Compute the density for all words in the vocab
                word_scores = self.log_multivariate_tdensity(embeddings, topic)
                word_probs = np.exp(word_scores - word_scores.max())
                word_probs /= word_probs.sum()
                topic_fmt.append(
                    "{}: {}".format(
                        topic,
                        " ".join(
                            "{} ({:.2e})".format(self.vocab[word], word_probs[word])
                            for word in [w for w in np.argsort(-word_scores)][:num_words]
                        )
                    )
                )

        return "\n".join(topic_fmt)

    def save(self):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(self.save_path)

        with open(os.path.join(self.save_path, "params.json"), "w") as f:
            json.dump({
                "alpha": self.alpha,
                "vocab": self.vocab,
                "num_tables": self.num_tables,
                "kappa": self.prior.kappa,
            }, f)
        for name, data in [
            ("table_counts", self.table_counts.np),
            ("table_means", self.table_means.np),
            ("log_determinants", self.log_determinants.np),
            ("sum_squared_table_customers", self.sum_squared_table_customers),
            ("table_cholesky_ltriangular_mat", self.table_cholesky_ltriangular_mat.np),
            ("vocab_embeddings", self.vocab_embeddings),
        ]:
            with open(os.path.join(self.save_path, "{}.pkl".format(name)), "wb") as f:
                pickle.dump(data, f)


class VoseAliases:
    """
    Implementation of Vose alias for faster sampling. This implementation
    stores all the aliases for the entire vocabulary in a single data
    structure, reducing memory overheads and making synchronization more
    efficient.

    The old implementation, where a separate table was maintained for every
    word, each with a separate lock, is in `._old_alias.py` for reference.

    See https://github.com/rajarshd/Gaussian_LDA/blob/master/src/util/VoseAlias.java.

    We enforce locking with a single lock that ensures that reading and
    writing of the same word's alias don't overlap. It's assumed that there's
    exactly one process reading and one writing, as in the Gaussian LDA
    sampler.

    """
    def __init__(self, alias, prob, log_likelihoods, likelihood_sum, lock):
        self.n = alias.shape[1]
        # Alias indices
        self.alias = alias
        # Contains proportions and alias probabilities
        self.prob = prob
        # Contains log likelihoods of the word given each topic
        self.log_likelihoods = log_likelihoods
        self.likelihood_sum = likelihood_sum

        #self.gen = random.Random()
        self.gen = BatchedRands()

        self.lock = lock

    @staticmethod
    def create(num_words, num_topics):
        # Create big arrays to hold all the values for all words
        alias = SharedArray.create((num_words, num_topics), "int")
        prob = SharedArray.create((num_words, num_topics), "float")
        log_likelihoods = SharedArray.create((num_words, num_topics), "float")
        likelihood_sum = SharedArray.create(num_words, "float")
        lock = TwoSidedLock.create()
        return VoseAliases(alias, prob, log_likelihoods, likelihood_sum, lock)

    def __getstate__(self):
        return self.alias, self.prob, self.log_likelihoods, self.likelihood_sum, self.lock

    def __setstate__(self, state):
        self.__init__(*state)

    def sample_vose(self, word):
        # 1. Generate a fair die roll from an n-sided die
        #fair_die = self.gen.randint(0, self.n-1)
        fair_die = self.gen.integer(self.n)
        # 2. Flip a biased coin that comes up heads with probability Prob[i]
        m = self.gen.random()
        # 3. If the coin comes up "heads," return i. Otherwise, return Alias[i]
        if m > self.prob.np[word, fair_die]:
            return self.alias.np[word, fair_die]
        else:
            return fair_die

    def sample_numpy(self, word):
        """
        Draw sample without using the Vose alias, instead just using Numpy's sample method.
        This should be less efficient, presumably, but is guaranteed to sample correctly,
        so I'm using it as debugging.

        """
        lp = self.log_likelihoods.np[word, :]
        lp -= lp.max()
        p = np.exp(lp)
        p /= p.sum()
        return np.random.choice(len(p), p=p)


class VoseAliasUpdater(Process):
    """
    Following the Java implementation, a process running in the background keeps updating
    the Vose alias' tables and copying the result into the tables used by the main alias.

    """
    def __init__(self, aliases, embeddings, kappa, nu, table_counts, table_means, table_cholesky_ltriangular_mat,
                 log_determinants, das_normalization=True):
        super().__init__()
        self.aliases = aliases
        self.das_normalization = das_normalization

        # Gaussian parameters
        self.log_determinants = log_determinants.np
        self.table_cholesky_ltriangular_mat = table_cholesky_ltriangular_mat.np
        self.table_means = table_means.np
        self.table_counts = table_counts.np
        # Prior params
        self.nu = nu
        self.kappa = kappa

        self.embeddings = embeddings

        self.embedding_size = self.embeddings.shape[1]
        self.num_tables = self.table_counts.shape[0]

        self.done = mp.Event()
        self.initialized = mp.Event()
        # Make sure we don't interleave bits of likelihood computation and gaussian parameter updates
        self.param_lock = GaussianLock(table_counts, table_means, table_cholesky_ltriangular_mat, log_determinants)
        # Pause the updating once an iteration finishes, while we do admin in the main process
        # Once it starts again, we can carry on updating as before
        self.running = mp.Event()

    def __enter__(self):
        # Start the process going and wait until it's finished it's first update round
        self.start_and_init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Set the done flag
        self.done.set()
        # Once the process has finished its next iteration, it will stop
        # Wait until it's finished
        self.join()

    def start_and_init(self):
        """
        Start the background updater process running and then wait until it's finished
        its first run and the parameters are initialized.

        """
        self.running.set()
        self.start()
        self.initialized.wait()

    def run(self):
        # Before starting looping, we make sure to go once over everything
        # The main process will wait for self.initialized to be set before sampling starts
        while not self.done.is_set():
            # Iterate over the word vectors
            for term_id, x in enumerate(self.embeddings):
                # Compute the likelihood under each possible topic assignment
                log_likelihoods = self.log_multivariate_tdensity_tables(x)
                # To prevent overflow, subtract by log(p_max)
                # This is because when we will be normalizing after exponentiating
                ll_max = log_likelihoods.max()
                log_likelihoods_scaled = log_likelihoods - ll_max
                # Exp them all to get the new w array
                w = np.exp(log_likelihoods_scaled)
                if self.das_normalization:
                    # Replicating exactly what the Java code does
                    # The sum that is used is scaled by the max, which is odd
                    # This value is used to compute select_pr
                    likelihood_sum = w.sum()
                    # We store in log_likelihoods[term] the scaled version of the LL
                    # In the Java impl, this is exp'd and stored in w
                    log_likelihoods = log_likelihoods_scaled
                else:
                    # Sum of likelihood has to be scaled back to its original sum, before we subtracted the max log
                    likelihood_sum = np.exp(np.log(w.sum()) + ll_max)
                # Update the alias and probs using this w
                new_alias, new_prob = self.generate_table(w)
                # Update the parameters of this word's alias
                # Acquire the lock to write to this word
                self.aliases.lock.acquire_write(term_id)
                np.copyto(self.aliases.log_likelihoods.np[term_id], log_likelihoods)
                np.copyto(self.aliases.alias.np[term_id], new_alias)
                np.copyto(self.aliases.prob.np[term_id], new_prob)
                self.aliases.likelihood_sum.np[term_id] = likelihood_sum  # This one is a single value
                self.aliases.lock.release_write()

            # If this was the first iteration, we tell the main process that we've finished it
            self.initialized.set()

            # If running has been unset, we pause until it gets set again
            while not self.running.wait(timeout=0.1):
                # Not running yet, but check whether done has been set: then we should exit straight away
                if self.done.is_set():
                    break

    def pause(self):
        # Unset running
        # Once the current iteration is complete, the updater will wait until running is set again
        self.running.clear()

    def unpause(self):
        self.running.set()

    def generate_table(self, w):
        # This implementation is copied from Java GaussianLDA
        # Generate a new prob array and alias table
        alias = np.zeros(self.num_tables, dtype=np.int32)
        # Contains proportions and alias probabilities
        prob = np.zeros(self.num_tables, dtype=np.float32)

        # 1. Create two worklists, Small and Large
        # 2. Multiply each probability by n
        p = (w * self.num_tables) / w.sum()
        # 3. For each scaled probability pi:
        #   a. If pi<1, add i to Small.
        #   b. Otherwise(pi>=1), add i to Large.
        small, large = [], []
        for i, pi in enumerate(p):
            if pi < 1.:
                small.append(i)
            else:
                large.append(i)
        #small = list(np.where(p < 1.)[0])
        #large = list(np.where(p >= 1.)[0])
        # 4. While Small and Large are not empty : (Large might be emptied first)
        #    a. Remove the first element from Small; call it l.
        #    b. Remove the first element from Large; call it g.
        #    c. Set Prob[l] = pl.
        #    d. Set Alias[l] = g.
        #    e. Set pg : = (pg + pl)−1. (This is a more numerically stable option.)
        #    f. If pg<1, add g to Small.
        #    g. Otherwise(pg≥1), add g to Large.
        while len(small) and len(large):
            l = small.pop(0)
            g = large.pop(0)
            prob[l] = p[l]
            alias[l] = g
            p[g] = (p[g] + p[l]) - 1.
            if p[g] < 1.:
                small.append(g)
            else:
                large.append(g)

        # 5. While Large is not empty :
        #    a. Remove the first element from Large; call it g.
        #    b. Set Prob[g] = 1.
        if len(large):
            prob[large] = 1.

        # 6. While Small is not empty : This is only possible due to numerical instability.
        #    a. Remove the first element from Small; call it l.
        #    b. Set Prob[l] = 1.
        if len(small):
            prob[small] = 1.

        return alias, prob

    def __log_multivariate_tdensity_tables(self, x):
        """
        A local version of the likelihood function from the main model, using the
        copy of the parameters we have in our local process.

        """
        # Make sure the main process doesn't update the parameters in the middle of the computation
        with self.param_lock:
            # Copy these values so we can release the lock and use a consistent state
            count = self.table_counts.copy()
            table_cholesky_ltriangular_mat = self.table_cholesky_ltriangular_mat.copy()
            log_determinants = self.log_determinants.copy()
            table_means = self.table_means.copy()

        k_n = self.kappa + count
        nu_n = self.nu + count
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.)))
        nu = self.nu + count - self.embedding_size + 1.
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver first calculate (x-mu)
        x_minus_mu = x[None, :] - table_means
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn[:, None, None] * table_cholesky_ltriangular_mat
        # We can't do solve_triangular for all matrices at once in scipy
        val = np.zeros(self.num_tables, dtype=np.float64)
        for table in range(self.num_tables):
            #table_solved = solve_triangular(ltriangular_chol[table], x_minus_mu[table], check_finite=False, lower=True)
            table_solved = _fast_solve_triangular(ltriangular_chol[table], x_minus_mu[table])
            # Now take xTx (dot product)
            val[table] = (table_solved ** 2.).sum()

        logprob = gammaln((nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + np.log(math.pi)) +
                          log_determinants +
                          (nu + self.embedding_size) / 2. * np.log(1. + val / nu)
                  )
        return logprob

    def log_multivariate_tdensity(self, x, table_id):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.

        """
        # Make sure the main process doesn't update the parameters in the middle of the computation
        with self.param_lock:
            # Copy these values so we can release the lock and use a consistent state
            count = self.table_counts[table_id].copy()
            table_cholesky_ltriangular_mat = self.table_cholesky_ltriangular_mat[table_id].copy()
            log_determinant = self.log_determinants[table_id].copy()
            table_mean = self.table_means[table_id].copy()

        k_n = self.kappa + count
        nu_n = self.nu + count
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.)))
        nu = self.nu + count - self.embedding_size + 1.
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver
        # first calculate (x-mu)
        x_minus_mu = x - table_mean
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn * table_cholesky_ltriangular_mat
        #solved = solve_triangular(ltriangular_chol, x_minus_mu, check_finite=False, lower=True)
        solved = _fast_solve_triangular(ltriangular_chol, x_minus_mu)
        # Now take xTx (dot product)
        val = (solved ** 2.).sum()

        logprob = gammaln((nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + np.log(math.pi)) +
                          log_determinant +
                          (nu + self.embedding_size) / 2. * np.log(1. + val / nu)
                  )
        return logprob

    def log_multivariate_tdensity_tables(self, x):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.
        This version computes the likelihood for all tables in parallel.

        """
        # Make sure the main process doesn't update the parameters in the middle of the computation
        with self.param_lock:
            # Copy these values so we can release the lock and use a consistent state
            counts = self.table_counts[:].copy()
            table_cholesky_ltriangular_mats = self.table_cholesky_ltriangular_mat[:].copy()
            log_determinants = self.log_determinants[:].copy()
            table_means = self.table_means[:].copy()

        k_n = self.kappa + counts  # Vector (K)
        nu_n = self.nu + counts    # Vector (K)
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.)))   # Vector (K)
        nu = self.nu + counts - self.embedding_size + 1.   # Vector (K)
        x_minus_mu = x[None, :] - table_means  # Matrix (K, D)
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn[:, None, None] * table_cholesky_ltriangular_mats
        # We can't do solve_triangular for all matrices at once in scipy
        val = np.zeros(self.num_tables, dtype=np.float64)
        for table in range(self.num_tables):
            #table_solved = solve_triangular(ltriangular_chol[table], x_minus_mu[table], check_finite=False, lower=True)
            table_solved = _fast_solve_triangular(ltriangular_chol[table], x_minus_mu[table])
            # Now take xTx (dot product)
            val[table] = (table_solved ** 2.).sum()

        logprob = gammaln((nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + np.log(math.pi)) +
                          log_determinants +
                          (nu + self.embedding_size) / 2. * np.log(1. + val / nu)
                  )
        return logprob


# Calling scipy's solve_triangular results in a lookup of the relevant
#  LAPACK function every time. Since we need to call it so many times and always
#  use the same internal function, force the lookup on load and use the funciton directly
trtrs, = get_lapack_funcs(('trtrs',))

def _fast_solve_triangular(a, b):
    try:
        x, info = trtrs(a, b, overwrite_b=False, lower=True, trans=0, unitdiag=False)
    except Exception as e:
        warnings.warn("error calling trtrs, trying again with solve_triangular: {}".format(e))
        solve_triangular(a, b, check_finite=True, lower=True)
        raise

    if info == 0:
        return x
    if info > 0:
        raise ValueError("solve_triangular error: singular matrix: resolution failed at diagonal %d" % (info-1))
    raise ValueError("solve_triangular error: illegal value in %d-th argument of internal trtrs" % (-info))


class SamplingDiagnostics:
    """
    Keeps track of a few statistics about the sampling process that can be useful to
    display between iterations to give an idea of what's happening during sampling.
    This is mainly for debugging, but is quite a useful thing to output to give
    an idea of what the sampler's doing in each iteration.

    """
    def __init__(self):
        self._select_pr_sum = 0.
        self._select_pr_uses = 0
        self._pr_accepted = 0
        self._acceptance_sum = 0.
        self._acceptance_uses = 0
        self._accepted_total = 0
        self._resampled_different = 0
        self._resampled_total = 0

    def log_select_pr(self, selected, select_pr):
        self._select_pr_sum += float(select_pr)
        self._select_pr_uses += 1
        if selected:
            self._pr_accepted += 1

    def log_acceptance(self, accepted, acceptance):
        self._acceptance_sum += float(acceptance)
        self._acceptance_uses += 1
        if accepted:
            self._accepted_total += 1

    def log_sampled_same(self):
        self._resampled_total += 1

    def log_sampled_different(self):
        self._resampled_total += 1
        self._resampled_different += 1

    def acceptance_used(self):
        return self._acceptance_uses > 0

    def mean_acceptance(self):
        return self._acceptance_sum / self._acceptance_uses

    def acceptance_rate(self):
        return float(self._accepted_total) / self._acceptance_uses

    def acceptance_samples(self):
        return self._acceptance_uses

    def mean_select_pr(self):
        return self._select_pr_sum / self._select_pr_uses

    def select_pr_rate(self):
        return float(self._pr_accepted) / self._select_pr_uses

    def sample_change_rate(self):
        return float(self._resampled_different) / self._resampled_total


class LogDensityCache:
    """
    Computing the log density of a given table for a given embedding is one
    of the most expensive operations in the sampling process. We speed things
    up by cacheing the values, within a single word-topic sample (i.e. while the
    parameters are kept fixed).

    """
    def __init__(self, trainer, num_tables):
        self.trainer = trainer
        self._cache = {}
        for table in range(num_tables):
            self._cache[table] = {}

    def clear(self, table):
        self._cache[table] = {}

    def logprob(self, word_id, x, table):
        try:
            return self._cache[table][word_id]
        except KeyError:
            self._cache[table][word_id] = self.trainer.log_multivariate_tdensity(x, table)
            return self._cache[table][word_id]
