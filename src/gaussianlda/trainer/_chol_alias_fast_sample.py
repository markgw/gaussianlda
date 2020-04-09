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

.. note::

   I've been working on this new version that runs the sampling in the alias
   updater background process and stores the samples instead of the probability
   tables. This is what's suggested by the authors of the sampling method and
   should save memory. However, the memory problems are currently more due to
   the large number of words.

"""
import functools
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import shutil
from contextlib import ExitStack
from multiprocessing import Process, sharedctypes
from operator import mul
import random

import numpy as np
from numpy.core.umath import isinf
from numpy.linalg import slogdet
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from gaussianlda.prior import Wishart
from gaussianlda.utils import get_logger, get_progress_bar, chol_rank1_downdate, chol_rank1_update, sum_logprobs


class GaussianLDAAliasTrainer:
    def __init__(self, corpus, vocab_embeddings, vocab, num_tables, alpha=None, kappa=0.1, log=None, save_path=None,
                 show_topics=None, mh_steps=2, num_words_for_formatting=None, das_normalization=True, show_progress=True):
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
        :param das_normalization: Use the normalization of probability distributions used by Das, Zaheer and Dyer's
            original implementation when computing the sampling probability to choose whether to use the document
            posterior or language model part of the topic posterior. If False, do not normalize in this way, but use
            an alternative, which looks to me like it's more correct mathematically.
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

        self.das_normalization = das_normalization

        # dataVectors
        self.vocab_embeddings = vocab_embeddings
        self.embedding_size = vocab_embeddings.shape[1]
        self.vocab_size = vocab_embeddings.shape[0]
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

        # Cache k_0\mu_0\mu_0^T, only compute it once
        # Used in calculate_table_params()
        self.k0mu0mu0T = self.prior.kappa * np.outer(self.prior.mu, self.prior.mu)

        self.num_words_for_formatting = num_words_for_formatting

        self.log.info("Initializing assignments")
        self.initialize()

        self.log.info("Creating {:,} aliases".format(self.vocab_size))
        pbar = get_progress_bar(self.vocab_size, title="Aliases")
        self.aliases = [
            VoseAlias.create(self.num_tables) for i in pbar(range(self.vocab_size))
        ]

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

        # Randomly assign customers to tables
        self.table_assignments = []
        pbar = get_progress_bar(len(self.corpus), title="Initializing", show_progress=self.show_progress)
        for doc_num, doc in enumerate(pbar(self.corpus)):
            tables = list(np.random.randint(self.num_tables, size=len(doc)))
            self.table_assignments.append(tables)
            for (word, table) in zip(doc, tables):
                self.table_counts.np[table] += 1
                self.table_counts_per_doc[table, doc_num] += 1
                # update the sumTableCustomers
                self.sum_squared_table_customers[table] += np.outer(self.vocab_embeddings[word], self.vocab_embeddings[word])

                self.update_table_params(table, word)

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
            if count == 0:
                # What are we supposed to do here? Presumably this just never happens on larger datasets
                raise ValueError("overall count for topic {} dropped to 0, so Cholesky update fails".format(table_id))
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
        nu_n = self.prior.nu + count
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.)))
        nu = self.prior.nu + count - self.embedding_size + 1.
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver
        # first calculate (x-mu)
        x_minus_mu = x - self.table_means.np[table_id]
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn * self.table_cholesky_ltriangular_mat.np[table_id]
        solved = solve_triangular(ltriangular_chol, x_minus_mu)
        # Now take xTx (dot product)
        val = (solved ** 2.).sum(-1)

        logprob = gammaln((nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + np.log(math.pi)) +
                          self.log_determinants.np[table_id] +
                          (nu + self.embedding_size) / 2. * np.log(1. + val / nu)
                  )
        return logprob

    def log_multivariate_tdensity_tables(self, x):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.
        This version computes the likelihood for all tables in parallel.

        """
        count = self.table_counts.np
        k_n = self.prior.kappa + count
        nu_n = self.prior.nu + count
        scaleTdistrn = np.sqrt((k_n + 1.) / (k_n * (nu_n - self.embedding_size + 1.)))
        nu = self.prior.nu + count - self.embedding_size + 1.
        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver first calculate (x-mu)
        x_minus_mu = x[None, :] - self.table_means.np
        # Now scale the lower tringular matrix
        ltriangular_chol = scaleTdistrn[:, None, None] * self.table_cholesky_ltriangular_mat.np
        # We can't do solve_triangular for all matrices at once in scipy
        val = np.zeros(self.num_tables, dtype=np.float64)
        for table in range(self.num_tables):
            table_solved = solve_triangular(ltriangular_chol[table], x_minus_mu[table])
            # Now take xTx (dot product)
            val[table] = (table_solved ** 2.).sum()

        logprob = gammaln((nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + np.log(math.pi)) +
                          self.log_determinants.np +
                          (nu + self.embedding_size) / 2. * np.log(1. + val / nu)
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

        with VoseAliasUpdater(
                self.aliases, self.vocab_embeddings,
                self.prior.kappa, self.prior.nu,
                self.table_counts, self.table_means, self.table_cholesky_ltriangular_mat,
                self.log_determinants, self.log, das_normalization=self.das_normalization,
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
                        # Check this isn't the last customer at the table
                        if self.table_counts.np[old_table_id] == 1:
                            # If we remove this customer from its table (to resample a topic for the word)
                            # there will be no counts for the topic, so we can't compute likelihoods
                            # There are various things we could do here, but we'll just skip resampling this
                            # sample: it shouldn't happen often on a large dataset anyway
                            continue
                        self.table_assignments[d][w] = -1  # Doesn't really make any difference, as only counts are used
                        with self.table_counts.lock:
                            self.table_counts.np[old_table_id] -= 1
                        self.table_counts_per_doc[old_table_id, d] -= 1
                        # Update vector means etc
                        self.sum_squared_table_customers[old_table_id] -= np.outer(x, x)

                        # Topic 'old_table_id' now has one member fewer
                        # Just update params for this customer
                        self.update_table_params(old_table_id, cust_id, is_removed=True)

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
                                log_likelihoods[nz_table] = self.log_multivariate_tdensity(x, table)
                            log_posterior = log_priors + log_likelihoods

                            # To prevent overflow, subtract by log(p_max)
                            max_log_posterior = log_posterior.max()
                            scaled_posterior = log_posterior - max_log_posterior
                            if self.das_normalization:
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

                        # Don't let the alias parameters get updated in the middle of the sampling
                        alias = self.aliases[cust_id]
                        with alias.lock:
                            select_pr = psum / (psum + self.alpha*alias.likelihood_sum.np)

                            # MHV to draw new topic
                            # Take a number of Metropolis-Hastings samples
                            current_sample = old_table_id
                            # Calculate the true likelihood of this word under the current sample,
                            # for calculating acceptance prob
                            current_sample_log_prob = self.log_multivariate_tdensity(x, current_sample)
                            for r in range(self.mh_steps):
                                # 1. Flip a coin
                                if not no_non_zero and np.random.sample() < select_pr:
                                    # Choose from the exactly computed posterior dist, only allowing
                                    # topics already sampled in the doc
                                    temp = np.random.choice(len(non_zero_tables), p=normed_posterior)
                                    new_sample = non_zero_tables[temp]
                                    stats.log_select_pr(True, select_pr)
                                else:
                                    # Choose from the alias, allowing any topic but using slightly
                                    # out-of-date likelihoods
                                    new_sample = alias.sample_vose()
                                    stats.log_select_pr(False, select_pr)

                                if new_sample != current_sample:
                                    # 2. Find acceptance probability
                                    new_sample_log_prob = self.log_multivariate_tdensity(x, new_sample)
                                    # This can sometimes generate an overflow warning from Numpy
                                    # We don't care, though: in that case acceptance > 1., so we always accept
                                    with np.errstate(over="ignore"):
                                        # From my reading of:
                                        # Li et al. (2014): Reducing the sampling complexity of topic models
                                        # the acceptance probability should be as follows:
                                        acceptance = \
                                            (self.table_counts_per_doc[new_sample, d] + self.alpha) / \
                                            (self.table_counts_per_doc[current_sample, d] + self.alpha) * \
                                            np.exp(new_sample_log_prob - current_sample_log_prob) * \
                                            (self.table_counts_per_doc[current_sample, d]*np.exp(current_sample_log_prob) +
                                             self.alpha*np.exp(alias.log_likelihoods.np[current_sample])) / \
                                            (self.table_counts_per_doc[new_sample, d]*np.exp(new_sample_log_prob) +
                                             self.alpha*np.exp(alias.log_likelihoods.np[new_sample]))
                                        # My earlier attempt:
                                        #acceptance = \
                                        #    (self.table_counts_per_doc[new_table_id, d] + self.alpha) / \
                                        #    (self.table_counts_per_doc[current_sample, d] + self.alpha) * \
                                        #    np.exp(new_log_prob - old_log_prob) * \
                                        #    (self.table_counts_per_doc[current_sample, d]*np.exp(old_log_prob) +
                                        #     self.alpha*alias.w.np[current_sample]) / \
                                        #    (self.table_counts_per_doc[new_table_id, d]*np.exp(new_log_prob) +
                                        #     self.alpha*alias.w.np[new_table_id])
                                        # The Java implementation, however, does this:
                                        #acceptance = \
                                        #    (self.table_counts_per_doc[new_table_id, d] + self.alpha) / \
                                        #    (self.table_counts_per_doc[current_sample, d] + self.alpha) * \
                                        #    np.exp(new_prob - old_prob) * \
                                        #    (self.table_counts_per_doc[current_sample, d]*old_log_prob +
                                        #     self.alpha*alias.w.np[current_sample]) / \
                                        #    (self.table_counts_per_doc[new_table_id, d]*new_log_prob +
                                        #     self.alpha*alias.w.np[new_table_id])
                                        # The difference is the Java impl doesn't exp the log likelihood in the last
                                        # fraction, i.e. it uses a log prob instead of a prob
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

                        # Now have a new assignment: add its counts
                        self.table_assignments[d][w] = current_sample
                        with self.table_counts.lock:
                            self.table_counts.np[current_sample] += 1
                        self.table_counts_per_doc[current_sample, d] += 1
                        self.sum_squared_table_customers[current_sample] += np.outer(x, x)

                        self.update_table_params(current_sample, cust_id)

                # Pause the alias updater until we start the next iteration
                alias_updater.pause()

                # Output some useful stats about sampling
                if stats.acceptance_used():
                    self.log.info("Acceptance rate = {:.2f}%, mean acceptance: {:.2f}".format(
                        stats.acceptance_rate()*100., stats.mean_acceptance()))
                else:
                    self.log.info("No new samples drawn")
                self.log.info("Prior select rate = {:.2f}%, mean select_pr = {:.2f}".format(
                    stats.select_pr_rate() * 100., stats.mean_select_pr()
                ))

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


class SamplingDiagnostics:
    """
    Keeps track of a few statistics about the sampling process that can be useful to
    display between iterations to give an idea of what's happening during sampling.

    """
    def __init__(self):
        self._select_pr_sum = 0.
        self._select_pr_uses = 0
        self._pr_accepted = 0
        self._acceptance_sum = 0.
        self._acceptance_uses = 0
        self._accepted_total = 0

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

    def acceptance_used(self):
        return self._acceptance_uses > 0

    def mean_acceptance(self):
        return self._acceptance_sum / self._acceptance_uses

    def acceptance_rate(self):
        return float(self._accepted_total) / self._acceptance_uses

    def mean_select_pr(self):
        return self._select_pr_sum / self._select_pr_uses

    def select_pr_rate(self):
        return float(self._pr_accepted) / self._select_pr_uses


class VoseAlias:
    """
    Implementation of Vose alias for faster sampling.

    See https://github.com/rajarshd/Gaussian_LDA/blob/master/src/util/VoseAlias.java.

    Each alias has its own lock. You should make sure you acquire the lock while either
    using the alias for sampling or updating lock parameters, to ensure that parameter
    updates and sampling using the parameters don't get interleaved.

    Following Li et al. (2014) Reducing the Sampling Complexity of Topic Models,
    we draw samples and store them, rather than storing the alias and probability tables.
    This reduces the memory requirements.

    """
    def __init__(self, n, log_likelihoods, likelihood_sum, lock, sample_queue_size=5):
        self.n = n
        # Contains log likelihoods of the word given each topic
        self.log_likelihoods = log_likelihoods
        self.likelihood_sum = likelihood_sum

        self.gen = random.Random()

        # Create a queue to put samples on, so we just store the samples and not the
        # alias and prob tables required to generate them
        self.sample_queue = IntegerQueue.create(sample_queue_size)

        # Make sure we don't mix sampling and updating
        self.lock = lock

    @staticmethod
    def create(num):
        log_likelihoods = SharedArray.create(num, "float")
        likelihood_sum = SharedArray.create(1, "float")
        lock = mp.Lock()
        return VoseAlias(num, log_likelihoods, likelihood_sum, lock)

    def __getstate__(self):
        return self.log_likelihoods, self.likelihood_sum, self.lock

    def __setstate__(self, state):
        self.__init__(*state)

    def draw_samples(self, alias, prob):
        """
        Draw samples until the sample queue is full.

        """
        # Draw N samples
        samples = [self._draw_vose_sample(alias, prob) for i in range(self.sample_queue.size)]
        # Fill the queue with these, throwing away any old ones
        self.sample_queue.put(samples)

    def _draw_vose_sample(self, alias, prob):
        # 1. Generate a fair die roll from an n-sided die
        fair_die = self.gen.randint(0, self.n-1)
        # 2. Flip a biased coin that comes up heads with probability Prob[i]
        m = self.gen.random()
        # 3. If the coin comes up "heads," return i. Otherwise, return Alias[i]
        if m > prob[fair_die]:
            return alias[fair_die]
        else:
            return fair_die

    def sample_vose(self):
        # Don't actually draw samples here, just retrieve one from the queue
        # If the queue is empty, we'll end up waiting until the background process adds something
        return self.sample_queue.get(block=True)

    def sample_numpy(self):
        """
        Draw sample without using the Vose alias, instead just using Numpy's sample method.
        This should be less efficient, presumably, but is guaranteed to sample correctly,
        so I'm using it as debugging.

        """
        lp = self.log_likelihoods.np
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
                 log_determinants, log, das_normalization=True):
        super().__init__()
        self.log = log
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
        self.log.info("Starting alias updater")
        # Before starting looping, we make sure to go once over everything
        # The main process will wait for self.initialized to be set before sampling starts
        while not self.done.is_set():
            # Iterate over the word vectors
            for term_id, x in enumerate(self.embeddings):
                # Compute the likelihood under each possible topic assignment
                log_likelihoods = self.log_multivariate_tdensity_tables(x)
                # To prevent overflow, subtract by log(p_max).
                # This is because when we will be normalizing after exponentiating, each entry will be
                # exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
                # The log p_max cancels put and prevents overflow in the exponentiating phase
                ll_max = log_likelihoods.max()
                log_likelihoods_scaled = log_likelihoods - ll_max
                # Exp them all to get the new w array
                w = np.exp(log_likelihoods_scaled)
                if self.das_normalization:
                    # Not doing this now, but following what the Java impl does, however odd that seems
                    likelihood_sum = w.sum()
                else:
                    # Sum of likelihood has to be scaled back to its original sum, before we subtracted the max log
                    likelihood_sum = np.exp(np.log(w.sum()) + ll_max)
                # Update the alias and probs using this w
                new_alias, new_prob = self.generate_table(w)
                # Update the parameters of this word's alias
                term_alias = self.aliases[term_id]
                with term_alias.lock:
                    # Fill the queue with samples using this alias and prob table
                    term_alias.draw_samples(new_alias, new_prob)
                    # Store the likelihoods and the normalization term
                    np.copyto(term_alias.log_likelihoods.np, log_likelihoods)
                    np.copyto(term_alias.likelihood_sum.np, likelihood_sum)

            # If this was the first iteration, we tell the main process that we've finished it
            if not self.initialized.is_set():
                self.log.info("Completed alias initialization")
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

    def __generate_table(self, p):
        """
        Implementation of table generation based on wikipedia description of
        algorithm, https://en.wikipedia.org/wiki/Alias_method

        The input probability distribution, p, should contain values proportional
        to the probabilities (i.e. not log probs), but does not need to be
        normalized.

        """
        # probs: the probability table, U
        # Initialize to np
        probs = (p * self.num_tables) / p.sum()
        # alias: the alias table, K
        alias = np.zeros(self.num_tables, dtype=np.int32)

        overfull = list(np.where(probs > 1.)[0])
        underfull = list(np.where(probs < 1.)[0])
        # This will generally be empty to start with
        full = list(np.where(probs == 1.)[0])

        # If there are any U=1 tables, set K=i, as recommended
        if len(full):
            alias[full] = full

        # Loop until everything is in full, or we can't choose something from both
        # Once something is neither in overfull nor underfull, it is exactly full
        while len(overfull) and len(underfull):
            # Choose an overfull entry i and underfull entry j
            i = overfull.pop(0)
            j = underfull.pop(0)
            # Allocate unused space in j to outcome i
            alias[j] = i
            # Remove allocated space from entry i
            # U_i = U_i - (1 - U_j)
            probs[i] += probs[j] - 1.
            # Entry j is now exactly full
            # Assign entry i to the appropriate category based on the new value of probs[i]
            if probs[i] < 1.:
                underfull.append(i)
            elif probs[i] > 1.:
                overfull.append(i)
            # Otherwise it is exactly full

        # Due to FP errors, overfull and underfull might not empty at the same time
        # Then we're supposed to set their prob values to 1
        if len(overfull):
            probs[overfull] = 1.
        if len(underfull):
            probs[underfull] = 1.

        return alias, probs

    def generate_table(self, w):
        ## The implementation copied from Java GaussianLDA
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
            table_solved = solve_triangular(ltriangular_chol[table], x_minus_mu[table])
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
        solved = solve_triangular(ltriangular_chol, x_minus_mu)
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
        lprobs = np.zeros(self.num_tables, dtype=np.float32)
        for i in range(self.num_tables):
            lprobs[i] = self.log_multivariate_tdensity(x, i)
        return lprobs


class IntegerQueue:
    """
    A simple queue datastructure using shared memory instead of pipes.

    It's a strange type of queue. You can get the next value or, if it's empty,
    wait until data comes in. But you can only put a full queue at a time, not a
    single value.

    """
    def __init__(self, size, values, pointer, not_empty, lock):
        self.size = size
        self._lock = lock

        self._values = values
        self._pointer = pointer
        # Indicates that the queue is not empty and can be used to wait for data from another process
        self.not_empty = not_empty

    @staticmethod
    def create(size):
        # Could allow this to be changed, but for now just use 32-bit ints
        ctype = "i"
        return IntegerQueue(
            size,
            # Allocate shared memory to store the integers
            sharedctypes.RawArray(ctype, size),
            # Allocate shared memory to store a single-value pointer to the next sample
            mp.Value("i"),
            mp.Event(), mp.Lock()
        )

    def __getstate__(self):
        """
        Arrays get pickled to be sent between processes. We ensure that the same
        shared array is used on the other side.

        """
        return self.size, self._values, self._pointer, self.not_empty, self._lock

    def __setstate__(self, state):
        self.__init__(*state)

    def get(self, block=True):
        while True:
            if not self.not_empty.is_set():
                if block:
                    # Wait until da ta comes from another process
                    self.not_empty.wait()
                else:
                    raise IntegerQueue.Empty()
            # There's something in the queue: get it
            with self._lock:
                # Check again that the pointer isn't past the end of the queue: this could happen
                # if multiple processes are getting
                if self._pointer.value >= self.size:
                    # Go round again and wait or raise
                    self.not_empty.clear()
                    continue
                val = self._values[self._pointer.value]
                # Increase the pointer
                self._pointer += 1
                if self._pointer.value >= self.size:
                    self.not_empty.clear()
                return val

    def put(self, values):
        if len(values) != self.size:
            raise ValueError("can only put a full queue of values at a time -- should be {} values, but got {}"
                             .format(self.size, len(values)))
        with self._lock:
            self._values[:] = values
            # Put the pointer back to the beginning
            self._pointer.value = 0
            # Signal to anyone waiting that there's data now
            self.not_empty.set()

    class Empty(Exception):
        pass


class GaussianLock(ExitStack):
    """
    Lock all the parameters of a gaussian, so we can either update them without them getting
    used in a partially updated state or use them without risking some getting updated.

    The arguments are SharedArrays. All of their locks will be acquired when entering
    the context manager.

    """
    def __init__(self, table_counts, table_means, table_cholesky_ltriangular_mat, log_determinants):
        super().__init__()
        self.log_determinants_lock = log_determinants.lock
        self.table_cholesky_ltriangular_mat_lock = table_cholesky_ltriangular_mat.lock
        self.table_means_lock = table_means.lock
        self.table_counts_lock = table_counts.lock

    def __enter__(self):
        obj = super().__enter__()
        obj.enter_context(obj.log_determinants_lock)
        obj.enter_context(obj.table_cholesky_ltriangular_mat_lock)
        obj.enter_context(obj.table_means_lock)
        obj.enter_context(obj.table_counts_lock)
        return obj


class SharedArray:
    """
    Small wrapper for a multiprocessing shared-memory array that will be used as the
    in-memory storage for a numpy array. This allows a numpy array to be easily shared
    between processes.

    The array has a lock that should be acquired before reading or writing the numpy
    array. The lock is not enforced automatically: you should always enclose any
    numpy operations that access the shared-memory array in:

       with arr.lock:
           # Some numpy operations on arr.np: the numpy array backed by the shared memory
           arr.np[0] = 1

    We are constrained to always using 64-bit floats, as choldate requires double types
    to operate on.

    """
    def __init__(self, array, shape, lock, dtype):
        self.array = array
        self.shape = shape
        np_array = np.frombuffer(self.array, dtype=dtype)
        np_array = np_array.reshape(self.shape)
        self.np = np_array
        # We use our own lock, not the mp.Array one, so that we can ensure all parameters are updated in a single op
        self.lock = lock

    @staticmethod
    def create(shape, dtype="float"):
        if dtype == "float":
            ctype = "d"
            np_type = np.float64
        elif dtype == "int":
            ctype = "i"
            np_type = np.int32
        else:
            raise ValueError("unknown type '{}'".format(dtype))

        lock = mp.Lock()
        # Allocate shared memory to sit behind the numpy array
        if type(shape) is int:
            size = shape
        else:
            size = functools.reduce(mul, shape)
        array = sharedctypes.RawArray(ctype, size)
        return SharedArray(array, shape, lock, np_type)

    def __getstate__(self):
        """
        Arrays get pickled to be sent between processes. We ensure that the same
        shared array is used on the other side.

        """
        return self.array, self.shape, self.lock, self.np.dtype

    def __setstate__(self, state):
        self.__init__(*state)
