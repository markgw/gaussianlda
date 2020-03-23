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
import functools
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
from contextlib import ExitStack
from multiprocessing import Process, sharedctypes
from operator import mul

import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from gaussianlda.prior import Wishart
from gaussianlda.utils import get_logger, get_progress_bar, chol_rank1_downdate, chol_rank1_update


class GaussianLDAAliasTrainer:
    def __init__(self, corpus, vocab_embeddings, vocab, num_tables, alpha, kappa=0.1, log=None, save_path=None,
                 show_topics=None, mh_steps=2):
        if log is None:
            log = get_logger("GLDA")
        self.log = log
        # Vocab is used for outputting topics
        self.vocab = vocab
        self.show_topics = show_topics
        self.save_path = save_path

        # MH sampling steps
        self.mh_steps = mh_steps

        # Dirichlet hyperparam
        self.alpha = alpha

        # dataVectors
        self.vocab_embeddings = vocab_embeddings
        self.embedding_size = vocab_embeddings.shape[1]
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

        self.log.info("Initializing assignments")
        self.initialize()

        self.aliases = [
            VoseAlias.create(self.num_tables) for i in range(self.vocab_embeddings.shape[0])
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
        pbar = get_progress_bar(len(self.corpus), title="Initializing")
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
        with VoseAliasUpdater(
                self.aliases, self.vocab_embeddings,
                self.prior.kappa, self.prior.nu,
                self.table_counts, self.table_means, self.table_cholesky_ltriangular_mat,
                self.log_determinants
        ) as alias_updater:
            for iteration in range(num_iterations):
                self.log.info("Iteration {}".format(iteration))

                alias_updater.unpause()
                pbar = get_progress_bar(len(self.corpus), title="Sampling")
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

                        # Avoid recomputing the same probability multiple times
                        _table_ll = {}
                        def _get_table_ll(tab):
                            if tab not in _table_ll:
                                _table_ll[tab] = self.log_multivariate_tdensity(x, tab)
                            return _table_ll[tab]

                        # Under the alias method, we only do the full likelihood computation for topics
                        # that already have a non-zero count in the current document
                        non_zero_tables = np.where(self.table_counts_per_doc[:, d] > 0)[0]
                        if len(non_zero_tables) == 0:
                            no_non_zero = True
                        else:
                            no_non_zero = True

                            # We only compute the posterior for these topics
                            posterior = np.zeros(len(non_zero_tables), dtype=np.float32)
                            for nz_table, table in enumerate(non_zero_tables):
                                log_likelihood = _get_table_ll(table)
                                posterior[nz_table] = np.log(self.table_counts_per_doc[table, d]) + log_likelihood

                            # To prevent overflow, subtract by log(p_max)
                            posterior -= posterior.max()
                            posterior = np.exp(posterior)
                            psum = posterior.sum()
                            posterior /= psum

                        # Don't let the alias parameters get updated in the middle of the sampling
                        alias = self.aliases[cust_id]
                        with alias.lock:
                            select_pr = psum / (psum + self.alpha*alias.wsum)

                            # MHV to draw new topic
                            # Take a number of Metropolis-Hastings samples
                            current_sample = old_table_id
                            for r in range(self.mh_steps):
                                # 1. Flip a coin
                                if not no_non_zero and np.random.sample() < select_pr:
                                    # Choose from the computed posterior dist
                                    temp = np.random.choice(len(non_zero_tables), p=posterior)
                                    new_table_id = non_zero_tables[temp]
                                else:
                                    new_table_id = alias.sample_vose()

                                if new_table_id != current_sample:
                                    # 2. Find acceptance probability
                                    old_prob = _get_table_ll(current_sample)
                                    new_prob = _get_table_ll(new_table_id)
                                    # This can sometimes generate an overflow warning from Numpy
                                    # We don't care, though: in that case acceptance > 1., so we always accept
                                    with np.errstate(over="ignore"):
                                        acceptance = \
                                            (self.table_counts_per_doc[new_table_id, d] + self.alpha) / \
                                            (self.table_counts_per_doc[current_sample, d] + self.alpha) * \
                                            np.exp(new_prob - old_prob) * \
                                            (self.table_counts_per_doc[current_sample, d]*old_prob +
                                             self.alpha*alias.w.np[current_sample]) / \
                                            (self.table_counts_per_doc[new_table_id, d]*new_prob +
                                             self.alpha*alias.w.np[new_table_id])
                                    # 3. Compare against uniform[0,1]
                                    if np.random.sample() < min(1., acceptance):
                                        # This seems puzzling, but follows the Java impl exactly...
                                        current_sample = new_table_id

                        # Now have a new assignment: add its counts
                        self.table_assignments[d][w] = current_sample
                        with self.table_counts.lock:
                            self.table_counts.np[current_sample] += 1
                        self.table_counts_per_doc[current_sample, d] += 1
                        self.sum_squared_table_customers[current_sample] += np.outer(x, x)

                        self.update_table_params(current_sample, cust_id)

                # Pause the alias updater until we start the next iteration
                alias_updater.pause()

                if self.show_topics is not None:
                    print("Topics after iteration {}".format(iteration))
                    print(self.format_topics())

                if self.save_path is not None:
                    self.log.info("Saving model")
                    self.save()

    def format_topics(self, num_words=10, topics=None):
        if topics is None:
            topics = list(range(self.num_tables))

        topic_fmt = []
        for topic in topics:
            # Compute the density for all words in the vocab
            word_scores = self.log_multivariate_tdensity(self.vocab_embeddings, topic)
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


class VoseAlias:
    """
    Implementation of Vose alias for faster sampling.

    See https://github.com/rajarshd/Gaussian_LDA/blob/master/src/util/VoseAlias.java.

    Each alias has its own lock. You should make sure you acquire the lock while either
    using the alias for sampling or updating lock parameters, to ensure that parameter
    updates and sampling using the parameters don't get interleaved.

    """
    def __init__(self, alias, prob, w, lock):
        self.n = alias.shape
        # Alias indices
        self.alias = alias
        # Contains proportions and alias probabilities
        self.prob = prob
        self.w = w

        # Make sure we don't mix sampling and updating
        self.lock = lock

    @staticmethod
    def create(num):
        alias = SharedArray.create(num, "int")
        prob = SharedArray.create(num, "float")
        w = SharedArray.create(num, "float")
        lock = mp.Lock()
        return VoseAlias(alias, prob, w, lock)

    def __getstate__(self):
        return self.alias, self.prob, self.w, self.lock

    def __setstate__(self, state):
        self.__init__(*state)

    @property
    def wsum(self):
        return self.prob.np.sum()

    def sample_vose(self):
        # 1. Generate a fair die roll from an n-sided die; call the side i
        fair_die = np.random.randint(self.n)
        # 2. Flip a biased coin that comes up heads with probability Prob[i]
        m = np.random.sample()
        # 3. If the coin comes up "heads," return i. Otherwise, return Alias[i]
        if m > self.prob.np[fair_die]:
            return self.alias.np[fair_die]
        else:
            return fair_die


class VoseAliasUpdater(Process):
    """
    Following the Java implementation, a process running in the background keeps updating
    the Vose alias' tables and copying the result into the tables used by the main alias.

    """
    def __init__(self, aliases, embeddings, kappa, nu, table_counts, table_means, table_cholesky_ltriangular_mat,
                 log_determinants):
        super().__init__()
        self.aliases = aliases

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
                # To prevent overflow, subtract by log(p_max).
                # This is because when we will be normalizing after exponentiating, each entry will be
                # exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
                # The log p_max cancels put and prevents overflow in the exponentiating phase
                log_likelihoods -= log_likelihoods.max()
                # Exp them all to get the new w array
                w = np.exp(log_likelihoods)
                # Update the alias and probs using this w
                new_alias, new_prob = self.generate_table(w)
                # Update the parameters of this word's alias
                term_alias = self.aliases[term_id]
                with term_alias.lock:
                    np.copyto(term_alias.w.np, w)
                    np.copyto(term_alias.alias.np, new_alias)
                    np.copyto(term_alias.prob.np, new_prob)

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
        # Generate a new prob array and alias table
        alias = np.zeros(self.num_tables, dtype=np.int32)
        # Contains proportions and alias probabilities
        prob = np.zeros(self.num_tables, dtype=np.float32)

        # 1. Create two worklists, Small and Large
        # 2. Multiply each probability by n
        p = (w * self.num_tables) / w.sum()
        # 3. For each scaled probability pi:
        #   a. If pi<1, add i to Small.
        #   b. Otherwise(pi≥1), add i to Large.
        small = list(np.where(p < 1.)[0])
        large = list(np.where(p > 1.)[0])
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

    def log_multivariate_tdensity_tables(self, x):
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

    """
    def __init__(self, array, shape, lock, dtype):
        self.array = array
        self.shape = shape
        self.np = np.frombuffer(self.array, dtype=dtype).reshape(self.shape)
        # We use our own lock, not the mp.Array one, so that we can ensure all parameters are updated in a single op
        self.lock = lock

    @staticmethod
    def create(shape, dtype="float"):
        if dtype == "float":
            ctype = "d"
            np_type = np.float64
        elif dtype == "int":
            ctype = "l"
            np_type = np.int64
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
