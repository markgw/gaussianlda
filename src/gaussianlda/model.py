import json
import math
import os
import pickle
import re
import warnings

import numpy as np
from sklearn.metrics import euclidean_distances

from gaussianlda.prior import Wishart
from scipy.linalg import solve_triangular
from scipy.special import gammaln
from sklearn.metrics.pairwise import cosine_similarity


class GaussianLDA:
    """
    Trained model.

    First train using the GaussianLDATrainer or GaussianLDAAliasTrainer.
    Then load using this class to get a GaussianLDA with the saved parameters for performing
    inference on new data without updating the parameters.

    """
    def __init__(self, vocab_embeddings, vocab, num_tables, alpha, kappa, table_counts, table_means, log_determinants,
                 table_cholesky_ltriangular_mat):
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
        # We can even scale the cholesky decomposition by scaleTdistrn
        self.scaled_table_cholesky_ltriangular_mat = \
            self.scaleTdistrn[:, np.newaxis, np.newaxis] * self.table_cholesky_ltriangular_mat

        self._topic_word_pdf_cache = {}

    @staticmethod
    def load_from_java(path, vocab_embeddings_path, vocab_path, alpha=None, kappa=None, nu=None, output_checks=False):
        """
        NB: This isn't fully working yet!
        I can't work out why not.

        Load a Gaussian LDA model trained and saved by the Gaussian LDA authors' original
        Java code, available at https://github.com/rajarshd/Gaussian_LDA/.
        The path given is to the directory containing all of the model files.

        Embeddings and vocab are not stored with the model, so need to be provided.
        They are given as paths to the files, stored in the same format expected by Gaussian
        LDA. This means you can use exactly the files you used to train the model.

        The files contain the output from each iteration of training. We just get the last iteration,
        or another one if explicitly requested.

        Unfortunately, the output does not store the hyperparameters alpha and kappa.
        kappa is needed to compute likelihoods under the topics and alpha is needed to perform
        topic inference, so you need to make sure these are correct to do correct inference.
        If not given, the default values from training are used.

        """
        # Load the vocab
        with open(vocab_path, "r") as f:
            vocab = [w.rstrip("\n") for w in f.readlines()]
        # Load the embeddings
        with open(vocab_embeddings_path, "r") as f:
            # Read the first line to check the dimensionality
            embedding_size = len(f.readline().split())
            # Put the embeddings into a Numpy array
            vocab_embeddings = np.zeros((len(vocab), embedding_size), dtype=np.float64)
            # We don't go back and read the first line, as it's never used by the model
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                vocab_embeddings[i, :] = [float(v) for v in line.split()]
        #non_zero = np.where(np.sum(vocab_embeddings**2., axis=-1) > 0)
        #vocab_embeddings[non_zero] /= np.sqrt(np.sum(vocab_embeddings[non_zero]**2., axis=-1))[:, np.newaxis]

        if output_checks:
            # Sanity check the embeddings and vocab
            for w, top_word in enumerate(vocab[:3]):
                nns = np.argsort(-cosine_similarity(vocab_embeddings[w].reshape(1,-1), vocab_embeddings)[0])
                print("NNs to {}: {}".format(top_word, ", ".join(vocab[nb] for nb in nns[:5])))

        filenames = os.listdir(path)
        table_params_re = re.compile("\d+\.txt")
        table_params_filenames = [f for f in filenames if re.match(table_params_re, f)]
        table_params_filenames.sort()

        num_tables = len(table_params_filenames)

        if alpha is None:
            alpha = 1. / num_tables
        if kappa is None:
            kappa = 0.1
        if nu is None:
            nu = embedding_size

        # Create empty arrays to fill
        table_cholesky_ltriangular_mat = np.zeros((num_tables, embedding_size, embedding_size), dtype=np.float64)
        table_means = np.zeros((num_tables, embedding_size), dtype=np.float64)
        for table_filename in table_params_filenames:
            # Filename is of the form k.txt
            table_num = int(table_filename[:-4])
            with open(os.path.join(path, table_filename), "r") as f:
                lines = f.readlines()
            if len(lines) % (embedding_size+1) != 0:
                warnings.warn("Gaussian LDA model does not have the right number of lines "
                              "({} lines, expected {}+1)".format(len(lines), embedding_size))
            # The last D lines are the matrix and the one before is the mean
            # The first line contains the table mean
            table_mean = [float(v) for v in lines[0].rstrip("\n").split()]
            if len(table_mean) != embedding_size:
                raise ValueError("expected {}-size mean, but got {} for table {}".format(
                    embedding_size, len(table_mean), table_num))
            table_means[table_num, :] = table_mean
            # The remaining lines are the chol decomp of the cov matrix
            chol_mat = np.array([
                [float(v) for v in line.rstrip("\n").split()]
                for line in lines[1:]
            ], dtype=np.float64)
            table_cholesky_ltriangular_mat[table_num, :, :] = chol_mat

        if output_checks:
            for topic in range(num_tables):
                topic_nn = np.argsort(-cosine_similarity(table_means[topic].reshape(1, -1), vocab_embeddings)[0])[0]
                print("Topic {} centroid by cos sim: {}".format(topic, vocab[topic_nn]))

        # Load the total counts of customers at each table
        with open(os.path.join(path, "topic_counts.txt"), "r") as f:
            lines = f.readlines()
        # Should be K lines, with a count for each table
        if len(lines) != num_tables:
            raise IOError("expected {} lines in topic_counts.txt, got {}".format(num_tables, len(lines)))
        # The last K lines give us the final counts
        table_counts = np.array([float(v.rstrip("\n")) for v in lines], dtype=np.float64)

        # Compute the log determinants from the chol decomposition of the covariance matrices
        log_determinants = np.zeros(num_tables, dtype=np.float64)
        # Compute this in the same way as the Java code does
        for table in range(num_tables):
            # Log det of cov matrix is 2*log det of chol decomp
            k_n = float(table_counts[table]) + kappa
            nu_n = float(table_counts[table]) + nu
            scale_t_distrn = (k_n + 1.) / (k_n * (nu_n - embedding_size + 1.))
            log_determinants[table] = np.sum(np.log(np.diagonal(table_cholesky_ltriangular_mat[table, :, :]))) \
                                      + embedding_size * np.log(scale_t_distrn) / 2.

        # Initialize a model
        model = GaussianLDA(
            vocab_embeddings, vocab, num_tables, alpha, kappa,
            table_counts, table_means, log_determinants, table_cholesky_ltriangular_mat,
        )
        if output_checks:
            for table in range(num_tables):
                print("Computing probs for table {}".format(table))
                logprob = model.log_multivariate_tdensity(vocab_embeddings, table)
                top_word_ids = np.argsort(-logprob)
                top_words = [vocab[i] for i in top_word_ids]
                print("Top words for table {}: {}".format(table, top_words[:3]))
        return model

    @staticmethod
    def load(path):
        # Load JSON hyperparams
        with open(os.path.join(path, "params.json"), "r") as f:
            hyperparams = json.load(f)

        # Load numpy arrays for model parameters
        arrs = {}
        for name in [
            "table_counts", "table_means", "log_determinants", "table_cholesky_ltriangular_mat", "vocab_embeddings",
        ]:
            with open(os.path.join(path, "{}.pkl".format(name)), "rb") as f:
                arrs[name] = pickle.load(f)

        # Initialize a model
        model = GaussianLDA(
            arrs["vocab_embeddings"], hyperparams["vocab"], hyperparams["num_tables"], hyperparams["alpha"],
            hyperparams["kappa"],
            arrs["table_counts"], arrs["table_means"], arrs["log_determinants"],
            arrs["table_cholesky_ltriangular_mat"],
        )
        return model

    def sample(self, doc, num_iterations, oovs_as_nones=False):
        """
        Run Gibbs sampler on a single document without updating global parameters.

        The doc is given as a list of tokens.
        Each token can be the following:

        - a string: if this is in the training vocab, it will be mapped to its ID,
           otherwise it will be treated as an unknown word (and get topic/concept None);
        - an int: represents the vocab ID of a word in the training vocabulary, for
           which the original embedding will be used;
        - a 1D number array: gives an embedding for this token explicitly, which can
           be for tokens not in the original training vocab.

        By default, any unknown words are simply removed, so topics are only returned
        for known words. This can make it difficult to match up topics with the
        input words.
        If `oovs_as_nones==True`, Nones are included in the list of topics where an input
        word was unknown.

        """
        if len(doc) == 0:
            return []

        # Check whether the doc is specified using words or word ids
        doc = [
            token if isinstance(token, np.ndarray)  # Embedding given explicitly
            or type(token) is int                   # Term ID in training vocab
            else self.vocab.index(token) if token in self.vocab  # Known word: map to ID
            else None                               # Unknown word: don't try to analyse
            for token in doc
        ]
        # Note where unknown words are, so we can indicate unknown topics/concepts in the result
        unknown_word_locs = [i for i, word in enumerate(doc) if word is None]
        # Now remove Nones from the doc and only analyse the words either in the vocab or with explicit vectors
        # Now all words are either IDs or vectors
        doc = [word for word in doc if word is not None]

        table_assignments = list(np.random.randint(self.num_tables, size=len(doc)))
        doc_table_counts = np.bincount(table_assignments, minlength=self.num_tables)

        for iteration in range(num_iterations):
            for w, cust_id_or_vec in enumerate(doc):
                # Remove custId from his old_table
                old_table_id = table_assignments[w]
                table_assignments[w] = -1  # Doesn't really make any difference, as only counts are used
                doc_table_counts[old_table_id] -= 1

                # Now calculate the prior and likelihood for the customer to sit in each table and sample
                # Go over each table
                counts = doc_table_counts[:] + self.alpha
                # Now calculate the likelihood for each table
                log_lls = self.log_multivariate_tdensity_tables(cust_id_or_vec)
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

        if oovs_as_nones:
            for idx in unknown_word_locs:
                # Put None into the lists where the input word wasn't known, so we can easily
                #  match up the topics/concepts with the input words
                table_assignments.insert(idx, None)
        return table_assignments

    def log_multivariate_tdensity(self, x, table_id):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.

        """
        if x.ndim > 1:
            logprobs = np.zeros(x.shape[0], dtype=np.float64)
            for i in range(x.shape[0]):
                logprobs[i] = self.log_multivariate_tdensity(x[i], table_id)
            return logprobs

        nu = self.nu[table_id]
        # first calculate (x-mu)
        x_minus_mu = x - self.table_means[table_id]
        ltriangular_chol = self.scaled_table_cholesky_ltriangular_mat[table_id]
        solved = solve_triangular(ltriangular_chol, x_minus_mu, check_finite=False)
        # Now take xTx (dot product)
        val = (solved ** 2.).sum(-1)

        logprob = gammaln((nu + self.embedding_size) / 2.) - \
                  (
                          gammaln(nu / 2.) +
                          self.embedding_size / 2. * (np.log(nu) + np.log(math.pi)) +
                          self.log_determinants[table_id] +
                          (nu + self.embedding_size) / 2. * np.log(1. + val / nu)
                  )
        return logprob

    def log_multivariate_tdensity_tables(self, x):
        """
        Gaussian likelihood for a table-embedding pair when using Cholesky decomposition.
        This version computes the likelihood for all tables in parallel.

        If x is an int, it is treated as an index to the vocabulary of known embeddings. This
        will often be a more efficient way to call this repeated times, since it allows us to
        caches the concept-word probs for words in the vocabulary. This is possible, since
        the concept parameters are never updated.

        """
        if type(x) is int:
            # Treat as vocab index and cache the probability
            if x not in self._topic_word_pdf_cache:
                # Not already computed this: compute now and cache the logprobs
                self._topic_word_pdf_cache[x] = self.log_multivariate_tdensity_tables(self.vocab_embeddings[x])
            return self._topic_word_pdf_cache[x]

        # Since I am storing lower triangular matrices, it is easy to calculate (x-\mu)^T\Sigma^-1(x-\mu)
        # therefore I am gonna use triangular solver first calculate (x-mu)
        x_minus_mu = x[None, :] - self.table_means
        # We can't do solve_triangular for all matrices at once in scipy
        val = np.zeros(self.num_tables, dtype=np.float64)
        for table in range(self.num_tables):
            table_solved = solve_triangular(self.scaled_table_cholesky_ltriangular_mat[table], x_minus_mu[table])
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
