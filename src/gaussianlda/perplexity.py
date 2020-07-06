import numpy as np
from scipy.linalg import solve_triangular


def corpus_categorical_mean_ll(corpus, table_assignments, table_word_logprobs, check_probs=True):
    """
    Compute the perplexity of a Gaussian LDA model on a given corpus, together with the
    assignment of topics to each word.

    This is computed just from the probability of the words given the assignments, not
    taking into account the priors on the assignments or the Gaussians. In this, we
    follow Util.calculateAvgLL() in the Java implementation.

    The perplexity is exp(-X).

    For comparability with other models, we treat the
    Gaussians as if they were categorical distributions over the vocabulary of known
    words in the corpus. This means that the word likelihoods sum to one over the vocabulary,
    which is not the case if you take the point PDF density from the Gaussian (as the
    original Gaussian LDA Java code does).

    This means that this perplexity is comparable, for example, with a similar value for
    standard LDA over the same corpus and the same vocabulary.

    """
    if check_probs:
        # Check that the distribution sums (roughly) to 1
        for table_logprobs in table_word_logprobs:
            max_logprob = table_logprobs.max()
            table_logprobs -= max_logprob
            table_total_scaled_prob = np.sum(np.exp(table_logprobs))
            total_prob = np.exp(np.log(table_total_scaled_prob) + max_logprob)
            # This should be roughly 1
            if abs(total_prob - 1.) > 1e-5:
                raise ValueError("Table-word distribution probs should sum to 1. Note that they're given as logprobs")

    total_logprob = 0.
    num_words = 0
    for doc, doc_tables in zip(corpus, table_assignments):
        for word_id, table_id in zip(doc, doc_tables):
            total_logprob += table_word_logprobs[table_id, word_id]
            num_words += 1

    return total_logprob / num_words


def calculate_avg_ll(corpus, table_assignments, embeddings, table_means, table_cholesky_ltriangular_mat, prior, table_counts_per_doc):
    """
    Calculates corpus perplexity (avg. log-likelihood)

    Reproduction of Gaussian_LDA's Util.calculateAvgLL(). It's not really perplexity, but
    the mean LL of the words given the sampled topics.

    """
    num_tables = table_means.shape[0]
    embedding_size = embeddings.shape[1]
    # Sum up the total number of customers per table
    n_k = table_counts_per_doc.sum(axis=1)

    scalar = n_k + prior.nu - embedding_size
    # now divide the choleskies by sqrt(scalar)
    scaled_choleskies = table_cholesky_ltriangular_mat / np.sqrt(scalar)[:, np.newaxis, np.newaxis]

    # logDensity of mulitvariate normal is given by -0.5*(log D + K*log(2*\pi)+(x-\mu)^T\Sigma^-1(x-\mu))
    # calculate log D for all table from cholesky
    log_det = np.zeros(num_tables, dtype=np.float64)
    for table in range(num_tables):
        log_det[table] = np.sum(np.log(np.diagonal(scaled_choleskies[table])))

    # Cache the table-word pairs' log densities to speed this up
    log_density_cache = {}

    total_log_ll = 0.
    total_words = 0
    for doc, tables in zip(corpus, table_assignments):
        for word, table in zip(doc, tables):
            if (table, word) in log_density_cache:
                log_density = log_density_cache[(table, word)]
            else:
                # Do exactly what the Java code does
                x = embeddings[word]
                x_minus_mu = x - table_means[table]
                ltriangular_chol = scaled_choleskies[table]
                solved = solve_triangular(ltriangular_chol, x_minus_mu, check_finite=False, lower=True)
                val = np.sum(solved ** 2.)
                log_density = 0.5 * (val + embedding_size * np.log(2. * np.pi)) + log_det[table]

                log_density_cache[(table, word)] = log_density

            total_log_ll -= log_density
            total_words += 1

    return total_log_ll / total_words
