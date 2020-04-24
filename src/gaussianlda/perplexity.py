import numpy as np


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
