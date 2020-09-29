"""
Some tests for the Vose alias sampler to confirm that it's behaving as it should.

Used for debugging only.

"""
import numpy as np


def generate_table(w, num_tables):
    ## The implementation copied from Java GaussianLDA
    # Generate a new prob array and alias table
    alias = np.zeros(num_tables, dtype=np.int32)
    # Contains proportions and alias probabilities
    prob = np.zeros(num_tables, dtype=np.float32)

    # 1. Create two worklists, Small and Large
    # 2. Multiply each probability by n
    p = (w * num_tables) / w.sum()
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


if __name__ == "__main__":
    from .chol_alias import VoseAlias

    alias = VoseAlias.create(5)

    # Use a predefined set of probabilities
    known_probs = np.array([0.1, 0.1, 0.4, 0.05, 0.35])
    print("Probabilities: ", known_probs)

    # Compute the tables in the same way that the updater does
    log_likelihoods = np.log(known_probs)
    ll_max = log_likelihoods.max()
    log_likelihoods_scaled = log_likelihoods - ll_max
    # Exp them all to get the new w array
    w = np.exp(log_likelihoods_scaled)
    # Sum of likelihood has to be scaled back to its original sum, before we subtracted the max log
    # likelihood_sum = np.exp(np.log(w.sum()) + ll_max)
    # NB Not doing this now, but following what the Java impl does, however odd that seems
    likelihood_sum = w.sum()
    # Update the alias and probs using this w
    new_alias, new_prob = generate_table(w, 5)
    # Update the parameters of this word's alias
    np.copyto(alias.log_likelihoods.np, log_likelihoods)
    np.copyto(alias.alias.np, new_alias)
    np.copyto(alias.prob.np, new_prob)
    np.copyto(alias.likelihood_sum.np, likelihood_sum)

    # Draw lots of samples from the alias
    print("Drawing 1000 samples")
    sample_counts = np.zeros(5, dtype=np.float32)
    for sample_num in range(1000):
        sample_counts[alias.sample_vose()] += 1
    print("Samples:", sample_counts)
    print("Proportions:", sample_counts/sample_counts.sum())
    # These should be close to the probabilities we put in
    # If so, the sampler is working
    # For comparison, use numpy to sample
    print("Sampling using Numpy")
    sample_counts = np.zeros(5, dtype=np.float32)
    for sample_num in range(1000):
        sample_counts[np.random.choice(5, p=known_probs)] += 1
    print("Samples:", sample_counts)
    print("Proportions:", sample_counts/sample_counts.sum())

