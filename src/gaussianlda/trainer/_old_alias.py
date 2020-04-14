"""
Old implementation of the Vose alias, where each word had its own separate
data structure. This was less memory (and probably time) efficient, but I'm
leaving the old implementation here for reference, as it may be easier to
understand or help understanding what the new implementation is doing.

"""
import random
import multiprocessing as mp
import numpy as np
from gaussianlda.mp_utils import SharedArray


class VoseAlias:
    """
    Implementation of Vose alias for faster sampling.

    See https://github.com/rajarshd/Gaussian_LDA/blob/master/src/util/VoseAlias.java.

    Each alias has its own lock. You should make sure you acquire the lock while either
    using the alias for sampling or updating lock parameters, to ensure that parameter
    updates and sampling using the parameters don't get interleaved.

    """
    def __init__(self, alias, prob, log_likelihoods, likelihood_sum, lock):
        self.n = alias.shape
        # Alias indices
        self.alias = alias
        # Contains proportions and alias probabilities
        self.prob = prob
        # Contains log likelihoods of the word given each topic
        self.log_likelihoods = log_likelihoods
        self.likelihood_sum = likelihood_sum

        self.gen = random.Random()

        # Make sure we don't mix sampling and updating
        self.lock = lock

    @staticmethod
    def create(num):
        alias = SharedArray.create(num, "int")
        prob = SharedArray.create(num, "float")
        log_likelihoods = SharedArray.create(num, "float")
        likelihood_sum = SharedArray.create(1, "float")
        lock = mp.Lock()
        return VoseAlias(alias, prob, log_likelihoods, likelihood_sum, lock)

    def __getstate__(self):
        return self.alias, self.prob, self.log_likelihoods, self.likelihood_sum, self.lock

    def __setstate__(self, state):
        self.__init__(*state)

    def sample_vose(self):
        # 1. Generate a fair die roll from an n-sided die
        fair_die = self.gen.randint(0, self.n-1)
        # 2. Flip a biased coin that comes up heads with probability Prob[i]
        m = self.gen.random()
        # 3. If the coin comes up "heads," return i. Otherwise, return Alias[i]
        if m > self.prob.np[fair_die]:
            return self.alias.np[fair_die]
        else:
            return fair_die

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
