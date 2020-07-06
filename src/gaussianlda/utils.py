import logging
import progressbar as pb

import numpy as np
from numpy.random import default_rng

from choldate import cholupdate, choldowndate


def get_logger(name):
    """
    Convenience function to make it easier to create new loggers.

    :param name: logging system logger name
    :return:
    """
    level = logging.INFO

    # Prepare a logger
    log = logging.getLogger(name)
    log.setLevel(level)

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # If coloredlogs is available, enable it
    # We don't make this one of Pimlico's core dependencies, but simply allow
    # it to be installed manually on the system and use it if it's there
    try:
        import coloredlogs
    except ImportError:
        # No coloredogs: never mind
        # Check there's a handler and add a stream handler if not
        if not log.handlers:
            # Just log to the console
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            # Put a timestamp on everything
            formatter = logging.Formatter(fmt)
            sh.setFormatter(formatter)

            log.addHandler(sh)
    else:
        coloredlogs.install(
            level=level, log=log, fmt=fmt
        )

    return log


def get_progress_bar(maxval, title=None, counter=False, show_progress=True):
    # Provides an easy way to switch to showing no progress bar
    if not show_progress:
        return lambda x: x

    widgets = []
    if title is not None:
        widgets.append("%s: " % title)
    if maxval is not pb.UnknownLength:
        widgets.extend([pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker())])
    if counter:
        widgets.extend([' (', pb.Counter(), ')'])
    if maxval is not pb.UnknownLength:
        widgets.extend([' ', pb.ETA()])
    pbar = pb.ProgressBar(widgets=widgets, maxval=maxval)
    return pbar


def chol_rank1_update(L, x):
    # choldate only uses float64
    cholupdate(L.T, x.copy())


def chol_rank1_downdate(L, x):
    choldowndate(L.T, x.copy())


def sum_logprobs(logprobs):
    """Sum probabilities represented as logprobs.

    Avoids underflow by a standard trick. Not massively efficient: I think there's a
    more efficient way to do this.

    Returns the log of the summed probabilities.

    """
    max_prob = logprobs.max()
    return np.log(np.sum(np.exp(logprobs - max_prob))) + max_prob


class BatchedRands:
    """
    Draw multiple random samples from Numpy at once to speed up repeated
    random sampling.

    Change how many are drawn at once by setting batch_size. Very small
    numbers (<10) will be very slow, like just drawing repeated samples. Numbers
    more like 100 will be better. 1000 probably gives a small speedup, but beyond
    that there's not much difference.

    """
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.rng = default_rng()
        self._it = iter(self)

    def __iter__(self):
        while True:
            batch = self.rng.random(self.batch_size)
            for v in batch:
                yield v

    def random(self):
        return next(self._it)

    def integer(self, high):
        return int(high * self.random())

    def choice(self, p):
        rand = self.random()
        return np.argmax(np.cumsum(p) > rand)

    def choice_cum(self, p):
        """
        Like choice, but probabilities are given in cumulative form. This is useful
        if you need to draw several times from the same dist, to avoid recomputing
        the cumsum. For a large number of samples, this can give a largeish speedup
        (like 2x).

        """
        rand = self.random()
        return np.argmax(p > rand)


class BatchedRandInts:
    """
    Similar to BatchedRands, but generates arrays of integers. The size of the
    arrays varies, but the range of the integers is fixed.

    """
    def __init__(self, high, batch_size=1000):
        self.high = high
        self.batch_size = batch_size
        self.rng = default_rng()
        self._new_batch()

    def _new_batch(self):
        self._current_id = 0
        self._batch = self.rng.integers(self.high, size=self.batch_size)

    def integers(self, size):
        self._current_id += size
        if self._current_id > self.batch_size:
            # Reached end of batch in middle of range
            self._new_batch()
            self._current_id += size

        return self._batch[self._current_id-size:self._current_id]
