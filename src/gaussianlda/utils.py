import logging
import progressbar as pb

import numpy as np

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


def get_progress_bar(maxval, title=None, counter=False):
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
