"""
Multiprocessing architecture utilities, used by the alias implementation.

"""
import functools
import multiprocessing as mp
from multiprocessing import sharedctypes
from contextlib import ExitStack
from operator import mul

import numpy as np


class MultiLock(ExitStack):
    """
    Group multiple locks together to lock them all at once and release all at once.

    """
    def __init__(self, *locks):
        super().__init__()
        self.locks = locks

    def __enter__(self):
        obj = super().__enter__()
        for lock in self.locks:
            self.enter_context(lock)
        return obj


class GaussianLock(MultiLock):
    """
    Lock all the parameters of a gaussian, so we can either update them without them getting
    used in a partially updated state or use them without risking some getting updated.

    The arguments are SharedArrays. All of their locks will be acquired when entering
    the context manager.

    """
    def __init__(self, table_counts, table_means, table_cholesky_ltriangular_mat,
                 log_determinants, *args):
        if len(args):
            # Allow more arrays to be added
            args = [a.lock for a in args]

        super().__init__(
            log_determinants.lock, table_cholesky_ltriangular_mat.lock, table_means.lock, table_counts.lock,
            *args
        )


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


class TwoSidedLock:
    """
    Make sure we don't mix sampling and updating of the same word

    This is enforced with a numeric value that says which word is being written
    and which is being read (if any), with event flags to allow the opposite
    process to wait for writing/reading to complete

    The state-change lock ensures that setting the value and the flag happen
    as a single action

    """
    def __init__(self, read_index, reading_finished, write_index, writing_finished, state_change_lock):
        self.read_index = read_index
        self.reading_finished = reading_finished
        self.write_index = write_index
        self.writing_finished = writing_finished
        self.state_change_lock = state_change_lock

    @staticmethod
    def create():
        read_index = mp.Value("i")
        reading_finished = mp.Event()
        write_index = mp.Value("i")
        writing_finished = mp.Event()
        state_change_lock = mp.Lock()
        return TwoSidedLock(read_index, reading_finished, write_index, writing_finished, state_change_lock)

    def __getstate__(self):
        return self.read_index, self.reading_finished, self.write_index, self.writing_finished, self.state_change_lock

    def __setstate__(self, state):
        self.__init__(*state)

    def acquire_read(self, index):
        """ Acquire a read lock """
        # Make sure we have the state change lock before we check what's being written
        self.state_change_lock.acquire()
        # Check whether something's being written
        if self.write_index == index:
            # This alias is currently being written, so we have to wait until the writer's finished
            # Release the state-change lock first, so we don't stop the writer from changing its state
            self.state_change_lock.release()
            self.writing_finished.wait()
            # Get the lock back again so we can change our state
            self.state_change_lock.acquire()
        # Otherwise, we don't need to wait: just acquire the read lock
        # Mark that we're reading this given index, so the writer is not allowed to write to it until we've finished
        self.read_index.value = index
        # Make sure the reading-finished flag is not set, so the writer can wait for us to finish if necessary
        self.reading_finished.clear()
        # Now we're ready to let the writer check the state
        self.state_change_lock.release()

    def release_read(self):
        """ Release the current read lock """
        # Make sure the writer doesn't try to change its state while we're in the middle
        self.state_change_lock.acquire()
        # Clear the index, as we're not reading anything now
        self.read_index.value = -1
        # Set the finished flag so that, if the writer is waiting for this index to become free, it can continue
        self.reading_finished.set()
        # Release the state change lock now that we're in a consistent state
        self.state_change_lock.release()

    def acquire_write(self, index):
        """ Acquire a lock to write the given index """
        # Make sure we have the state change lock before we check what's being read
        self.state_change_lock.acquire()
        # Check whether something's being read
        if self.read_index == index:
            # This alias is currently being read, so we have to wait until the reader's finished
            # Release the state-change lock first, so we don't stop the reader from changing its state
            self.state_change_lock.release()
            self.reading_finished.wait()
            # Get the lock back again so we can change our state
            self.state_change_lock.acquire()
        # Otherwise, we don't need to wait: just acquire the write lock
        # Mark that we're writing this given index, so the reader is not allowed to read it until we've finished
        self.write_index.value = index
        # Make sure the writing-finished flag is not set, so the reader can wait for us to finish if necessary
        self.writing_finished.clear()
        # Now we're ready to let the writer check the state
        self.state_change_lock.release()

    def release_write(self):
        """ Release the current write lock """
        # Make sure the reader doesn't try to change its state while we're in the middle
        self.state_change_lock.acquire()
        # Clear the index, as we're not writing anything now
        self.write_index.value = -1
        # Set the finished flag so that, if the reader is waiting for this index to become free, it can continue
        self.writing_finished.set()
        # Release the state change lock now that we're in a consistent state
        self.state_change_lock.release()