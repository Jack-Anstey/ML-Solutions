from random import sample
import numpy as np
import numpy.testing as npt
import time
import math


def gen_random_samples():
    """
    Generate 5 million random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size 5 million
        An array of 5 million random samples
    """
    ## TODO FILL IN
    return np.random.randn(5000000)  # use numpy


def sum_squares_for(samples):
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    timeElapse: float
        The time it took to calculate the sum of squares (in seconds)
    """
    timeElapse = 0
    ss = 0
    ## TODO FILL IN
    initialTime = time.time()  # get whatever the time is now
    for sample in samples:
        ss += math.pow(sample, 2)  # compute the sum of squares with the mean
    timeElapse = time.time() - initialTime  # do some math to get the delta
    return ss, timeElapse


def sum_squares_np(samples):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    timeElapse: float
        The time it took to calculate the sum of squares (in seconds)
    """
    timeElapse = 0
    ss = 0
    ## TODO FILL IN
    initialTime = time.time()  # get whatever the time is now
    ss = np.dot(samples, samples)  # Should be 1xN dot Nx1 = 1x1 = scalar
    timeElapse = time.time() - initialTime  # do some math to get the delta
    return ss, timeElapse


def main():
    # generate the random samples
    samples = gen_random_samples()
    # call the sum of squares
    ssFor, timeFor = sum_squares_for(samples)
    # call the numpy version
    ssNp, timeNp = sum_squares_np(samples)
    # make sure they're the same value
    npt.assert_almost_equal(ssFor, ssNp, decimal=5)
    # print out the values
    print("Time [sec] (for loop):", timeFor)
    print("Time [sec] (np loop):", timeNp)
    # Timenp is 1320.9687943262411236966389011954465124981836297892075108913617543

if __name__ == "__main__":
    main()
