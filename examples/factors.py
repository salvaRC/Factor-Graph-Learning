from numba import int32, njit

@njit(int32[:](int32[:]))
def accuracies(x):
    return x[0] * x[1:]

@njit(int32[:](int32[:]))
def propensities(x):
    return x[1:] * x[1:]

@njit(int32[:](int32[:]))
def identity(x):
    return x
