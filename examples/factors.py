def accuracies(x):
    return x[0] * x[1:]


def propensities(x):
    return x[1:] * x[1:]


def identity(x):
    return x
