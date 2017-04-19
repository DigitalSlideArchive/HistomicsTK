import numpy


def get_principal_components(m):
    """Take a matrix m (probably 3xN) and return the 3x3 matrix of
    "principal components".  (Actually computed with SVD)

    """
    return numpy.linalg.svd(m.astype(float), full_matrices=False)[0].astype(m.dtype)


def magnitude(m):
    """Get the magnitude of each column vector in a matrix"""
    return numpy.sqrt((m ** 2).sum(0))


def normalize(m):
    """Normalize each column vector in a matrix"""
    return m / magnitude(m)
