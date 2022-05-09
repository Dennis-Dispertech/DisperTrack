import numpy as np
from scipy.optimize import curve_fit


def gaussian_kernel(sigma, truncate=4.0):
    " 1D discretized gaussian, taken from Trackpy "

    lw = int(truncate * sigma + 0.5)
    x = np.arange(-lw, lw+1)
    result = np.exp(x**2/(-2*sigma**2))
    return result / np.sum(result)


def gaussian(x, *p):
    """ Gaussian with an offset, useful for fitting to data """
    A, mu, sigma, off = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+off


def fitgaussian(data, p0, x = None):
    """ Convenience function to fit a gaussian to data.

    Parameters
    ----------
    data : np.array
        The data to fit

    p0 : list
        See :func:`gaussian` to understand which parameters are needed. (A, mu, sigma, offset).
    """
    if x is None:
        x = np.arange(len(data))

    coeff, var_matrix = curve_fit(gaussian, x, data, p0=p0)
    return coeff