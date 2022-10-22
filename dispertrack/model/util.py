import numpy as np
from scipy.optimize import curve_fit

k_b = 1.38E-23
T = 300
eta = 0.0009532

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

def H(r, c=670E-9):
    """
    Hindrance factor

    Parameters
    ----------
    r : float, np.array
        Radius of the particle (in meters)
    c : float
        Diameter of the nano-channel (in meters)
    """

    λ = 2*r/c
    if np.any(λ > 1):
        raise ValueError('Particle is bigger than channel')
    return (1 - λ) ** 2 * (1 - 2.104 * λ + 2.09 * λ ** 3 - 0.95 * λ ** 5)


def r_d(D, T=T, eta=eta):
    """Calculates the radius of a particle given the diffusion coefficient, temperature, and viscosity.

    Returns
    -------
    radius : float
        In meters
    """
    return k_b*T/(6*np.pi*eta*D)

def d_r(r, T=T, eta=eta):
    return k_b*T/(6*np.pi*eta*r)