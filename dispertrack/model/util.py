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


def Renkin(particle_radius, channel_diameter=670E-9):
    """
    Hindrance factor (following the "Renkin equation")

    Parameters
    ----------
    particle_radius : float, np.array
        Radius of the particle (in meters)
    channel_diameter : float
        Diameter of the nano-channel (in meters)
    """

    λ = 2 * particle_radius / channel_diameter
    if np.any(λ > 1):
        raise ValueError('Particle is bigger than channel')
    return (1 - λ) ** 2 * (1 - 2.104 * λ + 2.09 * λ ** 3 - 0.95 * λ ** 5)

def DechadilokDeen(particle_radius, channel_diameter=670E-9):
    """
    Hindrance factor following Eq 15 in
        Hindrance Factors for Diffusion and Convection in Pores
        Panadda Dechadilok and William M. Deen
        Industrial & Engineering Chemistry Research 2006 45 (21), 6953-6959
        DOI: 10.1021/ie051387n

    Parameters
    ----------
    particle_radius : float, np.array
        Radius of the particle (in meters)
    channel_diameter : float
        Diameter of the nano-channel (in meters)
    """

    λ = 2 * particle_radius / channel_diameter
    if np.any(λ > 1):
        raise ValueError('Particle is bigger than channel')

    hindrance = 1 + 9/8. * λ * np.log(λ) - 1.56034 * λ \
                 + 0.528155 * λ**2 + 1.91521 * λ**3 \
                 - 2.81903 * λ**4 + 0.270788 * λ**5 \
                 + 1.10115 * λ**6 - 0.435933 * λ**7

    return hindrance
def viscosity_water(T):
    """
    Temperature dependent viscosity of water.
    The Vogel-Fulcher-Tammann equation is used with the parameters determined by Viswanath & Natarajan 1989, pp. 714–715.

    Parameters
    ----------
    T :   float, np.array
          Temperature of the liquid in Kelvin

    Returns
    -------
    eta : float, np.array
          Viscosity of water in Pascal*second
    """

    return (0.02939 * np.exp(507.88 / (T - 149.3)))/1000

def r_d(D, T=T, eta=eta):
    """
    Calculates the radius of a particle given the diffusion coefficient, temperature, and viscosity.
    If a function is passed for eta, eta(T) will be used.

    Parameters
    ----------
    D :   float, np.array
          Diffusion coefficient in m^2 s^-1
    T :   float, np.array
          Temperature of the liquid in Kelvin
    eta : float, function
          Viscosity of the liquid in Pascal*second, or a function that takes temperature T as input.

    Returns
    -------
    radius : float, np.array
             In meters
    """
    if callable(eta):
        return k_b * T / (6 * np.pi * eta(T) * D)
    else:
        return k_b * T / (6 * np.pi * eta * D)

def d_r(r, T=T, eta=eta):
    """
    Calculates the diffusion coefficient of a particle given the radius, temperature, and viscosity.
    If a function is passed for eta, eta(T) will be used.

    Parameters
    ----------
    r :   float, np.array
          Particle radius in meters
    eta : float, np.array, function
          Viscosity of the liquid in Pascal*second, or a function that takes temperature T as input.

    Returns
    -------
    D : float, np.array
        Diffusion coefficient in m^2 s^-1
    """
    if callable(eta):
        return k_b * T / (6 * np.pi * eta(T) * r)
    else:
        return k_b * T / (6 * np.pi * eta * r)