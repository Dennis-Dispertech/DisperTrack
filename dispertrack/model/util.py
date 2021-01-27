import numpy as np


def gaussian_kernel(sigma, truncate=4.0):
    "1D discretized gaussian, taken from Trackpy"

    lw = int(truncate * sigma + 0.5)
    x = np.arange(-lw, lw+1)
    result = np.exp(x**2/(-2*sigma**2))
    return result / np.sum(result)
