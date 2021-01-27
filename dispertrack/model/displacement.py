import numpy as np


def msd_iter(d, lagtimes):
    """ Calculate the mean displacement and mean squared displacement given 1-D data of positions and lagtimes
    defined in frames. Proper interpretation must be given when trying to fit the data.

    Yields a tuple MD and MSD
    """
    for lt in lagtimes:
        diff = d[lt:] - d[:-lt]
        yield np.nanmean(diff), np.nanmean(diff**2)
