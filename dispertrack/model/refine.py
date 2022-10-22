import numpy as np


def refine_positions(image, coords, radius, max_iterations=2, threshold=1):

    final_coords = np.empty_like(coords, dtype=np.float64)
    mass = np.empty_like(coords, dtype=np.float64)

    x = np.arange(-radius, radius+1)

    for feat, coord in enumerate(coords):
        for iteration in range(max_iterations):
            norm = np.sum(image[coord-radius:coord+radius+1])
            try:
                weight = np.sum(x*image[coord-radius:coord+radius+1], dtype=np.float64)
            except ValueError:
                raise ValueError(f'Cant calculate weight if coord is {coord} and radius is {radius}')
            cm_n = weight/norm
            cm_i = coord + cm_n

            if np.abs(cm_n) < threshold:
                break

            # Move the center to the next pixel and calculate again
            if cm_n > threshold:
                coord += 1
            elif cm_n < -threshold:
                coord -= 1
            else:
                break

            upper_bound = len(image) - 1 - radius
            if coord > upper_bound:
                coord = upper_bound

        final_coords[feat] = cm_i
        mass[feat] = norm

    return np.column_stack([final_coords, mass])