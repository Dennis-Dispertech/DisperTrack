import numpy as np


def link(frame_ini, frame_end, radius):
    positions_ini = frame_ini['position'].array
    positions_end = frame_end['position'].array

    links = {i: None for i in range(len(positions_end))}

    for i, pos in enumerate(positions_end):
        distance = np.abs(positions_ini - pos)
        candidates = np.argwhere(distance < radius)
        if candidates.size == 0:
            print('No candidates')
            continue
        if len(candidates) > 1:
            print('More than one candidate! Let\'s pick the closest in brightness')
            mass = np.squeeze(frame_ini['mass'].array)[candidates]
            mass_diff = np.abs(mass - frame_end['mass'][i])
            candidate = mass_diff.argmin()
        else:
            candidate = candidates[0][0]
        links[i] = candidate
    return links