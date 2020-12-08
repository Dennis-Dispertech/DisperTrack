from logging import getLogger

import numpy as np
import pandas as pd


logger = getLogger()

def link(frame_ini, frame_end, radius):
    """ Link the particles in frame_ini to the particles in frame_end that are within the search radius.
    """
    positions_ini = frame_ini['position'].array
    positions_end = frame_end['position'].array

    links = {i: None for i in range(len(positions_ini))}

    for i, pos in enumerate(positions_ini):
        distance = np.abs(positions_end - pos)
        candidates = np.argwhere(distance < radius)
        if candidates.size == 0:
            logger.info('No particles found')
            continue
        if len(candidates) > 1:
            logger.info('More than one candidate! Let\'s pick the closest in brightness')
            mass = np.squeeze(frame_end['mass'].array)[candidates]
            mass_diff = np.abs(mass - frame_ini.mass.array[i])
            candidate = mass_diff.argmin()
        else:
            candidate = candidates[0][0]
        links[i] = candidate
    return links


def link_frames(frames, radius):
    """ Link the particles in all the frames, using the specified radius.
    """

    frames['particle'] = pd.Series(np.nan*np.ones(len(frames), dtype=np.int), index=frames.index)

    curr_particle = 1
    for frame_num in range(frames['frame'].min(), frames['frame'].max()):
        links = link(frames[frames['frame'] == frame_num], frames[frames['frame'] == frame_num+1], radius=radius)

        for i, particle in enumerate(frames.loc[frames['frame'] == frame_num, 'particle']):
            if np.isnan(particle):
                frames.loc[frames.loc[frames['frame'] == frame_num, 'particle'].index[i], 'particle'] = curr_particle
                # frames.loc[frames['frame'] == frame_num, :].loc[i, ('particle',)] = curr_particle
                curr_particle += 1

        for source, dest in links.items():
            if dest is None:
                continue
            frames.loc[frames.loc[frames['frame'] == frame_num+1, 'particle'].index[dest], 'particle'] = frames.loc[frames.loc[frames['frame'] == frame_num, 'particle'].index[source], 'particle']
        # Find links to the same target particle
        duplicated_links = {}
        for key, value in links.items():
            duplicated_links.setdefault(value, set()).add(key)

        duplicated = [(key, values) for key, values in duplicated_links.items() if len(values) > 1]
        if len(duplicated) >= 1:
            print(f'Duplicated particles in frame {frame_num}: {duplicated}')
        if frame_num % 50 == 0:
            print(f'Frame: {frame_num} out of {frames["frame"].max()}', end='\r')
    return frames


