from logging import getLogger

import numpy as np
import pandas as pd


logger = getLogger()


def link(positions_ini, positions_end, radius):
    """ Link the particles in frame_ini to the particles in frame_end that are within the search radius.
    """

    links = {i: None for i in range(len(positions_ini))}

    for i, pos in enumerate(positions_ini):
        distance = np.abs(positions_end - pos)
        candidates = np.squeeze(np.argwhere(distance < radius))
        links[i] = candidates
    return links


def link_frames(frames, radius, memory=20):
    """ Link the particles in all the frames, using the specified radius.
    """

    frames['particle'] = pd.Series(np.nan*np.ones(len(frames), dtype=np.int), index=frames.index, dtype=np.int)

    curr_particle = 1
    missing = {}
    drift = 0

    for frame_num in range(frames['frame'].min(), frames['frame'].max()):
        frame_ini_array = np.array(frames.loc[frames['frame'] == frame_num, 'position'].array) + drift
        num_initial_particles = len(frame_ini_array)
        frame_end_array = frames.loc[frames['frame'] == frame_num+1, 'position'].array

        # Clean up missing:
        to_delete = []
        for pcle, info in missing.items():
            if frame_num - info['frame'] > memory:
                to_delete.append(pcle)
            else:
                info['position'] += drift
        for key in to_delete:
            del missing[key]
        to_delete = []
        # Add missing particles to the initial array
        if len(missing) > 0:
            missing_length = len(missing)
            temp_ini_array = frame_ini_array.copy()
            for info in missing.values():
                frame_ini_array = np.hstack((frame_ini_array, np.array(info['position'])))

        links = link(frame_ini_array, frame_end_array, radius=radius)

        for i, particle in enumerate(frames.loc[frames['frame'] == frame_num, 'particle']):
            if np.isnan(particle):
                frames.loc[frames.loc[frames['frame'] == frame_num, 'particle'].index[i], 'particle'] = curr_particle
                # frames.loc[frames['frame'] == frame_num, :].loc[i, ('particle',)] = curr_particle
                curr_particle += 1

        temp_missing={}
        for source, dest in links.items():
            # Check if the particle is in the frame or in the list of missing particles
            if source < num_initial_particles:
                if dest.size == 0:
                    pcle_num = int(frames.loc[frames.loc[frames['frame'] == frame_num, 'particle'].index[source],'particle'])
                    pcle_position = float(frames.loc[
                        frames.loc[frames['frame'] == frame_num,'particle'].index[source],'position'])
                    temp_missing[pcle_num] = {'position': pcle_position, 'frame': frame_num}
                    continue

                # Check one candidate or more than one, and pick the one with the most similar brightness
                if dest.size == 1:
                    frames.loc[frames.loc[frames['frame'] == frame_num+1, 'particle'].index[dest], 'particle'] \
                        = frames.loc[frames.loc[frames['frame'] == frame_num, 'particle'].index[source], 'particle']
                elif dest.size > 1:
                    masses = frames.loc[frames.loc[frames['frame'] == frame_num+1, 'particle'].index[dest], 'mass'].array
                    mass_ini = frames.loc[frames.loc[frames['frame'] == frame_num, 'particle'].index[source], 'mass']
                    mass_diff = np.abs(masses - mass_ini)
                    candidate = dest[mass_diff.argmin()]
                    frames.loc[frames.loc[frames['frame'] == frame_num + 1,'particle'].index[candidate], 'particle'] \
                        = frames.loc[frames.loc[frames['frame'] == frame_num,'particle'].index[source], 'particle']
            else:
                pcles = list(missing.keys())
                try:
                    pcle_num = pcles[source-num_initial_particles]
                except IndexError:
                    print(f'Frame: {frame_num}')
                    print(20*'=')
                    print(pcles, source, num_initial_particles, missing_length)
                    print(temp_ini_array)
                    print(missing)
                    print(frame_ini_array)
                    continue

                if dest.size > 1:
                    print('More than one possible dest in memory!')
                    print(f'Dest: {dest}')

                if dest.size == 1:
                    frames.loc[frames.loc[frames['frame'] == frame_num + 1,'particle'].index[dest], 'particle'] = pcle_num
                    to_delete.append(pcle_num)

        for d in to_delete:
            del missing[d]

        missing.update(temp_missing)
        # Calculate average drift
        i = 0
        d = 0
        for p in (frames[frames['frame'] == frame_num]['particle']):
            if p in set(frames[frames['frame'] == frame_num+1]['particle']):
                p1 = frames.loc[(frames['frame'] == frame_num+1) & (frames['particle'] == p), 'position'].values
                p2 = frames.loc[(frames['frame'] == frame_num) & (frames['particle'] == p), 'position'].values
                d += float(p1-p2)
                i += 1
        if i > 0:
            d = d/i
        drift = d

        # Find links to the same target particle
        # duplicated_links = {}
        # for key, value in links.items():
        #     if value.size == 0 or key is None:
        #         continue
        #     duplicated_links.setdefault(int(value), set()).add(key)
        #
        # duplicated = [(key, values) for key, values in duplicated_links.items() if len(values) > 1]
        #
        # if len(duplicated) >= 1:
        #     print(f'Duplicated particles in frame {frame_num}: {duplicated}')

        if frame_num % 50 == 0:
            print(f'Frame: {frame_num} out of {frames["frame"].max()}\nParticles so far: {curr_particle}', end='\r')
    return frames


