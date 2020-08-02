from copy import deepcopy

import pandas as pd
import numpy as np
from ast import literal_eval
from preprocess.run import OUTPUT_PATH

BLOCK_SIZE = 128
TARGET_SHAPE = (32, 32, 32, 3, 5)
ANCHOR_SIZES = [10, 30, 60]

positives = pd.read_csv(f'{OUTPUT_PATH}/positive_meta.csv', index_col=0)
negatives = pd.read_csv(f'{OUTPUT_PATH}/negative_meta.csv', index_col=0)

neg_grp = negatives.groupby(['seriesuid']).indices
print(len(neg_grp.keys()))
for series in neg_grp:
    print(series)
    print(neg_grp[series])
    print()
exit(0)
for i in range(len(positives)):
    meta = positives.iloc[i]
    centers = literal_eval(meta['centers'])
    real_world_centers = literal_eval(meta['centers_in_original_image'])
    radii = literal_eval(meta['radii'])
    file_path = f'''{OUTPUT_PATH}/positives/{meta['file_name']}'''
    patch = np.load(file_path)
    target = np.zeros(TARGET_SHAPE)
    coords = np.zeros(TARGET_SHAPE)
    for c in range(len(centers)):
        place = []
        point = []
        windows = []
        for ax in range(len(patch.shape)):
            window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
            windows.append(window)
            place.append(centers[c][ax] // window)
            point.append(centers[c][ax] % window)

        point_for_coords = deepcopy(point)
        # prepend sth to list
        point_for_coords[:0] = [0]
        for ax in range(len(patch.shape)):
            point_for_coords[0] = ax
            coords[tuple(point_for_coords)] = real_world_centers[c][ax]
        if radii[c] <= ANCHOR_SIZES[0] / 2:
            place.append(0)
        elif radii[c] <= ANCHOR_SIZES[1] / 2:
            place.append(1)
        else:
            place.append(2)
        vector = [1]
        for p in range(len(point)):
            vector.append(point[p] / windows[p] - 1)
        vector.append(radii[c])
        target[tuple(place)] = vector
