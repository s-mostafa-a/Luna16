import pandas as pd
import numpy as np
from ast import literal_eval

BLOCK_SIZE = 128
TARGET_SHAPE = (32, 32, 32, 3, 5)
ANCHOR_SIZES = [10, 30, 60]
from preprocess.run import OUTPUT_PATH

positives = pd.read_csv(f'{OUTPUT_PATH}/positive_meta.csv', index_col=0)
meta = positives.iloc[0]
centers = literal_eval(meta['centers'])
radii = literal_eval(meta['radii'])
file_path = f'''{OUTPUT_PATH}/positives/{meta['file_name']}'''
patch = np.load(file_path)
target = np.zeros(TARGET_SHAPE)
for c in range(len(centers)):
    place = []
    point = []
    windows = []
    for ax in range(len(patch.shape)):
        window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
        windows.append(window)
        place.append(centers[c][ax] // window)
        point.append(centers[c][ax] % window)
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
    print(place)
    print(vector)
print('----------------------------')
print(target[(1, 1, 0, 0)])
print(target[(11, 22, 4, 1)])
print(target[(11, 22, 4, 0)])
