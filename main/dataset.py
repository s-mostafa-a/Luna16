from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from copy import deepcopy
import numpy as np
from configs import BLOCK_SIZE, TARGET_SHAPE, COORDS_SHAPE, ANCHOR_SIZES, OUTPUT_PATH


class LunaDataSet(Dataset):
    def __init__(self, indices: list, meta_dataframe: pd.DataFrame):
        self.indices = indices
        self.meta_dataframe = meta_dataframe

    def __getitem__(self, idx, split=None):
        meta = self.meta_dataframe.iloc[self.indices[idx]]
        centers = literal_eval(meta['centers'])
        clazz = int(meta['class'])
        real_world_centers = literal_eval(meta['centers_in_original_image'])
        radii = literal_eval(meta['radii'])
        file_path = f'''{OUTPUT_PATH}/{meta['file_path']}'''
        patch = np.load(file_path)
        target = np.zeros(TARGET_SHAPE)
        coords = np.zeros(COORDS_SHAPE)
        if clazz == 1:
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
        else:
            for c in range(len(centers)):
                point = []
                for ax in range(len(patch.shape)):
                    window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
                    point.append(centers[c][ax] % window)

                point_for_coords = deepcopy(point)
                # prepend sth to list
                point_for_coords[:0] = [0]
                for ax in range(len(patch.shape)):
                    point_for_coords[0] = ax
                    coords[tuple(point_for_coords)] = real_world_centers[c][ax]
        out_patch = patch[np.newaxis,]
        return out_patch, target, coords

    def __len__(self):
        return len(self.indices)
