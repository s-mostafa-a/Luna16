from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
import numpy as np
import math
from configs import BLOCK_SIZE, TARGET_SHAPE, ANCHOR_SIZES, OUTPUT_PATH, PADDING_FOR_LOCALIZATION, COORDS_CUBE_SIZE


class LunaDataSet(Dataset):
    def __init__(self, indices: list, meta_dataframe: pd.DataFrame):
        self.indices = indices
        self.meta_dataframe = meta_dataframe

    def __getitem__(self, idx, split=None):
        meta = self.meta_dataframe.iloc[self.indices[idx]]
        centers = literal_eval(meta['centers'])
        radii = literal_eval(meta['radii'])
        lungs_bounding_box = literal_eval(meta['lungs_bounding_box'])
        clazz = int(meta['class'])
        sub_dir = 'positives' if clazz == 1 else 'negatives'
        file_path = f'''{OUTPUT_PATH}/augmented/{sub_dir}/{meta['seriesuid']}_{meta['sub_index']}.npy'''
        patch = np.load(file_path)
        target = np.zeros(TARGET_SHAPE)
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

        out_patch = patch[np.newaxis, ]

        coords = self._get_coords(lungs_bounding_box)
        return out_patch, target, coords

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _get_coords(bb):
        div_factor = BLOCK_SIZE / COORDS_CUBE_SIZE
        coords = np.ones((3, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE)) * PADDING_FOR_LOCALIZATION

        bb_new = [[], []]
        for i in (0, 1, 2):
            if bb[0][i] < bb[1][i]:
                bb_new[0].append(math.floor(bb[0][i] / div_factor))
                bb_new[1].append(math.ceil(bb[1][i] / div_factor))
            else:
                bb_new[0].append(math.ceil(bb[0][i] / div_factor))
                bb_new[1].append(math.floor(bb[1][i] / div_factor))

        np_bb0 = np.array(bb_new[0], dtype=int)
        np_bb1 = np.array(bb_new[1], dtype=int)
        distances = np.abs(np_bb0 - np_bb1)
        starts = np.minimum(np_bb0, np_bb1)
        ends = np.maximum(np_bb0, np_bb1)

        if (starts > np.array([32, 32, 32])).any() or (ends < np.array([0, 0, 0])).any():
            return coords
        else:
            for i in (0, 1, 2):
                shp = [1, 1, 1]
                shp[i] = -1
                vec = np.arange(-1 * math.ceil(distances[i] / 2), math.floor(distances[i] / 2)).reshape(
                    tuple(shp)) / math.ceil(
                    distances[i] / 2)
                if bb_new[0][i] > bb_new[1][i]:
                    vec = vec * -1
                matrix = np.broadcast_to(vec, tuple(distances))
                a1 = np.maximum(0, starts)
                b1 = np.minimum(ends, COORDS_CUBE_SIZE)
                a2 = np.maximum(-1 * starts, 0)
                b2 = np.minimum(ends, COORDS_CUBE_SIZE) - starts
                coords[i, a1[0]:b1[0], a1[1]:b1[1], a1[2]:b1[2]] = matrix[a2[0]:b2[0], a2[1]:b2[1], a2[2]:b2[2]]
            return coords
