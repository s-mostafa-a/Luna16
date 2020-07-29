from preprocess._ct_scan import CTScan, RESOURCES_PATH
import pandas as pd
import numpy as np
from glob import glob
OUTPUT_PATH = './tmp'
annotations = pd.read_csv(RESOURCES_PATH + '/annotations.csv')
candidates = pd.read_csv(RESOURCES_PATH + '/candidates.csv')


def get_positive_series():
    paths = glob(RESOURCES_PATH + '/*/' + "*.mhd")
    file_list = [f.split('/')[-1][:-4] for f in paths]
    series = annotations['seriesuid'].tolist()
    infected = [f for f in file_list if f in series]
    return infected


def get_negative_series():
    paths = glob(RESOURCES_PATH + '/*/' + "*.mhd")
    file_list = [f.split('/')[-1][:-4] for f in paths]
    series = annotations['seriesuid'].tolist()
    cleans = [f for f in file_list if f not in series]
    return cleans


def save_augmented_positive_cubes():
    data = pd.DataFrame(
        columns=['seriesuid', 'z_index', 'y_index', 'x_index', 'radius', 'z_in_original_image', 'y_in_original_image',
                 'x_in_original_image'])
    for series_id in get_positive_series():
        nodule_coords_annot = annotations[annotations['seriesuid'] == series_id]
        tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_annot.iloc]
        radii = [(a['diameter_mm'] / 2) for a in nodule_coords_annot.iloc]
        ct = CTScan(filename=series_id, coords=tp_co, radii=radii)
        ct.preprocess()
        for i, (original_z, original_y, original_x) in enumerate(tp_co):
            times_to_sample = 1
            if radii[i] > 15.:
                times_to_sample = 2
            elif radii[i] > 20.:
                times_to_sample = 6
            for j in range(times_to_sample):
                rot_id = int((j / times_to_sample) * 24 + np.random.randint(0, int(24 / times_to_sample)))
                img, radius, origin, spacing = ct.get_augmented_subimage(idx=i, rot_id=rot_id)
                new_file_name = f'{series_id}_{i}_{j}.npy'
                data = data.append(
                    pd.Series(
                        {'seriesuid': series_id, 'file_name': new_file_name, 'z_index': origin[0], 'y_index': origin[1],
                         'x_index': origin[2], 'radius': radius, 'z_in_original_image': original_z,
                         'y_in_original_image': original_y, 'x_in_original_image': original_x}), ignore_index=True)
                np.save(f'{OUTPUT_PATH}/positives/{new_file_name}', img)
        data.to_csv(f'{OUTPUT_PATH}/positive_meta.csv')


def save_augmented_negative_cubes():
    data = pd.DataFrame(
        columns=['seriesuid', 'z_index', 'y_index', 'x_index', 'radius', 'z_in_original_image', 'y_in_original_image',
                 'x_in_original_image'])
    for series_id in get_negative_series():
        nodule_coords_candid = candidates[candidates['seriesuid'] == series_id]
        tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_candid.iloc]
        radii = list(np.random.randint(40, size=len(tp_co)))
        ct = CTScan(filename=series_id, coords=tp_co, radii=radii)
        ct.preprocess()
        for i, (original_z, original_y, original_x) in enumerate(tp_co):
            times_to_sample = 1
            if radii[i] > 15.:
                times_to_sample = 2
            elif radii[i] > 20.:
                times_to_sample = 6
            for j in range(times_to_sample):
                rot_id = int((j / times_to_sample) * 24 + np.random.randint(0, int(24 / times_to_sample)))
                img, radius, origin, spacing = ct.get_augmented_subimage(idx=i, rot_id=rot_id)
                new_file_name = f'{series_id}_{i}_{j}.npy'
                data = data.append(
                    pd.Series(
                        {'seriesuid': series_id, 'file_name': new_file_name, 'z_index': origin[0], 'y_index': origin[1],
                         'x_index': origin[2], 'radius': radius, 'z_in_original_image': original_z,
                         'y_in_original_image': original_y, 'x_in_original_image': original_x}), ignore_index=True)
                np.save(f'{OUTPUT_PATH}/negatives/{new_file_name}', img)
        data.to_csv(f'{OUTPUT_PATH}/negative_meta.csv')


save_augmented_positive_cubes()
save_augmented_negative_cubes()
