from prepare._ct_scan import CTScan
import pandas as pd
import numpy as np
from glob import glob
import os
from configs import OUTPUT_PATH, RESOURCES_PATH

annotations = pd.read_csv(RESOURCES_PATH + '/annotations.csv')
candidates = pd.read_csv(RESOURCES_PATH + '/candidates.csv')


def _get_positive_series():
    paths = glob(RESOURCES_PATH + '/*/' + "*.mhd")
    file_list = [f.split('/')[-1][:-4] for f in paths]
    series = annotations['seriesuid'].tolist()
    infected = [f for f in file_list if f in series]
    return infected


def _get_negative_series():
    paths = glob(RESOURCES_PATH + '/*/' + "*.mhd")
    file_list = [f.split('/')[-1][:-4] for f in paths]
    series = annotations['seriesuid'].tolist()
    cleans = [f for f in file_list if f not in series]
    return cleans


def _save_augmented_positive_cubes(data: pd.DataFrame):
    for series_id in _get_positive_series():
        nodule_coords_annot = annotations[annotations['seriesuid'] == series_id]
        tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_annot.iloc]
        radii = [(a['diameter_mm'] / 2) for a in nodule_coords_annot.iloc]
        ct = CTScan(filename=series_id, coords=tp_co, radii=radii)
        ct.preprocess()
        for i in range(len(tp_co)):
            times_to_sample = 1
            if radii[i] > 15.:
                times_to_sample = 2
            elif radii[i] > 20.:
                times_to_sample = 6
            for j in range(times_to_sample):
                rot_id = int((j / times_to_sample) * 24 + np.random.randint(0, int(24 / times_to_sample)))
                img, radii2, centers, spacing, existing_nodules_in_patch = ct.get_augmented_subimage(idx=i,
                                                                                                    rot_id=rot_id)
                existing_radii = [radii2[i] for i in existing_nodules_in_patch]
                existing_centers = [centers[i] for i in existing_nodules_in_patch]
                centers_in_original_image = [tuple(np.array(ct.get_coords()[i]) / np.array(ct.get_image().shape)) for i
                                             in existing_nodules_in_patch]
                file_path = f'positives/{series_id}_{i}_{j}.npy'
                data = data.append(
                    pd.Series(
                        {'seriesuid': series_id, 'file_path': file_path, 'centers': existing_centers,
                         'radii': existing_radii, 'centers_in_original_image': centers_in_original_image, 'class': 1}),
                    ignore_index=True)
                np.save(f'{OUTPUT_PATH}/{file_path}', img)
    return data


def _save_augmented_negative_cubes(data: pd.DataFrame):
    needing_number_of_negatives = len(data) / 2
    all_negatives_added = 0
    for series_id in _get_negative_series():
        nodule_coords_candid = candidates[candidates['seriesuid'] == series_id]
        tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_candid.iloc]
        radii = list(np.random.randint(40, size=len(tp_co)))
        max_numbers_to_use = min(len(tp_co), 3)
        tp_co = tp_co[:max_numbers_to_use]
        radii = radii[:max_numbers_to_use]
        ct = CTScan(filename=series_id, coords=tp_co, radii=radii)
        ct.preprocess()
        for i in range(len(tp_co)):
            times_to_sample = 1
            if radii[i] > 15.:
                times_to_sample = 2
            elif radii[i] > 20.:
                times_to_sample = 6
            for j in range(times_to_sample):
                rot_id = int((j / times_to_sample) * 24 + np.random.randint(0, int(24 / times_to_sample)))
                img, radii2, centers, spacing, existing_nodules_in_patch = ct.get_augmented_subimage(idx=i,
                                                                                                    rot_id=rot_id)
                existing_radii = [radii2[i] for i in existing_nodules_in_patch]
                existing_centers = [centers[i] for i in existing_nodules_in_patch]
                centers_in_original_image = [tuple(np.array(ct.get_coords()[i]) / np.array(ct.get_image().shape)) for i
                                             in existing_nodules_in_patch]
                file_path = f'negatives/{series_id}_{i}_{j}.npy'
                data = data.append(
                    pd.Series(
                        {'seriesuid': series_id, 'file_path': file_path, 'centers': existing_centers,
                         'radii': existing_radii, 'centers_in_original_image': centers_in_original_image, 'class': 0}),
                    ignore_index=True)
                np.save(f'{OUTPUT_PATH}/{file_path}', img)
                all_negatives_added += 1
        if all_negatives_added > needing_number_of_negatives:
            break
    return data


def save_augmented_data():
    [os.makedirs(d, exist_ok=True) for d in [f'{OUTPUT_PATH}/positives', f'{OUTPUT_PATH}/negatives']]
    data = pd.DataFrame(columns=['seriesuid', 'file_path', 'centers', 'radii', 'centers_in_original_image', 'class'])
    data = _save_augmented_positive_cubes(data=data)
    data = _save_augmented_negative_cubes(data=data)
    data.to_csv(f'{OUTPUT_PATH}/meta.csv')


if __name__ == '__main__':
    save_augmented_data()
