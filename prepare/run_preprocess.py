from prepare._classes import CTScan
import pandas as pd
import numpy as np
from glob import glob
import os
from configs import OUTPUT_PATH, RESOURCES_PATH
import matplotlib.pyplot as plt

annotations = pd.read_csv(RESOURCES_PATH + '/annotations.csv')
candidates = pd.read_csv(RESOURCES_PATH + '/candidates.csv')


def _get_positive_series():
    paths = glob(RESOURCES_PATH + '/*/' + "*.mhd")
    file_list = [f.split('/')[-1][:-4] for f in paths]
    series = annotations['seriesuid'].tolist()
    infected = [f for f in file_list if f in series]
    return infected[:3]


def _get_negative_series():
    paths = glob(RESOURCES_PATH + '/*/' + "*.mhd")
    file_list = [f.split('/')[-1][:-4] for f in paths]
    series = annotations['seriesuid'].tolist()
    cleans = [f for f in file_list if f not in series]
    return cleans[:3]


def save_preprocessed_data():
    [os.makedirs(d, exist_ok=True) for d in
     [f'{OUTPUT_PATH}/preprocessed/positives', f'{OUTPUT_PATH}/preprocessed/negatives']]
    meta_data = pd.DataFrame(columns=['seriesuid', 'spacing', 'bounding_box', 'centers', 'radii', 'class'])
    for series_id in _get_positive_series():
        nodule_coords_annot = annotations[annotations['seriesuid'] == series_id]
        tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_annot.iloc]
        radii = [(a['diameter_mm'] / 2) for a in nodule_coords_annot.iloc]
        ct = CTScan(seriesuid=series_id, centers=tp_co, radii=radii, clazz=1)
        ct.preprocess()
        ct.save_preprocessed_image()
        diction = ct.get_info_dict()
        meta_data = meta_data.append(pd.Series(diction), ignore_index=True)
    for series_id in _get_negative_series():
        nodule_coords_candid = candidates[candidates['seriesuid'] == series_id]
        tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_candid.iloc]
        radii = list(np.random.randint(40, size=len(tp_co)))
        max_numbers_to_use = min(len(tp_co), 3)
        tp_co = tp_co[:max_numbers_to_use]
        radii = radii[:max_numbers_to_use]
        ct = CTScan(seriesuid=series_id, centers=tp_co, radii=radii, clazz=0)
        ct.preprocess()
        ct.save_preprocessed_image()
        diction = ct.get_info_dict()
        meta_data = meta_data.append(pd.Series(diction), ignore_index=True)
    meta_data.to_csv(f'{OUTPUT_PATH}/preprocessed_meta.csv')


if __name__ == '__main__':
    save_preprocessed_data()
    plt.imshow(np.load(
        f'{OUTPUT_PATH}/preprocessed/positives/1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410.npy')[
               200, :, :], cmap=plt.cm.bone)
    plt.show()
