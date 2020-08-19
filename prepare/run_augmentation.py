from prepare._classes import PatchMaker
import pandas as pd
import os
from configs import OUTPUT_PATH
from ast import literal_eval


def _get_patches(record):
    rec = record
    seriesuid = rec['seriesuid']
    spacing = literal_eval(rec['spacing'])
    lungs_bounding_box = literal_eval(rec['lungs_bounding_box'])
    centers = literal_eval(rec['centers'])
    radii = literal_eval(rec['radii'])
    clazz = int(rec['class'])
    file_directory = 'preprocessed/positives' if clazz == 1 else 'preprocessed/negatives'
    file_path = f'{OUTPUT_PATH}/{file_directory}/{seriesuid}.npy'
    pm = PatchMaker(seriesuid=seriesuid, coords=centers, radii=radii, spacing=spacing,
                    lungs_bounding_box=lungs_bounding_box,
                    file_path=file_path, clazz=clazz)
    return pm.get_augmented_patches()


def save_augmented_data(preprocess_meta):
    [os.makedirs(d, exist_ok=True) for d in
     [f'{OUTPUT_PATH}/augmented/positives', f'{OUTPUT_PATH}/augmented/negatives']]
    augmentation_meta = pd.DataFrame(columns=['seriesuid', 'sub_index', 'centers', 'lungs_bounding_box', 'radii',
                                              'class'])
    list_of_positives = []
    list_of_negatives = []
    for rec in preprocess_meta.loc[preprocess_meta['class'] == 1].iloc:
        list_of_positives += _get_patches(rec)
    for rec in preprocess_meta.loc[preprocess_meta['class'] == 0].iloc:
        list_of_negatives += _get_patches(rec)
        # 33 percent of the data will be negative samples
        if len(list_of_negatives) > len(list_of_positives) / 2:
            break
    augmentation_meta = augmentation_meta.append(list_of_positives + list_of_negatives)
    augmentation_meta.to_csv(f'{OUTPUT_PATH}/augmented_meta.csv')


if __name__ == '__main__':
    p_meta = pd.read_csv(f'{OUTPUT_PATH}/preprocessed_meta.csv', index_col=0)
    save_augmented_data(p_meta)
