from preprocess._ct_scan import CTScan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame(
    columns=['seriesuid', 'z_index', 'y_index', 'x_index', 'radius', 'z_in_original_image', 'y_in_original_image',
             'x_in_original_image'])
annotations = pd.read_csv('/Users/mostafa/Desktop/dsb_analyse/input/annotations.csv')

test_filename = '1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410'
nodule_coords_annot = annotations[annotations['seriesuid'] == test_filename]
tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords_annot.iloc]
radii = [(a['diameter_mm'] / 2) for a in nodule_coords_annot.iloc]
ct = CTScan(filename=test_filename, coords=tp_co, radii=radii)
ct.transform()
dicts_list = ct.get_augmented_subimages_around_coords()
for i, (dikt, original_origin) in enumerate(zip(dicts_list, tp_co)):
    print(dikt['img'].shape, dikt['radius'], dikt['origin'], dikt['spacing'])
    new_file_name = f'{test_filename}_{i}'
    data = data.append(pd.Series({'seriesuid': test_filename, 'file_name': new_file_name, 'z_index': dikt['origin'][0],
                                  'y_index': dikt['origin'][1], 'x_index': dikt['origin'][2],
                                  'radius': dikt['radius'], 'z_in_original_image': original_origin[0],
                                  'y_in_original_image': original_origin[1],
                                  'x_in_original_image': original_origin[2]}), ignore_index=True)
    np.save(f'./tmp/{new_file_name}', dikt['img'])

data.to_csv('./tmp/meta.csv')
