from preprocess._ct_scan import CTScan
import pandas as pd
import numpy as np

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
for i, (original_z, original_y, original_x) in enumerate(tp_co):
    times_to_sample = 1
    if radii[i] > 15.:
        times_to_sample = 2
    elif radii[i] > 20.:
        times_to_sample = 6
    for j in range(times_to_sample):
        rot_id = int((j / times_to_sample) * 24 + np.random.randint(0, int(24 / times_to_sample)))
        img, radius, origin, spacing = ct.get_augmented_subimage(idx=i, rot_id=rot_id)
        new_file_name = f'{test_filename}_{i}{j}'
        data = data.append(
            pd.Series(
                {'seriesuid': test_filename, 'file_name': new_file_name, 'z_index': origin[0], 'y_index': origin[1],
                 'x_index': origin[2], 'radius': radius, 'z_in_original_image': original_z,
                 'y_in_original_image': original_y, 'x_in_original_image': original_x}), ignore_index=True)
        np.save(f'./tmp/{new_file_name}', img)

data.to_csv('./tmp/meta.csv')
