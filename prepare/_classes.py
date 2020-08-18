import scipy.misc
import numpy as np
import SimpleITK as sitk
from prepare.utility import get_segmented_lungs, get_augmented_cube
from configs import RESOURCES_PATH, OUTPUT_PATH
from glob import glob


class CTScan(object):
    def __init__(self, seriesuid, coords, radii):
        self._seriesuid = seriesuid
        self._coords = coords
        paths = glob(f'''{RESOURCES_PATH}/*/{self._seriesuid}.mhd''')
        path = paths[0]
        self._ds = sitk.ReadImage(path)
        self._spacing = np.array(list(reversed(self._ds.GetSpacing())))
        self._origin = np.array(list(reversed(self._ds.GetOrigin())))
        self._image = sitk.GetArrayFromImage(self._ds)
        self._radii = radii
        self._mask = None

    def preprocess(self):
        self._resample()
        self._segment_lung_from_ct_scan()
        self._normalize()
        self._zero_center()
        self._change_coords()

    def get_preprocessed_info_dict(self):
        return {'seriesuid': self._seriesuid, 'radii': self._radii, 'centers': self._coords,
                'spacing': list(self._spacing)}

    def get_ds(self):
        return self._ds

    def get_image(self):
        return self._image

    def get_mask(self):
        return self._mask

    def get_coords(self):
        return self._coords

    def _resample(self):
        spacing = np.array(self._spacing, dtype=np.float32)
        new_spacing = [1, 1, 1]
        imgs = self._image
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = scipy.ndimage.interpolation.zoom(imgs, resize_factor, mode='nearest')
        self._image = imgs
        self._spacing = true_spacing

    def _segment_lung_from_ct_scan(self):
        result_img = []
        result_mask = []
        for slicee in self._image:
            rimg, rmsk = get_segmented_lungs(slicee)
            result_img.append(rimg)
            result_mask.append(rmsk)
        self._image = np.asarray(result_img)
        self._mask = np.asarray(result_mask, dtype=int)

    def _world_to_voxel(self, worldCoord):
        stretchedVoxelCoord = np.absolute(np.array(worldCoord) - np.array(self._origin))
        voxelCoord = stretchedVoxelCoord / np.array(self._spacing)
        return voxelCoord.astype(int)

    def _get_world_to_voxel_coords(self, idx):
        return tuple(self._world_to_voxel(self._coords[idx]))

    def _get_voxel_coords(self):
        voxel_coords = [self._get_world_to_voxel_coords(j) for j in range(len(self._coords))]
        return voxel_coords

    def _change_coords(self):
        new_coords = self._get_voxel_coords()
        self._coords = new_coords

    def _normalize(self):
        MIN_BOUND = -1200
        MAX_BOUND = 600.
        self._image = (self._image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        self._image[self._image > 1] = 1.
        self._image[self._image < 0] = 0.
        self._image *= 255.

    def _zero_center(self):
        PIXEL_MEAN = 0.25 * 256
        self._image = self._image - PIXEL_MEAN


class PatchMaker(object):
    def __init__(self, seriesuid: str, coords: list, radii: list, spacing: list, file_path: str, mask_path: str,
                 clazz: int):
        self._seriesuid = seriesuid
        self._coords = coords
        self._spacing = spacing
        self._radii = radii
        self._image = np.load(file=f'{file_path}')
        self._mask = np.load(file=f'{mask_path}')
        self._clazz = clazz

    def _get_augmented_patch(self, idx, rot_id=None):
        return get_augmented_cube(img=self._image, radii=self._radii, centers=self._coords,
                                  spacing=tuple(self._spacing), rot_id=rot_id, main_nodule_idx=idx)

    def get_augmented_patches(self):
        radii = self._radii
        list_of_dicts = []
        for i in range(len(self._coords)):
            times_to_sample = 1
            if radii[i] > 15.:
                times_to_sample = 2
            elif radii[i] > 20.:
                times_to_sample = 6
            for j in range(times_to_sample):
                rot_id = int((j / times_to_sample) * 24 + np.random.randint(0, int(24 / times_to_sample)))
                img, radii2, centers, spacing, existing_nodules_in_patch = self._get_augmented_patch(idx=i,
                                                                                                     rot_id=rot_id)
                existing_radii = [radii2[i] for i in existing_nodules_in_patch]
                existing_centers = [centers[i] for i in existing_nodules_in_patch]
                subdir = 'negatives' if self._clazz == 0 else 'positives'
                file_path = f'''augmented/{subdir}/{self._seriesuid}_{i}_{j}.npy'''
                list_of_dicts.append(
                    {'seriesuid': self._seriesuid, 'file_path': file_path, 'centers': existing_centers,
                     'radii': existing_radii, 'class': self._clazz})
                np.save(f'{OUTPUT_PATH}/{file_path}', img)
        return list_of_dicts
