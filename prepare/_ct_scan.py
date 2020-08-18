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
        return {'seriesuid': self._seriesuid, 'radii': self._radii, 'centers': self._coords, 'spacing': self._spacing}

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
    def __init__(self, seriesuid, coords, radii, spacing, file_path, mask_path, clazz):
        self._seriesuid = seriesuid
        self._coords = coords
        self._spacing = spacing
        self._radii = radii
        self._image = np.load(file=f'{OUTPUT_PATH}{file_path}')
        self._mask = np.load(file=f'{OUTPUT_PATH}{mask_path}')
        self._clazz = clazz

    def get_augmented_subimage(self, idx, rot_id=None):
        return get_augmented_cube(img=self._image, radii=self._radii, centers=self._coords,
                                  spacing=tuple(self._spacing), rot_id=rot_id, main_nodule_idx=idx)
