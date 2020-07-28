from PIL import Image
import scipy.misc
import numpy as np
import SimpleITK as sitk
from preprocess.utility import get_segmented_lungs, get_augmented_cube


class CTScan(object):
    def __init__(self, filename=None, coords=None):
        self.filename = filename
        self.coords = coords
        path = '/Users/mostafa/Desktop/dsb_analyse/input/subset0/' + self.filename + '.mhd'
        self.ds = sitk.ReadImage(path)
        self.spacing = np.array(list(reversed(self.ds.GetSpacing())))
        self.origin = np.array(list(reversed(self.ds.GetOrigin())))
        self.image = sitk.GetArrayFromImage(self.ds)

    def reset_coords(self, coords):
        self.coords = coords

    def resample(self):
        spacing = np.array(self.spacing, dtype=np.float32)
        new_spacing = [1, 1, 1]
        imgs = self.image
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = scipy.ndimage.interpolation.zoom(imgs, resize_factor, mode='nearest')
        self.image = imgs
        self.spacing = true_spacing

    def segment_lung_from_ct_scan(self):
        self.image = np.asarray([get_segmented_lungs(slicee) for slicee in self.image])

    def transform(self):
        self.resample()
        self.segment_lung_from_ct_scan()
        self.normalize()
        self.zero_center()

    def get_world_to_voxel_coords(self, idx):
        return self.world_to_voxel(self.coords[idx])

    def world_to_voxel(self, worldCoord):
        stretchedVoxelCoord = np.absolute(np.array(worldCoord) - np.array(self.origin))
        voxelCoord = stretchedVoxelCoord / np.array(self.spacing)
        return voxelCoord.astype(int)

    def get_ds(self):
        return self.ds

    def get_voxel_coords(self):
        voxel_coords = [self.get_world_to_voxel_coords(j) for j in range(len(self.coords))]
        return tuple(voxel_coords)

    def get_image(self):
        return self.image

    def get_augmented_subimages_around_coords(self):
        list_of_images_dict = []
        for i, (z, y, x) in enumerate(self.get_voxel_coords()):
            img, radius, origin, spacing = get_augmented_cube(self.image, 5, (z, y, x), tuple(self.spacing))
            list_of_images_dict.append({'img': img, 'radius': radius, 'origin': origin, 'spacing': spacing})
        return list_of_images_dict

    def get_subimages(self):
        sub_images = []
        shape = self.image.shape
        for i, (z, y, x) in enumerate(self.get_voxel_coords()):
            width_candidates = []
            width_candidates.append(abs(shape[1]-y))
            width_candidates.append(y)
            width_candidates.append(abs(shape[2]-x))
            width_candidates.append(x)
            width = int(np.min(np.array(width_candidates)))
            sub_image = self.image[int(z), int(y - width / 2):int(y + width / 2), int(x - width / 2):int(x + width / 2)]
            sub_images.append(sub_image)
        return sub_images

    def normalizePlanes(self, npzarray):
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray > 1] = 1.
        npzarray[npzarray < 0] = 0.
        return npzarray

    def normalize(self):
        MIN_BOUND = -1200
        MAX_BOUND = 600.
        self.image = (self.image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        self.image[self.image > 1] = 1.
        self.image[self.image < 0] = 0.
        self.image *= 255.

    def zero_center(self):
        PIXEL_MEAN = 0.25 * 256
        self.image = self.image - PIXEL_MEAN

    def save_image(self, filename, width):
        image = self.get_subimage(width)
        image = self.normalizePlanes(image)
        Image.fromarray(image * 255).convert('L').save(filename)
