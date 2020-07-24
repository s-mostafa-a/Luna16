from PIL import Image
from skimage.morphology import disk, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import SimpleITK as sitk


def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    image label: 0
    '''
    plt_number = 0
    if plot:
        f, plots = plt.subplots(12, 1, figsize=(5, 40))
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(im, cmap=plt.cm.bone)
        plt_number += 1

    '''
    Step 1: Convert into a binary image. 
    image label: 1
    '''
    binary = im < -604
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    '''
    Step 2: Remove the blobs connected to the border of the image.
    image label: 2
    '''
    cleared = clear_border(binary)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(cleared, cmap=plt.cm.bone)
        plt_number += 1
    '''
    Step 3: Label the image.
    image label: 3
    '''
    label_image = label(cleared)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(label_image, cmap=plt.cm.bone)
        plt_number += 1

    '''
    Step 4: Keep the labels with 2 largest areas and segment two lungs.
    image label: 4
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    labels = []
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
            else:
                coordinates = region.coords[0]
                labels.append(label_image[coordinates[0], coordinates[1]])
    else:
        labels = [1, 2]
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(label_image, cmap=plt.cm.bone)
        plt_number += 1
    '''
    Step 5: Fill in the small holes inside the mask of lungs which we seperate right and left lung. r and l are symbolic and they can be actually left and right!
    image labels: 5, 6
    '''
    r = label_image == labels[0]
    l = label_image == labels[1]
    r_edges = roberts(r)
    l_edges = roberts(l)
    r = ndi.binary_fill_holes(r_edges)
    l = ndi.binary_fill_holes(l_edges)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(r, cmap=plt.cm.bone)
        plt_number += 1

        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(l, cmap=plt.cm.bone)
        plt_number += 1

    '''
    Step 6: convex hull of each lung
    image labels: 7, 8
    '''
    r = convex_hull_image(r)
    l = convex_hull_image(l)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(r, cmap=plt.cm.bone)
        plt_number += 1

        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(l, cmap=plt.cm.bone)
        plt_number += 1
    '''
    Step 7: joint two separated right and left lungs.
    image label: 9
    '''
    sum_of_lr = r + l
    binary = sum_of_lr > 0
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    '''
    Step 8: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    image label: 10
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    '''
    Step 9: Superimpose the binary mask on the input image.
    image label: 11
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(im, cmap=plt.cm.bone)
        plt_number += 1

    return im


class CTScan(object):
    def __init__(self, filename=None, coords=None):
        self.filename = filename
        self.coords = coords
        path = '/Users/mostafa/Desktop/dsb_analyse/input/subset0/' + self.filename + '.mhd'
        self.ds = sitk.ReadImage(path)
        self.spacing = self.ds.GetSpacing()
        self.origin = self.ds.GetOrigin()
        self.image = sitk.GetArrayFromImage(self.ds)

    def reset_coords(self, coords):
        self.coords = coords

    def resample(self):
        image = self.image
        previous_spacing = self.spacing
        new_spacing = [1, 1, 1]
        spacing = np.array(previous_spacing, dtype=np.float32)
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        self.image = image
        self.spacing = new_spacing

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
        stretchedVoxelCoord = np.absolute(worldCoord - self.origin)
        voxelCoord = stretchedVoxelCoord / self.spacing
        return voxelCoord

    def get_ds(self):
        return self.ds

    def get_voxel_coords(self):
        voxel_coords = [self.get_world_to_voxel_coords(j) for j in range(len(self.coords))]
        return tuple(voxel_coords)

    def get_image(self):
        return self.image

    def get_subimage(self, width):
        x, y, z = self.get_voxel_coords()
        subImage = self.image[z, y - width / 2:y + width / 2, x - width / 2:x + width / 2]
        return subImage

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


# if __name__ == '__main__':
#     ct = CTScan(filename='1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410')
#     ct.transform()
#     plt.imshow(ct.image[20, :, :], cmap=plt.cm.gray)
#     plt.show()
