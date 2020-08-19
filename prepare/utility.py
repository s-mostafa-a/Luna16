from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import scipy
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image, disk, binary_closing
from skimage.segmentation import clear_border


def argmax_3d(img: np.array):
    max1 = np.max(img, axis=0)
    argmax1 = np.argmax(img, axis=0)
    max2 = np.max(max1, axis=0)
    argmax2 = np.argmax(max1, axis=0)
    argmax3 = np.argmax(max2, axis=0)
    argmax_3d = (argmax1[argmax2[argmax3], argmax3], argmax2[argmax3], argmax3)
    return argmax_3d, img[argmax_3d]


def _get_cube_from_img_new(img, origin: tuple, block_size=128, pad_value=106.):
    assert 2 <= len(origin) <= 3
    final_image_shape = tuple([block_size] * len(origin))
    result = np.ones(final_image_shape) * pad_value
    start_at_original_images = []
    end_at_original_images = []
    start_at_result_images = []
    end_at_result_images = []
    for i, center_of_a_dim in enumerate(origin):
        start_at_original_image = int(center_of_a_dim - block_size / 2)
        end_at_original_image = start_at_original_image + block_size
        if start_at_original_image < 0:
            start_at_result_image = abs(start_at_original_image)
            start_at_original_image = 0
        else:
            start_at_result_image = 0
        if end_at_original_image > img.shape[i]:
            end_at_original_image = img.shape[i]
            end_at_result_image = start_at_result_image + (end_at_original_image - start_at_original_image)
        else:
            end_at_result_image = block_size
        start_at_original_images.append(start_at_original_image)
        end_at_original_images.append(end_at_original_image)
        start_at_result_images.append(start_at_result_image)
        end_at_result_images.append(end_at_result_image)
    # for simplicity
    sri = start_at_result_images
    eri = end_at_result_images
    soi = start_at_original_images
    eoi = end_at_original_images
    if len(origin) == 3:
        result[sri[0]:eri[0], sri[1]:eri[1], sri[2]:eri[2]] = img[soi[0]:eoi[0], soi[1]:eoi[1], soi[2]:eoi[2]]
    elif len(origin) == 2:
        result[sri[0]:eri[0], sri[1]:eri[1]] = img[soi[0]:eoi[0], soi[1]:eoi[1]]

    return result


def random_crop(img: np.array, centers: list, lungs_bounding_box: list, radii: list, main_nodule_idx: int,
                spacing: tuple,
                block_size: int,
                pad_value: float, margin: int):
    max_radius_index = np.max(np.round(radii[main_nodule_idx] / np.array(spacing)).astype(int))
    center_of_cube = list(centers[main_nodule_idx])
    shifts = []
    for i in range(len(centers[main_nodule_idx])):
        high = int(block_size / 2) - max_radius_index - margin
        if high < 0:
            print('negative high!!!')
            high = 0
        shift = np.random.randint(low=-abs(high), high=abs(high))
        center_of_cube[i] += shift
        shifts.append(shift)
    out_img = _get_cube_from_img_new(img, origin=tuple(center_of_cube), block_size=block_size, pad_value=pad_value)
    out_centers = []
    out_lungs_bounding_box = []
    for i in range(len(centers)):
        diff = np.array(centers[main_nodule_idx]) - np.array(centers[i])
        out_centers.append(
            tuple(np.array([int(block_size / 2)] * len(centers[i]), dtype=int) - np.array(shifts, dtype=int) - diff))
    for i in range(len(lungs_bounding_box)):
        diff = np.array(centers[main_nodule_idx]) - np.array(lungs_bounding_box[i])
        out_lungs_bounding_box.append(tuple(
            np.array([int(block_size / 2)] * len(lungs_bounding_box[i]), dtype=int) - np.array(shifts,
                                                                                               dtype=int) - diff))

    return out_img, out_centers, out_lungs_bounding_box


def _get_point_after_2d_rotation(in_points: list, shape: tuple, rot90s: int, flip: bool = False):
    assert len(in_points[0]) == 2 and len(shape) == 2
    rot90s = rot90s % 4
    result_points = []
    for in_point in in_points:
        result_point = list(in_point)
        for i in range(rot90s):
            previous = result_point.copy()
            axes = [0, 1]
            point_complement = (shape[0] - previous[0], shape[1] - previous[1])
            result_point[axes[0]] = point_complement[axes[1]]
            result_point[axes[1]] = previous[axes[0]]
        if flip:
            result_point[0] = shape[0] - result_point[0]
        result_points.append(tuple(result_point))
    return result_points


def _get_point_after_3d_rotation(in_points: list, shape: tuple, axes, rot90s: int, flip: bool = False):
    rot90s = rot90s % 4
    result_points = []
    for in_point in in_points:
        result_point = list(in_point)
        other_axis = [item for item in [0, 1, 2] if item not in axes]
        for i in range(rot90s):
            previous = result_point.copy()
            point_complement = np.array(shape, dtype=int) - np.array(previous, dtype=int)
            result_point[axes[0]] = point_complement[axes[1]]
            result_point[axes[1]] = previous[axes[0]]
        if flip:
            result_point[other_axis[0]] = shape[other_axis[0]] - result_point[other_axis[0]]
        result_points.append(tuple(result_point))
    return result_points


def rotate(img: np.array, spacing: tuple, centers: list, lungs_bounding_box: list, rotate_id: int):
    spacing = list(spacing)
    dimensions = len(img.shape)
    assert (dimensions == 3 and rotate_id < 24) or (dimensions == 2 and rotate_id < 8)
    other_axes = [i for i in range(dimensions)]

    if dimensions == 2:
        axis = [0]
        out_points = partial(_get_point_after_2d_rotation, shape=tuple(img.shape))
    else:  # dimensions == 3
        axis = rotate_id // 8
        other_axes.pop(axis)
        out_points = partial(_get_point_after_3d_rotation, shape=tuple(img.shape), axes=other_axes)

    which_rotation = rotate_id % 8
    flip = which_rotation >= 4
    rotation_times = (which_rotation % 4)

    spacing_exchanged = (which_rotation % 2) != 0
    if spacing_exchanged:
        if dimensions == 3:
            tmp = spacing[other_axes[0]]
            spacing[other_axes[0]] = spacing[other_axes[1]]
            spacing[other_axes[1]] = tmp
        elif dimensions == 2:
            tmp = spacing[0]
            spacing[0] = spacing[1]
            spacing[1] = tmp
    img = np.rot90(img, k=rotation_times, axes=other_axes)
    if flip:
        img = np.flip(img, axis=axis)
    return img, tuple(spacing), out_points(in_points=centers, rot90s=rotation_times, flip=flip), out_points(
        in_points=lungs_bounding_box, rot90s=rotation_times, flip=flip)


def scale(img: np.array, scale_factor: float, spacing: tuple, centers: list, lungs_bounding_box: list, radii: list):
    assert (.75 <= scale_factor <= 1.25)
    out_centers = [tuple(np.rint(np.array(c) * scale_factor).astype(int)) for c in centers]
    out_lungs_bounding_box = [tuple(np.rint(np.array(b) * scale_factor).astype(int)) for b in lungs_bounding_box]
    out_radii = [r * scale_factor for r in radii]
    spacing = np.array(spacing) * scale_factor
    img1 = scipy.ndimage.interpolation.zoom(img, spacing, mode='nearest')
    return img1, tuple(spacing), out_centers, out_lungs_bounding_box, out_radii


def get_augmented_cube(img: np.array, radii: list, centers: list, main_nodule_idx: int, spacing: tuple,
                       lungs_bounding_box: list, block_size=128, pad_value=106, margin=10, rot_id=None):
    scale_factor = np.random.random() / 2 + .75
    rotate_id = np.random.randint(0, 24) if not rot_id else rot_id
    img1, spacing1, centers1, lungs_bounding_box1, radii1 = scale(img, scale_factor=scale_factor, spacing=spacing,
                                                                  centers=centers,
                                                                  lungs_bounding_box=lungs_bounding_box, radii=radii)
    img2, centers2, lungs_bounding_box2 = random_crop(img=img1, centers=centers1,
                                                      lungs_bounding_box=lungs_bounding_box1, radii=radii1,
                                                      main_nodule_idx=main_nodule_idx, spacing=spacing1,
                                                      block_size=block_size, pad_value=pad_value, margin=margin)
    existing_centers_in_patch = []
    for i in range(len(centers2)):
        dont_count = False
        for ax in centers2[i]:
            if not (0 <= ax <= block_size):
                dont_count = True
                break
        if not dont_count:
            existing_centers_in_patch.append(i)
    img3, spacing2, centers3, lungs_bounding_box3 = rotate(img=img2, spacing=spacing1, centers=centers2,
                                                           lungs_bounding_box=lungs_bounding_box2, rotate_id=rotate_id)
    return img3, radii1, centers3, lungs_bounding_box3, spacing2, existing_centers_in_patch


def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    plt_number = 0
    # Original image label: 0
    if plot:
        f, plots = plt.subplots(12, 1, figsize=(5, 40))
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(im, cmap=plt.cm.bone)
        plt_number += 1

    # Step 1: Convert into a binary image.
    # image label: 1
    binary = im < -604
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    # Step 2: Remove the blobs connected to the border of the image.
    # image label: 2
    cleared = clear_border(binary)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(cleared, cmap=plt.cm.bone)
        plt_number += 1
    # Step 3: Label the image.
    # image label: 3
    label_image = label(cleared)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(label_image, cmap=plt.cm.bone)
        plt_number += 1

    # Step 4: Keep the labels with 2 largest areas and segment two lungs.
    # image label: 4
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
    # Step 5: Fill in the small holes inside the mask of lungs which we seperate right and left lung.
    # r and l are symbolic and they can be actually left and right!
    # image labels: 5, 6
    rig = label_image == labels[0]
    lef = label_image == labels[1]
    r_edges = roberts(rig)
    l_edges = roberts(lef)
    rig = ndi.binary_fill_holes(r_edges)
    lef = ndi.binary_fill_holes(l_edges)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(rig, cmap=plt.cm.bone)
        plt_number += 1

        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(lef, cmap=plt.cm.bone)
        plt_number += 1

    # Step 6: convex hull of each lung
    # image labels: 7, 8
    rig = convex_hull_image(rig)
    lef = convex_hull_image(lef)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(rig, cmap=plt.cm.bone)
        plt_number += 1

        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(lef, cmap=plt.cm.bone)
        plt_number += 1
    # Step 7: joint two separated right and left lungs.
    # image label: 9
    sum_of_lr = rig + lef
    binary = sum_of_lr > 0
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    # Step 8: Closure operation with a disk of radius 10. This operation is
    # to keep nodules attached to the lung wall.
    # image label: 10
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(binary, cmap=plt.cm.bone)
        plt_number += 1
    # Step 9: Superimpose the binary mask on the input image.
    # image label: 11
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot:
        plots[plt_number].axis('off')
        plots[plt_number].set_title(f'{plt_number}')
        plots[plt_number].imshow(im, cmap=plt.cm.bone)
        plt_number += 1

    return im, convex_hull_image(binary)
