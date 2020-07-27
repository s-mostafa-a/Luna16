import math
from functools import partial

import numpy as np
from scipy.ndimage.interpolation import zoom
import scipy


def nms(output, nms_th):
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


# works fine
def get_cube_from_img_new(img, origin: tuple, block_size=128, pad_value=106):
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
            end_at_result_image = start_at_result_image + img.shape[i]
            end_at_original_image = img.shape[i]
        else:
            end_at_result_image = block_size
        start_at_original_images.append(start_at_original_image)
        end_at_original_images.append(end_at_original_image)
        start_at_result_images.append(start_at_result_image)
        end_at_result_images.append(end_at_result_image)
    if len(origin) == 3:
        result[start_at_result_images[0]:end_at_result_images[0], start_at_result_images[1]:end_at_result_images[1],
        start_at_result_images[2]:end_at_result_images[2]] = img[start_at_original_images[0]:end_at_original_images[0],
                                                             start_at_original_images[1]:end_at_original_images[1],
                                                             start_at_original_images[2]:end_at_original_images[2]]
    elif len(origin) == 2:
        result[start_at_result_images[0]:end_at_result_images[0],
        start_at_result_images[1]:end_at_result_images[1]] = img[start_at_original_images[0]:end_at_original_images[0],
                                                             start_at_original_images[1]:end_at_original_images[1]]

    return result


def random_crop(img: np.array, origin: tuple, radius: float, spacing: tuple, block_size=128, pad_value=106):
    max_radius_index = np.max(np.round(radius / np.array(spacing)).astype(int))
    new_origin = list(origin)
    shifts = []
    for i in range(len(origin)):
        high = int(block_size / 2) - max_radius_index
        shift = np.random.randint(low=-abs(high), high=abs(high))
        new_origin[i] += shift
        shifts.append(shift)
    print('origin:', tuple(new_origin))
    print('shifts:', shifts)
    print('block_size:', block_size)
    out_img = get_cube_from_img_new(img, origin=tuple(new_origin), block_size=block_size, pad_value=pad_value)
    out_origin = np.array([int(block_size / 2)] * len(origin), dtype=int) - np.array(shifts, dtype=int)
    return out_img, tuple(out_origin)


# works for final version
def _get_point_after_2d_rotation(in_point: tuple, shape: tuple, rot90s: int, flip: bool = False):
    assert len(in_point) == 2 and len(shape) == 2
    rot90s = rot90s % 4
    result_point = list(in_point)
    for i in range(rot90s):
        previous = result_point.copy()
        axes = [0, 1]
        point_complement = (shape[0] - 1 - previous[0], shape[1] - 1 - previous[1])
        result_point[axes[0]] = point_complement[axes[1]]
        result_point[axes[1]] = previous[axes[0]]
    if flip:
        result_point[0] = shape[0] - 1 - result_point[0]
    return result_point


# works for final version
def _get_point_after_3d_rotation(in_point: tuple, shape: tuple, axes, rot90s: int, flip: bool = False):
    rot90s = rot90s % 4
    result_point = list(in_point)
    other_axis = [item for item in [0, 1, 2] if item not in axes]
    for i in range(rot90s):
        previous = result_point.copy()
        point_complement = np.array(shape, dtype=int) - np.array(previous, dtype=int) - 1
        result_point[axes[0]] = point_complement[axes[1]]
        result_point[axes[1]] = previous[axes[0]]
    if flip:
        result_point[other_axis[0]] = shape[other_axis[0]] - 1 - result_point[other_axis[0]]
    return result_point


# works for final version
def rotate(img: np.array, spacing: list, origin: tuple, rotate_id: int):
    dimensions = len(img.shape)
    assert (dimensions == 3 and rotate_id < 24) or (dimensions == 2 and rotate_id < 8)
    other_axes = [i for i in range(dimensions)]

    if dimensions == 2:
        axis = [0]
        out_origin = partial(_get_point_after_2d_rotation, in_point=origin, shape=tuple(img.shape))
    else:  # dimensions == 3
        axis = rotate_id // 8
        other_axes.pop(axis)
        out_origin = partial(_get_point_after_3d_rotation, in_point=origin, shape=tuple(img.shape), axes=other_axes)

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
    return img, spacing, out_origin(rot90s=rotation_times, flip=flip)


# works for final version
def scale(img: np.array, scale_factor: float, spacing: list, origin: tuple, r: float):
    assert (.75 <= scale_factor <= 1.25)
    out_origin = tuple(np.floor(np.array(origin) * scale_factor).astype(int))
    out_r = math.ceil(r * scale_factor)
    spacing = np.array(spacing) * scale_factor
    img1 = scipy.ndimage.interpolation.zoom(img, spacing, mode='nearest')
    return img1, list(spacing), out_origin, out_r


# tst_img = np.ones((8, 8))
# tst_img[3, 3] = 0
# tst_img[3, 7] = 0
# print(tst_img)
# new_img, sp, ouor, oura = scale(img=tst_img, scale_factor=.75, spacing=[1., 1.], origin=(3, 3), r=4.)
# for row in new_img:
#     print(list(row))
# print('spacing', sp)
# print('origin', ouor)
# print('radius', oura)


# tst_img = np.ones((7, 8))
# tst_img[3, 3] = 0
# tst_img[3, 7] = 0
# print(tst_img[:, :].astype(int))
# new_img, sp = rotate(tst_img, 1, [1., 2.], (3, 3))
# for row in new_img[:, :].astype(int):
#     print(list(row))
# print(sp)
# print(new_img.shape)

# print(_get_point_after_2d_rotation((2, 1), (3, 3), 2, True))
# tst_img = np.ones((3, 3))
# tst_img[1, 2] = 0
# print(np.flip(tst_img, 0))
# print(tst_img)

#
# tst_img = np.ones((9, 9))
# tst_img[1, 2] = 0
#
# new_img, sp, org = rotate(tst_img, [1., 1.], (1, 2), 2)
# print(new_img[org[0], org[1]])

tst_img = np.ones((500, 500, 500), dtype=int)
tst_img[100, 100, 100] = 0
new_img, new_out = random_crop(img=tst_img, origin=(100, 100, 100), radius=2, spacing=(1., 1., 1.), block_size=100,
                               pad_value=1)
print(new_img[new_out])
#
# for i in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
#     for j in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
#         for k in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
#             ls.append(new_img[new_out[0] + i, new_out[1] + j, new_out[2] + k])
#             if new_img[new_out[0] + i, new_out[1] + j, new_out[2] + k] == 0.:
#                 print('alooooo')
# print(sum(ls)/len(ls))
