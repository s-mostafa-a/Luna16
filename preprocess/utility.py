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

# works totally fine
def get_cube_from_img_new(img, origin: tuple, block_size, pad_value=106):
    assert 2 <= len(origin) <= 3
    final_image_shape = tuple([block_size] * len(origin))
    result = np.ones(final_image_shape) * pad_value
    start_at_original_images = []
    end_at_original_images = []
    start_at_result_images = []
    end_at_result_images = []
    for i, center_of_a_dim in enumerate(origin):
        print(i, center_of_a_dim)
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
        print(3)
        result[start_at_result_images[0]:end_at_result_images[0], start_at_result_images[1]:end_at_result_images[1],
        start_at_result_images[2]:end_at_result_images[2]] = img[start_at_original_images[0]:end_at_original_images[0],
                                                             start_at_original_images[1]:end_at_original_images[1],
                                                             start_at_original_images[2]:end_at_original_images[2]]
    elif len(origin) == 2:
        print(2)
        result[start_at_result_images[0]:end_at_result_images[0],
        start_at_result_images[1]:end_at_result_images[1]] = img[start_at_original_images[0]:end_at_original_images[0],
                                                             start_at_original_images[1]:end_at_original_images[1]]

    return result


# works totally fine
def rotate_3d(img: np.array, rotate_id: int, height_width_length: list):
    assert (rotate_id < 24)
    axis = rotate_id // 8
    which_rotation = rotate_id % 8
    flip = which_rotation >= 4
    rotation_degree = (which_rotation % 4) * 90
    other_axes = [0, 1, 2]
    other_axes.pop(axis)
    hwl_exchanged = (which_rotation % 2) != 0
    if hwl_exchanged:
        tmp = height_width_length[other_axes[0]]
        height_width_length[other_axes[0]] = height_width_length[other_axes[1]]
        height_width_length[other_axes[1]] = tmp
    img = scipy.ndimage.interpolation.rotate(img, angle=rotation_degree, axes=other_axes)
    if flip:
        img = np.flip(img, axis=axis)
    return img, height_width_length


# totally works fine
def scale(img: np.array, scale_factor: float, height_width_length: list):
    assert (.75 <= scale_factor <= 1.25)
    spacing = np.array(height_width_length) * scale_factor
    img1 = scipy.ndimage.interpolation.zoom(img, spacing, mode='nearest')
    return img1, list(spacing)


tst_img = np.ones((8, 8))
tst_img[4, 4] = 0
tst_img[7, 7] = 0
print(tst_img)
new_img, hwl = scale(img=tst_img, scale_factor=.75, height_width_length=[1., 1.])
for row in new_img:
    print(list(row))
# print([list(row) for row in new_img])
print(hwl)
