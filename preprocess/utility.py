import numpy as np
from scipy.ndimage.interpolation import zoom


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


def get_cube_from_img(img, center_z, center_y, center_x, block_size):
    result = np.ones((block_size, block_size, block_size)) * 106
    start_z_orig = int(max(center_z - block_size / 2, 0))
    end_z_orig = start_z_orig + block_size
    start_y_orig = int(max(center_y - block_size / 2, 0))
    end_y_orig = start_y_orig + block_size
    start_x_orig = int(max(center_x - block_size / 2, 0))
    end_x_orig = start_x_orig + block_size

    if start_z_orig < 0:
        start_z_res = abs(start_z_orig)
        start_z_orig = 0
    else:
        start_z_res = 0
    if end_z_orig > img.shape[0]:
        end_z_res = start_z_res + img.shape[0]
        end_z_orig = img.shape[0]
    else:
        end_z_res = block_size

    if start_y_orig < 0:
        start_y_res = abs(start_y_orig)
        start_y_orig = 0
    else:
        start_y_res = 0
    if end_y_orig > img.shape[1]:
        end_y_res = start_y_res + img.shape[1]
        end_y_orig = img.shape[1]
    else:
        end_y_res = block_size

    if start_x_orig < 0:
        start_x_res = abs(start_x_orig)
        start_x_orig = 0
    else:
        start_x_res = 0
    if end_x_orig > img.shape[2]:
        end_x_res = start_x_res + img.shape[2]
        end_x_orig = img.shape[2]
    else:
        end_x_res = block_size

    result[start_z_res:end_z_res, start_y_res:end_y_res, start_x_res:end_x_res] = img[start_z_orig:end_z_orig,
                                                                                  start_y_orig:end_y_orig,
                                                                                  start_x_orig:end_x_orig]
    return result
