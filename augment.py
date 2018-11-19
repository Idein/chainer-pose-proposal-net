import numpy as np
from scipy import ndimage


def rotate_point(point_yx, degree, center_yx):
    offset_x, offset_y = center_yx
    shift = point_yx - center_yx
    shift_y, shift_x = shift[:, 0], shift[:, 1]
    cos_rad = np.cos(np.deg2rad(degree))
    sin_rad = np.sin(np.deg2rad(degree))
    qx = offset_x + cos_rad * shift_x + sin_rad * shift_y
    qy = offset_y - sin_rad * shift_x + cos_rad * shift_y
    return np.array([qy, qx]).transpose()


def rot_image(image, degree):
    # CHW => HWC
    image = image.transpose(1, 2, 0)
    rot = ndimage.rotate(image, degree, reshape=False)
    # HWC => CHW
    rot = rot.transpose(2, 0, 1)
    return rot


def rotate(image, keypoints, bbox, degree):
    new_keypoints = []
    center_yx = np.array(image.shape[1:]) / 2
    for points in keypoints:
        rot_points = rotate_point(np.array(points),
                                  degree,
                                  center_yx)
        new_keypoints.append(rot_points)

    new_bbox = []
    for x, y, w, h in bbox:
        points = np.array([[y + h / 2, x + w / 2]])
        ry, rx = rotate_point(points,
                              degree,
                              center_yx)[0]
        new_bbox.append([rx - w / 2, ry - h / 2, w, h])

    rot = rot_image(image, degree)
    return rot, new_keypoints, new_bbox
