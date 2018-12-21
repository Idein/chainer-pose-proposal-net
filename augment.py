import random

import numpy as np
from scipy import ndimage
import chainercv.transforms as transforms
from chainercv.links.model.ssd.transforms import random_distort
import PIL
from PIL import ImageChops, ImageOps, ImageFilter, ImageEnhance


def rotate_point(point_yx, angle, center_yx):
    offset_y, offset_x = center_yx
    shift = point_yx - center_yx
    shift_y, shift_x = shift[:, 0], shift[:, 1]
    cos_rad = np.cos(np.deg2rad(angle))
    sin_rad = np.sin(np.deg2rad(angle))
    qx = offset_x + cos_rad * shift_x + sin_rad * shift_y
    qy = offset_y - sin_rad * shift_x + cos_rad * shift_y
    return np.array([qy, qx]).transpose()


def rotate_image(image, angle):
    rot = ndimage.rotate(image, angle, axes=(2, 1), reshape=False)
    # disable image collapse
    rot = np.clip(rot, 0, 255)
    return rot


def random_rotate(image, keypoints, bbox):
    angle = np.random.uniform(-40, 40)
    param = {}
    param['angle'] = angle
    new_keypoints = []
    center_yx = np.array(image.shape[1:]) / 2
    for points in keypoints:
        rot_points = rotate_point(np.array(points),
                                  angle,
                                  center_yx)
        new_keypoints.append(rot_points)

    new_bbox = []
    for x, y, w, h in bbox:

        points = np.array(
            [
                [y, x],
                [y, x + w],
                [y + h, x],
                [y + h, x + w]
            ]
        )

        rot_points = rotate_point(
            points,
            angle,
            center_yx
        )
        xmax = np.max(rot_points[:, 1])
        ymax = np.max(rot_points[:, 0])
        xmin = np.min(rot_points[:, 1])
        ymin = np.min(rot_points[:, 0])
        # x,y,w,h
        new_bbox.append([xmin, ymin, xmax - xmin, ymax - ymin])

    image = rotate_image(image, angle)
    return image, new_keypoints, new_bbox, param


def spot_light(pil_img):
    w, h = pil_img.size
    effect_img = np.zeros((h, w, 3))
    scale_w = random.choice([5, 6, 7, 8, 9])
    scale_h = random.choice([5, 6, 7, 8, 9])
    x = random.choice(range(w // scale_w, w - w // scale_w))
    y = random.choice(range(h // scale_h, h - h // scale_h))
    light = random.choice(range(128, 220))
    effect_img[y - h // scale_h:y + h // scale_h, x - w // scale_w:x + w // scale_w] = light
    effect_img = PIL.Image.fromarray(effect_img.astype(np.uint8))
    return ImageChops.add(pil_img, effect_img)


def blend_alpha(pil_img, direction='left'):
    w, h = pil_img.size
    effect_img = np.zeros((h, w, 3))
    if direction == 'right':
        for x in range(w):
            effect_img[:, x] = x * 255 / w
    elif direction == 'left':
        for x in range(w):
            effect_img[:, x] = (w - x) * 255 / w
    elif direction == 'up':
        for y in range(h):
            effect_img[y, :] = (h - y) * 255 / h
    elif direction == 'down':
        for y in range(h):
            effect_img[y, :] = y * 255 / h
    else:
        raise Exception("invalid argument direction is 'right','left','up','down' actual {}".format(direction))
    effect_img = PIL.Image.fromarray(effect_img.astype(np.uint8))
    return PIL.Image.blend(pil_img, effect_img, 0.5)


def chop_image(pil_img, direction='left', op='add'):
    w, h = pil_img.size
    effect_img = np.zeros((h, w, 3))
    if direction == 'right':
        for x in range(w):
            effect_img[:, x] = x * 255 / w
    elif direction == 'left':
        for x in range(w):
            effect_img[:, x] = (w - x) * 255 / w
    elif direction == 'up':
        for y in range(h):
            effect_img[y, :] = (h - y) * 255 / h
    elif direction == 'down':
        for y in range(h):
            effect_img[y, :] = y * 255 / h
    else:
        raise Exception("invalid argument direction. It should be 'right','left','up','down' actual {}".format(direction))
    effect_img = PIL.Image.fromarray(effect_img.astype(np.uint8))
    if op == 'add':
        operation = ImageChops.add
    elif op == 'subtract':
        operation = ImageChops.subtract
    elif op == 'multiply':
        operation = ImageChops.multiply
    elif op == 'screen':
        operation = ImageChops.screen
    elif op == 'lighter':
        operation = ImageChops.lighter
    elif op == 'darker':
        operation = ImageChops.darker
    else:
        ops = ['add', 'subtract', 'multiply', 'screen', 'lighter', 'darker']
        raise Exception("invalid argument op. {} actual {}".format(ops, direction))
    return operation(pil_img, effect_img)


def filter_image(pil_img):
    method = random.choice(['gaussian', 'blur', 'sharpen'])
    if method == 'gaussian':
        return pil_img.filter(ImageFilter.GaussianBlur(random.choice([0.5, 1.0, 1.5])))
    if method == 'blur':
        return pil_img.filter(ImageFilter.BLUR)
    if method == 'sharpen':
        return pil_img.filter(ImageFilter.SHARPEN)


def random_process_by_PIL(image):
    # convert CHW -> HWC -> PIL.Image
    pil_img = PIL.Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))

    method = np.random.choice(
        ['equalize', 'spot_light', 'chop', 'blend'],
        p=[0.15, 0.15, 0.35, 0.35]
    )

    param = {'pil': method, 'filter': False}
    if method == 'equalize':
        pil_img = ImageOps.equalize(pil_img)
    if method == 'spot_light':
        pil_img = spot_light(pil_img)
    if method == 'chop':
        direction = random.choice(['left', 'right', 'up', 'down'])
        op = random.choice(['add', 'subtract', 'multiply', 'screen', 'lighter', 'darker'])
        pil_img = chop_image(pil_img, direction, op)
    if method == 'blend':
        direction = random.choice(['left', 'right', 'up', 'down'])
        pil_img = blend_alpha(pil_img, direction)

    if np.random.choice([True, False], p=[0.1, 0.9]):
        pil_img = filter_image(pil_img)
        param['filter'] = True
    # back to CHW
    image = np.asarray(pil_img).transpose(2, 0, 1).astype(np.float32)
    return image, param


def augment_image(image, dataset_type):
    """color augmentation"""
    param = {}

    if dataset_type == 'mpii':
        method = np.random.choice(
            ['random_distort', 'pil'],
            p=[1.0, 0.0],
        )
    elif dataset_type == 'coco':
        method = np.random.choice(
            ['random_distort', 'pil'],
            p=[0.5, 0.5],
        )

    if method == 'random_distort':
        param['method'] = method
        # color augmentation provided by ChainerCV
        ret = random_distort(image, contrast_low=0.3, contrast_high=2)
        return ret, param
    if method == 'pil':
        ret, param = random_process_by_PIL(image)
        param['method'] = method
        return ret, param


def random_flip(image, keypoints, bbox, is_labeled, is_visible, flip_indices):
    """
    random x_flip
    Note that if image is flipped, `flip_indices` translate elements.
    e.g. left_shoulder -> right_shoulder.
    """
    _, H, W = image.shape
    image, param = transforms.random_flip(image, x_random=True, return_param=True)

    if param['x_flip']:
        keypoints = [
            transforms.flip_point(points, (H, W), x_flip=True)[flip_indices]
            for points in keypoints
        ]

        is_labeled = [label[flip_indices] for label in is_labeled]
        is_visible = [vis[flip_indices] for vis in is_visible]

        new_bbox = []
        for x, y, w, h in bbox:
            [[y, x]] = transforms.flip_point(np.array([[y, x + w]]), (H, W), x_flip=True)
            new_bbox.append([x, y, w, h])
        bbox = new_bbox

    return image, keypoints, bbox, is_labeled, is_visible, param


def scale_fit_short(image, keypoints, bbox, length):
    _, H, W = image.shape
    min_hw = min(H, W)
    scale = length / min_hw
    new_image = transforms.scale(image, size=length, fit_short=True)
    new_keypoints = [scale * k for k in keypoints]
    new_bbox = [scale * np.asarray(b) for b in bbox]
    return new_image, new_keypoints, new_bbox


def intersection(bbox0, bbox1):
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

    def relu(x): return max(0, x)
    w = relu(min(x0 + w0, x1 + w1) - max(x0, x1))
    h = relu(min(y0 + h0, y1 + h1) - max(y0, y1))
    return w * h


def translate_bbox(bbox, size, y_offset, x_offset):
    cropped_H, cropped_W = size
    new_bbox = []
    for x, y, w, h in bbox:
        x_shift = x + x_offset
        y_shift = y + y_offset
        is_intersect = intersection([0, 0, cropped_W, cropped_H], [x_shift, y_shift, w, h])
        if is_intersect:
            xmin = max(0, x_shift)
            ymin = max(0, y_shift)
            xmax = min(cropped_W, x_shift + w)
            ymax = min(cropped_H, y_shift + h)
            wnew = xmax - xmin
            hnew = ymax - ymin
            new_bbox.append([xmin, ymin, wnew, hnew])
        else:
            new_bbox.append([x_shift, y_shift, w, h])
    return new_bbox


def crop(img, y_slice, x_slice, copy=False):
    ret = img.copy() if copy else img
    return ret[:, y_slice, x_slice]


def crop_all_humans(image, keypoints, bbox, is_labeled):
    _, H, W = image.shape
    aspect = W / H
    param = {}
    if len(keypoints) == 0:
        param['do_nothing'] = True
        return image, keypoints, bbox, param

    kymax = max([np.max(ks[l, 0]) for l, ks in zip(is_labeled, keypoints)])
    kxmax = max([np.max(ks[l, 1]) for l, ks in zip(is_labeled, keypoints)])
    kymin = min([np.min(ks[l, 0]) for l, ks in zip(is_labeled, keypoints)])
    kxmin = min([np.min(ks[l, 1]) for l, ks in zip(is_labeled, keypoints)])

    bxmax = max([b[0] + b[2] for b in bbox])
    bymax = max([b[1] + b[3] for b in bbox])
    bxmin = min([b[0] for b in bbox])
    bymin = min([b[1] for b in bbox])

    ymax = max(kymax, bymax)
    xmax = max(kxmax, bxmax)
    ymin = min(kymin, bymin)
    xmin = min(kxmin, bxmin)

    if (xmax + xmin) / 2 < W / 2:
        x_start = random.randint(0, max(0, int(xmin)))
        y_start = random.randint(0, max(0, int(ymin)))
        y_end = random.randint(min(H, int(ymax)), H)
        ylen = y_end - y_start
        xlen = aspect * ylen
        x_end = min(W, int(x_start + xlen))
        x_slice = slice(x_start, x_end, None)
        y_slice = slice(y_start, y_end, None)
    else:
        x_end = random.randint(min(int(xmax), W), W)
        y_end = random.randint(min(int(ymax), H), H)
        y_start = random.randint(0, max(0, int(ymin)))
        ylen = y_end - y_start
        xlen = aspect * ylen
        x_start = max(0, int(x_end - xlen))
        x_slice = slice(x_start, x_end, None)
        y_slice = slice(y_start, y_end, None)

    cropped = crop(image, y_slice=y_slice, x_slice=x_slice, copy=True)
    _, cropped_H, cropped_W = cropped.shape
    param['x_slice'] = x_slice
    param['y_slice'] = y_slice
    if cropped_H <= 50 or cropped_W <= 50:
        """
        This case, for example, cropped_H=0 will cause an error when try to resize image
        or resize small image to insize will cause low resolution human image.
        To avoid situations, we will stop crop image.
        """
        param['do_nothing'] = True
        return image, keypoints, bbox, param
    image = cropped

    keypoints = [
        transforms.translate_point(
            points, x_offset=-x_slice.start, y_offset=-y_slice.start)
        for points in keypoints
    ]

    bbox = translate_bbox(
        bbox,
        size=(cropped_H, cropped_W),
        x_offset=-x_slice.start,
        y_offset=-y_slice.start,
    )

    return image, keypoints, bbox, param


def random_sized_crop(image, keypoints, bbox):
    image, param = transforms.random_sized_crop(
        image,
        scale_ratio_range=(0.5, 5),
        aspect_ratio_range=(0.75, 1.3333333333333333),
        return_param=True
    )

    keypoints = [
        transforms.translate_point(points,
                                   x_offset=-param['x_slice'].start,
                                   y_offset=-param['y_slice'].start
                                   )
        for points in keypoints
    ]

    _, cropped_H, cropped_W = image.shape

    bbox = translate_bbox(
        bbox,
        size=(cropped_H, cropped_W),
        x_offset=-param['x_slice'].start,
        y_offset=-param['y_slice'].start,
    )

    return image, keypoints, bbox, {random_sized_crop.__name__: param}


def resize(image, keypoints, bbox, size):
    _, H, W = image.shape
    new_h, new_w = size
    image = transforms.resize(image, (new_h, new_w))

    keypoints = [
        transforms.resize_point(points, (H, W), (new_h, new_w))
        for points in keypoints
    ]

    new_bbox = []
    for x, y, bw, bh in bbox:
        [[y, x]] = transforms.resize_point(np.array([[y, x]]), (H, W), (new_h, new_w))
        bw *= new_w / W
        bh *= new_h / H
        new_bbox.append([x, y, bw, bh])
    return image, keypoints, new_bbox


def random_resize(image, keypoints, bbox):
    # Random resize
    _, H, W = image.shape
    scalew, scaleh = np.random.uniform(0.7, 1.5, 2)
    resizeW, resizeH = int(W * scalew), int(H * scaleh)
    image, keypoints, bbox = resize(image, keypoints, bbox, (resizeH, resizeW))
    return image, keypoints, bbox, {'H': resizeH, 'W': resizeW}


def random_crop(image, keypoints, bbox, is_labeled, dataset_type):
    if dataset_type == 'mpii':
        crop_target = np.random.choice(
            ['random_sized_crop', 'crop_all_humans'],
            p=[0.1, 0.9],
        )
    if dataset_type == 'coco':
        crop_target = np.random.choice(
            ['random_sized_crop', 'crop_all_humans'],
            p=[0.5, 0.5],
        )

    param = {'crop_target': crop_target}
    if crop_target == 'random_sized_crop':
        image, keypoints, bbox, p = random_resize(image, keypoints, bbox)
        param.update(p)
        image, keypoints, bbox, p = random_sized_crop(image, keypoints, bbox)
        param.update(p)
    elif crop_target == 'crop_all_humans':
        image, keypoints, bbox, p = crop_all_humans(image, keypoints, bbox, is_labeled)
        param.update(p)

    return image, keypoints, bbox, param
