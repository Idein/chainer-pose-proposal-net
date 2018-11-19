import numpy as np
import json
from logging import getLogger
logger = getLogger('__main__')

from dataset import KeypointDataset2D
from utils import pairwise


DEFAULT_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

FLIP_CONVERTER = {
    'nose': 'nose',
    'neck': 'neck',
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}

# update keypoints
KEYPOINT_NAMES = ['neck'] + DEFAULT_KEYPOINT_NAMES
FLIP_INDICES = [KEYPOINT_NAMES.index(FLIP_CONVERTER[k]) for k in KEYPOINT_NAMES]
# update keypoints
KEYPOINT_NAMES = ['instance'] + KEYPOINT_NAMES

COLOR_MAP = {
    'instance': (225, 225, 225),
    'nose': (255, 0, 0),
    'neck': (255, 85, 0),
    'right_shoulder': (255, 170, 0),
    'right_elbow': (255, 255, 0),
    'right_wrist': (170, 255, 0),
    'left_shoulder': (85, 255, 0),
    'left_elbow': (0, 127, 0),
    'left_wrist': (0, 255, 85),
    'right_hip': (0, 170, 170),
    'right_knee': (0, 255, 255),
    'right_ankle': (0, 170, 255),
    'left_hip': (0, 85, 255),
    'left_knee': (0, 0, 255),
    'left_ankle': (85, 0, 255),
    'right_eye': (170, 0, 255),
    'left_eye': (255, 0, 255),
    'right_ear': (255, 0, 170),
    'left_ear': (255, 0, 85),
}

EDGES_BY_NAME = [
    ['instance', 'neck'],
    ['neck', 'nose'],
    ['nose', 'left_eye'],
    ['left_eye', 'left_ear'],
    ['nose', 'right_eye'],
    ['right_eye', 'right_ear'],
    ['neck', 'left_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['neck', 'right_shoulder'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['neck', 'left_hip'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['neck', 'right_hip'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
]

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

TRACK_ORDER_0 = ['instance', 'neck', 'nose', 'left_eye', 'left_ear']
TRACK_ORDER_1 = ['instance', 'neck', 'nose', 'right_eye', 'right_ear']
TRACK_ORDER_2 = ['instance', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist']
TRACK_ORDER_3 = ['instance', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist']
TRACK_ORDER_4 = ['instance', 'neck', 'left_hip', 'left_knee', 'left_ankle']
TRACK_ORDER_5 = ['instance', 'neck', 'right_hip', 'right_knee', 'right_ankle']

TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4, TRACK_ORDER_5]
DIRECTED_GRAPHS = []

for keypoints in TRACK_ORDERS:
    es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
    ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
    DIRECTED_GRAPHS.append([es, ts])


def get_coco_dataset(insize, image_root, annotations,
                     min_num_keypoints=1, use_cache=False, do_augmentation=False):
    cat_id = 1
    dataset_type = 'coco'
    dataset = json.load(open(annotations, 'r'))
    cat = dataset['categories'][cat_id - 1]
    assert cat['keypoints'] == DEFAULT_KEYPOINT_NAMES
    # image_id => filename, keypoints, bbox, is_visible, is_labeled
    images = {}

    for image in dataset['images']:
        images[image['id']] = image['file_name'], [], [], [], []

    for anno in dataset['annotations']:
        if anno['num_keypoints'] < min_num_keypoints:
            continue
        if anno['category_id'] != cat_id:
            continue
        if anno['iscrowd'] != 0:
            continue
        image_id = anno['image_id']
        d = np.array(anno['keypoints'], dtype='float32').reshape(-1, 3)
        # define neck from left_shoulder and right_shoulder
        left_shoulder_idx = DEFAULT_KEYPOINT_NAMES.index('left_shoulder')
        right_shoulder_idx = DEFAULT_KEYPOINT_NAMES.index('right_shoulder')
        left_shoulder, left_v = d[left_shoulder_idx][:2], d[left_shoulder_idx][2]
        right_shoulder, right_v = d[right_shoulder_idx][:2], d[right_shoulder_idx][2]
        if left_v >= 1 and right_v >= 1:
            neck = (left_shoulder + right_shoulder) / 2.
            labeled = 1
            d = np.vstack([np.array([*neck, labeled]), d])
        else:
            labeled = 0
            # insert dummy data correspond to `neck`
            d = np.vstack([np.array([0.0, 0.0, labeled]), d])

        keypoints = d[:, [1, 0]]  # array of y,x
        bbox = anno['bbox']
        is_visible = d[:, 2] == 2
        is_labeled = d[:, 2] >= 1

        entry = images[image_id]
        entry[1].append(keypoints)
        entry[2].append(bbox)
        entry[3].append(is_visible)
        entry[4].append(is_labeled)

    # filter-out non annotated images
    image_paths = []
    keypoints = []
    bbox = []
    is_visible = []
    is_labeled = []

    for filename, k, b, v, l in images.values():
        if len(k) == 0:
            continue
        image_paths.append(filename)
        bbox.append(b)
        keypoints.append(k)
        is_visible.append(v)
        is_labeled.append(l)

    return KeypointDataset2D(
        dataset_type=dataset_type,
        insize=insize,
        keypoint_names=KEYPOINT_NAMES,
        edges=np.array(EDGES),
        flip_indices=FLIP_INDICES,
        keypoints=keypoints,
        bbox=bbox,
        is_visible=is_visible,
        is_labeled=is_labeled,
        image_paths=image_paths,
        image_root=image_root,
        use_cache=use_cache,
        do_augmentation=do_augmentation
    )
