import numpy as np
import json

from dataset import KeypointDataset2D
from chainer.datasets import split_dataset_random

from utils import pairwise

KEYPOINT_NAMES = [
    'head_top',
    'upper_neck',
    'l_shoulder',
    'r_shoulder',
    'l_elbow',
    'r_elbow',
    'l_wrist',
    'r_wrist',
    'l_hip',
    'r_hip',
    'l_knee',
    'r_knee',
    'l_ankle',
    'r_ankle',
]

FLIP_CONVERTER = {'head_top': 'head_top',
                  'upper_neck': 'upper_neck',
                  'l_shoulder': 'r_shoulder',
                  'r_shoulder': 'l_shoulder',
                  'l_elbow': 'r_elbow',
                  'r_elbow': 'l_elbow',
                  'l_wrist': 'r_wrist',
                  'r_wrist': 'l_wrist',
                  'l_hip': 'r_hip',
                  'r_hip': 'l_hip',
                  'l_knee': 'r_knee',
                  'r_knee': 'l_knee',
                  'l_ankle': 'r_ankle',
                  'r_ankle': 'l_ankle',
                  }

FLIP_INDICES = [KEYPOINT_NAMES.index(FLIP_CONVERTER[k]) for k in KEYPOINT_NAMES]

KEYPOINT_NAMES = ['instance'] + KEYPOINT_NAMES

COLOR_MAP = {
    'instance': (225, 225, 225),
    'head_top': (255, 0, 0),
    'upper_neck': (255, 85, 0),
    'r_shoulder': (255, 170, 0),
    'r_elbow': (255, 255, 0),
    'r_wrist': (170, 255, 0),
    'l_shoulder': (85, 255, 0),
    'l_elbow': (0, 127, 0),
    'l_wrist': (0, 255, 85),
    'r_hip': (0, 170, 170),
    'r_knee': (0, 255, 255),
    'r_ankle': (0, 170, 255),
    'l_hip': (0, 85, 255),
    'l_knee': (0, 0, 255),
    'l_ankle': (85, 0, 255),
    'r_eye': (170, 0, 255),
    'l_eye': (255, 0, 255),
    'r_ear': (255, 0, 170),
    'l_ear': (255, 0, 85),
}

EDGES_BY_NAME = [
    ['instance', 'upper_neck'],
    ['upper_neck', 'head_top'],
    ['upper_neck', 'l_shoulder'],
    ['upper_neck', 'r_shoulder'],
    ['upper_neck', 'l_hip'],
    ['upper_neck', 'r_hip'],
    ['l_shoulder', 'l_elbow'],
    ['l_elbow', 'l_wrist'],
    ['r_shoulder', 'r_elbow'],
    ['r_elbow', 'r_wrist'],
    ['l_hip', 'l_knee'],
    ['l_knee', 'l_ankle'],
    ['r_hip', 'r_knee'],
    ['r_knee', 'r_ankle'],
]

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

TRACK_ORDER_0 = ['instance', 'upper_neck', 'head_top']
TRACK_ORDER_1 = ['instance', 'upper_neck', 'l_shoulder', 'l_elbow', 'l_wrist']
TRACK_ORDER_2 = ['instance', 'upper_neck', 'r_shoulder', 'r_elbow', 'r_wrist']
TRACK_ORDER_3 = ['instance', 'upper_neck', 'l_hip', 'l_knee', 'l_ankle']
TRACK_ORDER_4 = ['instance', 'upper_neck', 'r_hip', 'r_knee', 'r_ankle']

TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4]
DIRECTED_GRAPHS = []

for keypoints in TRACK_ORDERS:
    es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
    ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
    DIRECTED_GRAPHS.append([es, ts])


def get_mpii_dataset(insize, image_root, annotations,
                     train_size=0.5, min_num_keypoints=1, use_cache=False, seed=0):
    dataset_type = 'mpii'
    annotations = json.load(open(annotations, 'r'))

    # filename => keypoints, bbox, is_visible, is_labeled
    images = {}

    for filename in np.unique([anno['filename'] for anno in annotations]):
        images[filename] = [], [], [], []

    for anno in annotations:
        is_visible = [anno['is_visible'][k] for k in KEYPOINT_NAMES[1:]]
        if sum(is_visible) < min_num_keypoints:
            continue
        keypoints = [anno['joint_pos'][k][::-1] for k in KEYPOINT_NAMES[1:]]

        x1, y1, x2, y2 = anno['head_rect']

        entry = images[anno['filename']]
        entry[0].append(np.array(keypoints))  # array of y,x
        entry[1].append(np.array([x1, y1, x2 - x1, y2 - y1]))  # x, y, w, h
        entry[2].append(np.array(is_visible, dtype=np.bool))
        entry[3].append(np.ones(len(is_visible), dtype=np.bool))

    # split dataset
    train_images, test_images = split_dataset_random(
        list(images.keys()), int(len(images) * train_size), seed=seed)

    train_set = KeypointDataset2D(
        dataset_type=dataset_type,
        insize=insize,
        keypoint_names=KEYPOINT_NAMES,
        edges=np.array(EDGES),
        flip_indices=FLIP_INDICES,
        keypoints=[images[i][0] for i in train_images],
        bbox=[images[i][1] for i in train_images],
        is_visible=[images[i][2] for i in train_images],
        is_labeled=[images[i][3] for i in train_images],
        image_paths=train_images,
        image_root=image_root,
        use_cache=use_cache,
        do_augmentation=True
    )

    test_set = KeypointDataset2D(
        dataset_type=dataset_type,
        insize=insize,
        keypoint_names=KEYPOINT_NAMES,
        edges=np.array(EDGES),
        flip_indices=FLIP_INDICES,
        keypoints=[images[i][0] for i in test_images],
        bbox=[images[i][1] for i in test_images],
        is_visible=[images[i][2] for i in test_images],
        is_labeled=[images[i][3] for i in test_images],
        image_paths=test_images,
        image_root=image_root,
        use_cache=use_cache,
        do_augmentation=False
    )
    return train_set, test_set
