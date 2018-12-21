import argparse
import configparser
from collections import defaultdict
import itertools
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
import random
import time

import numpy as np

import chainer
if chainer.backends.cuda.available:
    import cupy as xp
else:
    xp = np

from chainercv.utils import non_maximum_suppression
from PIL import ImageDraw, Image

from coco_dataset import get_coco_dataset
from mpii_dataset import get_mpii_dataset
from model import PoseProposalNet
from network_resnet import ResNet50
from utils import parse_size

COLOR_MAP = {}
DIRECTED_GRAPHS = [[]]
DATA_MODULE = None


def get_feature(model, image):
    start = time.time()
    image = xp.asarray(image)
    processed_image = model.feature_layer.prepare(image)
    resp, conf, x, y, w, h, e = model.predict(xp.expand_dims(processed_image, axis=0))
    resp = chainer.backends.cuda.to_cpu(resp.array)
    conf = chainer.backends.cuda.to_cpu(conf.array)
    w = chainer.backends.cuda.to_cpu(w.array)
    h = chainer.backends.cuda.to_cpu(h.array)
    x = chainer.backends.cuda.to_cpu(x.array)
    y = chainer.backends.cuda.to_cpu(y.array)
    e = chainer.backends.cuda.to_cpu(e.array)

    resp = np.squeeze(resp, axis=0)
    conf = np.squeeze(conf, axis=0)
    x = np.squeeze(x, axis=0)
    y = np.squeeze(y, axis=0)
    w = np.squeeze(w, axis=0)
    h = np.squeeze(h, axis=0)
    e = np.squeeze(e, axis=0)
    logger.info('inference time {:.5f}'.format(time.time() - start))
    return resp, conf, x, y, w, h, e


def estimate(model, image, detection_thresh=0.15, min_num_keypoints=-1):
    feature_map = get_feature(model, image)
    return get_humans_by_feature(model, feature_map, detection_thresh, min_num_keypoints)


def get_humans_by_feature(model, feature_map, detection_thresh=0.15, min_num_keypoints=-1):
    resp, conf, x, y, w, h, e = feature_map
    start = time.time()
    delta = resp * conf
    K = len(model.keypoint_names)
    outW, outH = model.outsize
    ROOT_NODE = 0  # instance
    start = time.time()
    rx, ry = model.restore_xy(x, y)
    rw, rh = model.restore_size(w, h)
    ymin, ymax = ry - rh / 2, ry + rh / 2
    xmin, xmax = rx - rw / 2, rx + rw / 2
    bbox = np.array([ymin, xmin, ymax, xmax])
    bbox = bbox.transpose(1, 2, 3, 0)
    root_bbox = bbox[ROOT_NODE]
    score = delta[ROOT_NODE]
    candidate = np.where(score > detection_thresh)
    score = score[candidate]
    root_bbox = root_bbox[candidate]
    selected = non_maximum_suppression(
        bbox=root_bbox, thresh=0.3, score=score)
    root_bbox = root_bbox[selected]
    logger.info('detect instance {:.5f}'.format(time.time() - start))
    start = time.time()

    humans = []
    e = e.transpose(0, 3, 4, 1, 2)
    ei = 0  # index of edges which contains ROOT_NODE as begin
    # alchemy_on_humans
    for hxw in zip(candidate[0][selected], candidate[1][selected]):
        human = {ROOT_NODE: bbox[(ROOT_NODE, hxw[0], hxw[1])]}  # initial
        for graph in DIRECTED_GRAPHS:
            eis, ts = graph
            i_h, i_w = hxw
            for ei, t in zip(eis, ts):
                index = (ei, i_h, i_w)  # must be tuple
                u_ind = np.unravel_index(np.argmax(e[index]), e[index].shape)
                j_h = i_h + u_ind[0] - model.local_grid_size[1] // 2
                j_w = i_w + u_ind[1] - model.local_grid_size[0] // 2
                if j_h < 0 or j_w < 0 or j_h >= outH or j_w >= outW:
                    break
                if delta[t, j_h, j_w] < detection_thresh:
                    break
                human[t] = bbox[(t, j_h, j_w)]
                i_h, i_w = j_h, j_w
        if min_num_keypoints <= len(human) - 1:
            humans.append(human)
    logger.info('alchemy time {:.5f}'.format(time.time() - start))
    logger.info('num humans = {}'.format(len(humans)))
    return humans


def draw_humans(keypoint_names, edges, pil_image, humans, mask=None, visbbox=True):
    """
    This is what happens when you use alchemy on humans...
    note that image should be PIL object
    """
    start = time.time()
    drawer = ImageDraw.Draw(pil_image)
    for human in humans:
        for k, b in human.items():
            if mask:
                fill = (255, 255, 255) if k == 0 else None
            else:
                fill = None
            ymin, xmin, ymax, xmax = b
            if k == 0:  # human instance
                # adjust size
                t = 1
                xmin = int(xmin * t + xmax * (1 - t))
                xmax = int(xmin * (1 - t) + xmax * t)
                ymin = int(ymin * t + ymax * (1 - t))
                ymax = int(ymin * (1 - t) + ymax * t)
                if mask:
                    resized = mask.resize(((xmax - xmin), (ymax - ymin)))
                    pil_image.paste(resized, (xmin, ymin), mask=resized)
                else:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
            else:
                if visbbox:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
                else:
                    r = 2
                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2
                    drawer.ellipse((x - r, y - r, x + r, y + r),
                                   fill=COLOR_MAP[keypoint_names[k]])

        for s, t in edges:
            if s in human and t in human:
                by = (human[s][0] + human[s][2]) / 2
                bx = (human[s][1] + human[s][3]) / 2
                ey = (human[t][0] + human[t][2]) / 2
                ex = (human[t][1] + human[t][3]) / 2

                drawer.line([bx, by, ex, ey],
                            fill=COLOR_MAP[keypoint_names[s]], width=3)

    logger.info('draw humans {: .5f}'.format(time.time() - start))
    return pil_image


def create_model(args, config):
    global DIRECTED_GRAPHS, COLOR_MAP

    dataset_type = config.get('dataset', 'type')

    if dataset_type == 'mpii':
        import mpii_dataset as x_dataset
    elif dataset_type == 'coco':
        import coco_dataset as x_dataset
    else:
        raise Exception('Unknown dataset {}'.format(dataset_type))

    KEYPOINT_NAMES = x_dataset.KEYPOINT_NAMES
    EDGES = x_dataset.EDGES
    DIRECTED_GRAPHS = x_dataset.DIRECTED_GRAPHS
    COLOR_MAP = x_dataset.COLOR_MAP

    model = PoseProposalNet(
        model_name=config.get('model_param', 'model_name'),
        insize=parse_size(config.get('model_param', 'insize')),
        keypoint_names=KEYPOINT_NAMES,
        edges=np.array(EDGES),
        local_grid_size=parse_size(config.get('model_param', 'local_grid_size')),
        parts_scale=parse_size(config.get(dataset_type, 'parts_scale')),
        instance_scale=parse_size(config.get(dataset_type, 'instance_scale')),
        width_multiplier=config.getfloat('model_param', 'width_multiplier'),
    )

    result_dir = args.model
    chainer.serializers.load_npz(
        os.path.join(result_dir, 'bestmodel.npz'),
        model
    )

    logger.info('cuda enable {}'.format(chainer.backends.cuda.available))
    logger.info('ideep enable {}'.format(chainer.backends.intel64.is_ideep_available()))
    if chainer.backends.cuda.available:
        logger.info('gpu mode')
        model.to_gpu()
    elif chainer.backends.intel64.is_ideep_available():
        logger.info('Indel64 mode')
        model.to_intel64()
    return model


def load_config(args):
    config = configparser.ConfigParser()
    config_path = os.path.join(args.model, 'src', 'config.ini')
    logger.info(config_path)
    config.read(config_path, 'UTF-8')
    return config


def predict(args):
    config = load_config(args)
    detection_thresh = config.getfloat('predict', 'detection_thresh')
    min_num_keypoints = config.getint('predict', 'min_num_keypoints')
    dataset_type = config.get('dataset', 'type')
    logger.info('loading {}'.format(dataset_type))
    if dataset_type == 'mpii':
        _, test_set = get_mpii_dataset(
            insize=parse_size(config.get('model_param', 'insize')),
            image_root=config.get(dataset_type, 'images'),
            annotations=config.get(dataset_type, 'annotations'),
            train_size=config.getfloat(dataset_type, 'train_size'),
            min_num_keypoints=config.getint(dataset_type, 'min_num_keypoints'),
            seed=config.getint('training_param', 'seed'),
        )
    elif dataset_type == 'coco':
        test_set = get_coco_dataset(
            insize=parse_size(config.get('model_param', 'insize')),
            image_root=config.get(dataset_type, 'val_images'),
            annotations=config.get(dataset_type, 'val_annotations'),
            min_num_keypoints=config.getint(dataset_type, 'min_num_keypoints'),
        )
    else:
        raise Exception('Unknown dataset {}'.format(dataset_type))

    model = create_model(args, config)

    idx = random.choice(range(len(test_set)))
    image = test_set.get_example(idx)['image']
    humans = estimate(
        model,
        image.astype(np.float32),
        detection_thresh,
        min_num_keypoints,
    )
    pil_image = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))
    pil_image = draw_humans(
        keypoint_names=model.keypoint_names,
        edges=model.edges,
        pil_image=pil_image,
        humans=humans,
        visbbox=config.getboolean('predict', 'visbbox')
    )

    pil_image.save('result.png', 'PNG')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path/to/model', type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    predict(args)

if __name__ == '__main__':
    main()
