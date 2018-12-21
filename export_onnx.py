"""
Export pretrained model to ONNX format.
This is a rough sketch.
For more information see

https://github.com/chainer/onnx-chainer

"""
import argparse
import configparser
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os

import chainer
import chainer.links as L
from chainer import initializers
import numpy as np
import onnx
import onnx_chainer

from predict import load_config
from utils import parse_size


def get_network(model, **kwargs):
    if model == 'mv2':
        from network_mobilenetv2 import MobilenetV2
        return MobilenetV2(**kwargs)
    elif model == 'resnet50':
        from network_resnet import ResNet50
        return ResNet50(**kwargs)
    elif model == 'resnet18':
        from network_resnet import ResNet
        return ResNet(n_layers=18)
    elif model == 'resnet34':
        from network_resnet import ResNet
        return ResNet(n_layers=34)
    else:
        raise Exception('Invalid model name')


class MyModel(chainer.Chain):

    def __init__(self, config):
        super(MyModel, self).__init__()

        dataset_type = config.get('dataset', 'type')
        if dataset_type == 'mpii':
            import mpii_dataset as x_dataset
        elif dataset_type == 'coco':
            import coco_dataset as x_dataset
        else:
            raise Exception('Unknown dataset {}'.format(dataset_type))

        with self.init_scope():
            dtype = np.float32
            self.feature_layer = get_network(config.get('model_param', 'model_name'), dtype=dtype, width_multiplier=1.0)
            ksize = self.feature_layer.last_ksize
            self.local_grid_size = parse_size(config.get('model_param', 'local_grid_size'))
            self.keypoint_names = x_dataset.KEYPOINT_NAMES
            self.edges = x_dataset.EDGES
            self.lastconv = L.Convolution2D(None,
                                            6 * len(self.keypoint_names) +
                                            self.local_grid_size[0] * self.local_grid_size[1] * len(self.edges),
                                            ksize=ksize, stride=1, pad=ksize // 2,
                                            initialW=initializers.HeNormal(1 / np.sqrt(2), dtype))

    def __call__(self, x):
        h = self.feature_layer(x)
        h = self.feature_layer.last_activation(self.lastconv(h))
        return h


def export_onnx(args):
    config = load_config(args)
    model = MyModel(config)
    chainer.serializers.load_npz(os.path.join(args.model, 'bestmodel.npz'), model)
    w, h = parse_size(config.get('model_param', 'insize'))
    x = np.zeros((1, 3, h, w), dtype=np.float32)
    logger.info('begin export')
    output = os.path.join(args.model, 'bestmodel.onnx')
    with chainer.using_config('train', False):
        onnx_chainer.export(model, x, filename=output)
    logger.info('end export')
    logger.info('run onnx.check')
    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    logger.info('done')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path/to/model', type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    export_onnx(args)

if __name__ == '__main__':
    main()
