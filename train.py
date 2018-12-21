import argparse
import random
import configparser

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import matplotlib
matplotlib.use('Agg')

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer import serializers
from chainer.training import extensions
import numpy as np

from model import PoseProposalNet
from coco_dataset import get_coco_dataset
from mpii_dataset import get_mpii_dataset
from utils import parse_size, parse_kwargs, save_files
import visualize


def setup_devices(ids):
    if ids == '':
        return {'main': -1}
    devices = parse_kwargs(ids)
    for key in devices:
        devices[key] = int(devices[key])
    return devices


def set_random_seed(devices, seed):
    random.seed(seed)
    np.random.seed(seed)
    for key, id in devices.items():
        if id < 0:
            break
        if key == 'main':
            chainer.cuda.get_device_from_id(id).use()
            chainer.cuda.cupy.random.seed(seed)


def create_model(config, dataset):
    dataset_type = config.get('dataset', 'type')
    return PoseProposalNet(
        model_name=config.get('model_param', 'model_name'),
        insize=parse_size(config.get('model_param', 'insize')),
        keypoint_names=dataset.keypoint_names,
        edges=dataset.edges,
        local_grid_size=parse_size(config.get('model_param', 'local_grid_size')),
        parts_scale=parse_size(config.get(dataset_type, 'parts_scale')),
        instance_scale=parse_size(config.get(dataset_type, 'instance_scale')),
        width_multiplier=config.getfloat('model_param', 'width_multiplier'),
        lambda_resp=config.getfloat('model_param', 'lambda_resp'),
        lambda_iou=config.getfloat('model_param', 'lambda_iou'),
        lambda_coor=config.getfloat('model_param', 'lambda_coor'),
        lambda_size=config.getfloat('model_param', 'lambda_size'),
        lambda_limb=config.getfloat('model_param', 'lambda_limb'),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.ini')
    parser.add_argument('--resume')
    parser.add_argument('--plot_samples', type=int, default=0)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path, 'UTF-8')

    chainer.global_config.autotune = True
    # chainer.cuda.set_max_workspace_size(11388608)
    chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
    chainer.config.cudnn_fast_batch_normalization = True

    # create result dir and copy file
    logger.info('> store file to result dir %s', config.get('result', 'dir'))
    save_files(config.get('result', 'dir'))

    logger.info('> set up devices')
    devices = setup_devices(config.get('training_param', 'gpus'))
    set_random_seed(devices, config.getint('training_param', 'seed'))

    logger.info('> get dataset')
    dataset_type = config.get('dataset', 'type')
    if dataset_type == 'coco':
        # force to set `use_cache = False`
        train_set = get_coco_dataset(
            insize=parse_size(config.get('model_param', 'insize')),
            image_root=config.get(dataset_type, 'train_images'),
            annotations=config.get(dataset_type, 'train_annotations'),
            min_num_keypoints=config.getint(dataset_type, 'min_num_keypoints'),
            use_cache=False,
            do_augmentation=True,
        )
        test_set = get_coco_dataset(
            insize=parse_size(config.get('model_param', 'insize')),
            image_root=config.get(dataset_type, 'val_images'),
            annotations=config.get(dataset_type, 'val_annotations'),
            min_num_keypoints=config.getint(dataset_type, 'min_num_keypoints'),
            use_cache=False,
        )
    elif dataset_type == 'mpii':
        train_set, test_set = get_mpii_dataset(
            insize=parse_size(config.get('model_param', 'insize')),
            image_root=config.get(dataset_type, 'images'),
            annotations=config.get(dataset_type, 'annotations'),
            train_size=config.getfloat(dataset_type, 'train_size'),
            min_num_keypoints=config.getint(dataset_type, 'min_num_keypoints'),
            use_cache=config.getboolean(dataset_type, 'use_cache'),
            seed=config.getint('training_param', 'seed'),
        )
    else:
        raise Exception('Unknown dataset {}'.format(dataset_type))
    logger.info('dataset type: %s', dataset_type)
    logger.info('training images: %d', len(train_set))
    logger.info('validation images: %d', len(test_set))

    if args.plot_samples > 0:
        for i in range(args.plot_samples):
            data = train_set[i]
            visualize.plot('train-{}.png'.format(i),
                           data['image'], data['keypoints'], data['bbox'], data['is_labeled'], data['edges'])
            data = test_set[i]
            visualize.plot('val-{}.png'.format(i),
                           data['image'], data['keypoints'], data['bbox'], data['is_labeled'], data['edges'])

    logger.info('> load model')
    model = create_model(config, train_set)

    logger.info('> transform dataset')
    train_set = TransformDataset(train_set, model.encode)
    test_set = TransformDataset(test_set, model.encode)

    logger.info('> create iterators')
    train_iter = chainer.iterators.MultiprocessIterator(
        train_set, config.getint('training_param', 'batchsize'),
        n_processes=config.getint('training_param', 'num_process')
    )
    test_iter = chainer.iterators.MultiprocessIterator(
        test_set, config.getint('training_param', 'batchsize'),
        repeat=False, shuffle=False,
        n_processes=config.getint('training_param', 'num_process')
    )

    logger.info('> setup optimizer')
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    logger.info('> setup trainer')
    updater = training.updaters.ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = training.Trainer(updater,
                               (config.getint('training_param', 'train_iter'), 'iteration'),
                               config.get('result', 'dir')
                               )

    logger.info('> setup extensions')
    trainer.extend(
        extensions.LinearShift('lr',
                               value_range=(config.getfloat('training_param', 'learning_rate'), 0),
                               time_range=(0, config.getint('training_param', 'train_iter'))
                               ),
        trigger=(1, 'iteration')
    )

    trainer.extend(extensions.Evaluator(test_iter, model, device=devices['main']))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport([
            'main/loss', 'validation/main/loss',
        ], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.PrintReport([
        'epoch', 'elapsed_time', 'lr',
        'main/loss', 'validation/main/loss',
        'main/loss_resp', 'validation/main/loss_resp',
        'main/loss_iou', 'validation/main/loss_iou',
        'main/loss_coor', 'validation/main/loss_coor',
        'main/loss_size', 'validation/main/loss_size',
        'main/loss_limb', 'validation/main/loss_limb',
    ]))
    trainer.extend(extensions.ProgressBar())

    trainer.extend(extensions.snapshot(filename='best_snapshot'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.snapshot_object(model, filename='bestmodel.npz'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    logger.info('> start training')
    trainer.run()


if __name__ == '__main__':
    import logging
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    main()
