import os
import numpy as np
from chainer.dataset import DatasetMixin
from chainercv import utils
import chainercv.transforms as transforms
import numpy as np

from augment import rotate


class KeypointDataset2D(DatasetMixin):

    def __init__(self,
                 dataset_type,
                 insize,
                 keypoint_names,
                 edges,
                 flip_indices,
                 keypoints,
                 bbox,
                 is_visible,
                 is_labeled,
                 image_paths,
                 image_root='.',
                 use_cache=False,
                 do_augmentation=False
                 ):
        self.dataset_type = dataset_type
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.flip_indices = flip_indices
        self.keypoints = keypoints  # [array of y,x]
        self.bbox = bbox  # [x,y,w,h]
        self.is_visible = is_visible
        self.is_labeled = is_labeled
        self.image_paths = image_paths
        self.image_root = image_root
        self.do_augmentation = do_augmentation
        self.use_cache = use_cache
        self.cached_samples = [None] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def transform(self, image, keypoints, bbox, is_labeled):
        _, H, W = image.shape
        # PCA Lighting
        image = transforms.pca_lighting(image, sigma=5)

        # Random rotate
        degree = np.random.uniform(-40, 40)
        image, keypoints, bbox = rotate(image, keypoints, bbox, degree)
        # Random flip
        image, param = transforms.random_flip(image, x_random=True, return_param=True)
        if param['x_flip']:
            keypoints = [
                transforms.flip_point(points, (H, W), x_flip=True)[self.flip_indices]
                for points in keypoints
            ]

            is_labeled = [label[self.flip_indices] for label in is_labeled]

            new_bbox = []
            for x, y, w, h in bbox:
                [[y, x]] = transforms.flip_point(np.array([[y, x + w]]), (H, W), x_flip=True)
                new_bbox.append([x, y, w, h])
            bbox = new_bbox

        # Random resize
        scalew, scaleh = np.random.uniform(1.0, 2.0, 2)
        resizeW, resizeH = int(W * scalew), int(H * scalew)
        image, keypoints, bbox = self.resize(image, keypoints, bbox, (resizeH, resizeW))

        # Random crop
        image, param = transforms.random_sized_crop(image,
                                                    scale_ratio_range=(0.5, 5), return_param=True)
        keypoints = [
            transforms.translate_point(points,
                                       x_offset=-param['x_slice'].start,
                                       y_offset=-param['y_slice'].start
                                       )
            for points in keypoints
        ]
        new_bbox = []
        for x, y, w, h in bbox:
            new_bbox.append([x - param['x_slice'].start, y - param['y_slice'].start, w, h])
        bbox = new_bbox

        return image, keypoints, bbox, is_labeled

    def resize(self, image, keypoints, bbox, size):
        _, h, w = image.shape
        new_h, new_w = size

        image = transforms.resize(image, (new_h, new_w))
        keypoints = [
            transforms.resize_point(points, (h, w), (new_h, new_w))
            for points in keypoints
        ]
        new_bbox = []
        for x, y, bw, bh in bbox:
            [[y, x]] = transforms.resize_point(np.array([[y, x]]), (h, w), (new_h, new_w))
            bw *= new_w / w
            bh *= new_h / h
            new_bbox.append([x, y, bw, bh])
        return image, keypoints, new_bbox

    def get_example(self, i):
        w, h = self.insize

        if self.use_cache and self.cached_samples[i] is not None:
            image, keypoints, bbox, is_labeled = self.cached_samples[i]
        else:
            path = os.path.join(self.image_root, self.image_paths[i])
            image = utils.read_image(path, dtype=np.float32, color=True)
            keypoints = self.keypoints[i]
            bbox = self.bbox[i]
            is_labeled = self.is_labeled[i]

            image, keypoints, bbox = self.resize(image, keypoints, bbox, (h, w))
            if self.use_cache:
                self.cached_samples[i] = image, keypoints, bbox, is_labeled

        image = image.copy()
        keypoints = keypoints.copy()
        bbox = bbox.copy()
        is_labeled = is_labeled.copy()

        if self.do_augmentation:
            image, keypoints, bbox, is_labeled = self.transform(image, keypoints, bbox, is_labeled)
            image, keypoints, bbox = self.resize(image, keypoints, bbox, (h, w))

        return {
            'path': self.image_paths[i],
            'keypoint_names': self.keypoint_names,
            'edges': self.edges,
            'image': image,
            'keypoints': keypoints,
            'bbox': bbox,
            'is_labeled': is_labeled,
            'dataset_type': self.dataset_type,
        }
