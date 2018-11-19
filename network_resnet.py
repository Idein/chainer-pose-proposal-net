import os

import chainer
from chainer import initializers
from chainer.backends.cuda import get_array_module
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision import resnet


def process_image(image):
    image = image.copy()
    """
    Taken from https://github.com/chainer/chainer/blob/master/chainer/links/model/vision/resnet.py
    Converts the given image to the numpy array for ResNets.
    """

    # CHW -> HWC
    image = image.transpose((1, 2, 0))
    # RGB -> BGR
    image = image[:, :, ::-1]
    # NOTE: in the original paper they subtract a fixed mean image,
    #       however, in order to support arbitrary size we instead use the
    #       mean pixel (rather than mean image) as with VGG team. The mean
    #       value used in ResNet is slightly different from that of VGG16.
    xp = get_array_module(image)
    image -= xp.array([103.063, 115.903, 123.152])
    # HWC -> CHW
    image = image.transpose((2, 0, 1))
    return image


class AddionalLayer(chainer.Chain):

    def __init__(self, ch):
        super(AddionalLayer, self).__init__()
        initialW = initializers.HeNormal(scale=1.0)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, ch, ksize=3, stride=1, pad=3 // 2, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, ksize=3, stride=1, pad=3 // 2, initialW=initialW)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.1)
        h = F.leaky_relu((self.conv2(h)), slope=0.1)
        return h


class ResNet50(chainer.Chain):

    def __init__(self, **kwargs):
        super(ResNet50, self).__init__()
        self.last_ksize = 1
        self.last_activation = F.sigmoid
        npz_model = os.path.join(
            os.path.dirname(__file__), 'ResNet-50-model.npz')
        pretrained_model = npz_model if os.path.exists(npz_model) else None
        with self.init_scope():
            self.base = L.ResNet50Layers(pretrained_model)
            self.addional_layer = AddionalLayer(ch=512)

    def __call__(self, x):
        h = self.base(x, layers=['res5'])['res5']
        h = self.addional_layer(h)
        return h

    @staticmethod
    def prepare(image):
        return process_image(image)


class BlockA(chainer.Chain):

    def __init__(self, inch, outch, first_stride=2):
        super(BlockA, self).__init__()
        initialW = initializers.HeNormal(scale=1.0)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                inch, outch, ksize=3, stride=first_stride, pad=3 // 2, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(outch)
            self.conv2 = L.Convolution2D(
                outch, outch, ksize=3, stride=1, pad=3 // 2, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(outch)

            self.conv_skip = L.Convolution2D(
                inch, outch, ksize=3, stride=first_stride, pad=3 // 2, initialW=initialW, nobias=True)
            self.bn_skip = L.BatchNormalization(outch)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))

        skip_h = self.bn_skip(self.conv_skip(x))

        return F.relu(h + skip_h)


class BlockB(chainer.Chain):

    def __init__(self, ch):
        super(BlockB, self).__init__()
        initialW = initializers.HeNormal(scale=1.0)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                ch, ch, ksize=3, stride=1, pad=3 // 2, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, ksize=3, stride=1, pad=3 // 2, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))

        return F.relu(x + h)


class Block(chainer.ChainList):

    def __init__(self, n_layer, inch, outch, first_stride=2):
        super(Block, self).__init__()
        self.add_link(BlockA(inch, outch, first_stride=first_stride))
        for _ in range(n_layer - 1):
            self.add_link(BlockB(outch))

    def __call__(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResNet(chainer.Chain):

    def __init__(self, **kwargs):
        n_layers = kwargs.get("n_layers")
        if n_layers == 18:
            blocks = [2, 2, 2, 2]
        elif n_layers == 34:
            blocks = [3, 4, 6, 3]
        else:
            raise ValueError('The n_layers argument should be either 18 or 34,'
                             'but {} was given.'.format(n_layers))
        super(ResNet, self).__init__()
        self.last_ksize = 1
        self.last_activation = F.sigmoid
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, 64, ksize=7, stride=2, pad=7 // 2)
            self.bn1 = L.BatchNormalization(64)
            self.block2 = Block(blocks[0], inch=64, outch=64, first_stride=1)
            self.block3 = Block(blocks[1], inch=64, outch=128)
            self.block4 = Block(blocks[2], inch=128, outch=256)
            self.block5 = Block(blocks[3], inch=256, outch=512)
            self.addional_layer = AddionalLayer(ch=512)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.addional_layer(h)
        return h

    @staticmethod
    def prepare(image):
        return process_image(image)


def main():
    import numpy as np
    img = np.random.random((1, 3, 224, 224)).astype(np.float32)
    resnet = ResNet50()
    assert resnet(img).shape == (1, 512, 7, 7)
    resnet = ResNet(n_layers=18)
    assert resnet(img).shape == (1, 512, 7, 7)
    resnet = ResNet(n_layers=34)
    assert resnet(img).shape == (1, 512, 7, 7)


if __name__ == '__main__':
    main()
