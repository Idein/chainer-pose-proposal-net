import chainer
from chainer import initializers
import chainer.functions as F
import chainer.links as L
import numpy as np


class Convolution2d(chainer.Chain):
    """
    convert pose_estimation.network_base.convolution2d written in tensorflow.contrib.slim
    into Chainer implementation
    """

    def __init__(self, in_channels, out_channels, ksize=3, stride=1):
        super(Convolution2d, self).__init__()
        self.dtype = np.float32
        initialW = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels,
                                        out_channels,
                                        ksize=ksize,
                                        stride=stride,
                                        pad=ksize // 2,
                                        initialW=initialW,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels,
                                           eps=0.001, decay=0.9997)

    def __call__(self, x):
        return F.clipped_relu(self.bn(self.conv(x)), 6.0)


class ExpandedConv(chainer.Chain):

    def __init__(self, expand_ratio, in_channels, out_channels, stride):
        super(ExpandedConv, self).__init__()
        ksize = 3
        self.dtype = np.float32
        self.expand_ratio = expand_ratio
        expanded_channels = int(in_channels * expand_ratio)
        initialW = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        with self.init_scope():
            if expand_ratio != 1:
                self.expand_conv = L.Convolution2D(in_channels,
                                                   expanded_channels,
                                                   ksize=1,
                                                   initialW=initialW,
                                                   nobias=True)
                self.expand_bn = L.BatchNormalization(expanded_channels,
                                                      eps=0.001, decay=0.997)

            self.depthwise_conv = L.DepthwiseConvolution2D(expanded_channels,
                                                           channel_multiplier=1,
                                                           ksize=ksize,
                                                           stride=stride,
                                                           pad=ksize // 2,
                                                           initialW=initialW,
                                                           nobias=True)
            self.depthwise_bn = L.BatchNormalization(expanded_channels,
                                                     eps=0.001, decay=0.9997)
            self.project_conv = L.Convolution2D(expanded_channels,
                                                out_channels,
                                                ksize=1,
                                                initialW=initialW,
                                                nobias=True)
            self.project_bn = L.BatchNormalization(out_channels,
                                                   eps=0.001, decay=0.9997)

    def __call__(self, x):
        h = x
        if self.expand_ratio != 1:
            h = F.clipped_relu(self.expand_bn(self.expand_conv(h)), 6.0)
        h = F.clipped_relu(self.depthwise_bn(self.depthwise_conv(h)), 6.0)
        h = self.project_bn(self.project_conv(h))
        if h.shape == x.shape:
            return h + x
        else:
            return h


class MobilenetV2(chainer.Chain):

    def __init__(self, width_multiplier=1.0, dtype=np.float32):
        super(MobilenetV2, self).__init__()
        min_depth = 8
        self.last_ksize = 1
        self.last_activation = F.sigmoid

        def multiplier(d): return max(int(d * width_multiplier), min_depth)
        with self.init_scope():
            self.conv0 = Convolution2d(None, multiplier(32), stride=2)
            self.conv1 = ExpandedConv(1, multiplier(32), multiplier(16), stride=1)
            self.conv2 = ExpandedConv(6, multiplier(16), multiplier(24), stride=2)
            self.conv3 = ExpandedConv(6, multiplier(24), multiplier(24), stride=1)
            self.conv4 = ExpandedConv(6, multiplier(24), multiplier(32), stride=2)
            self.conv5 = ExpandedConv(6, multiplier(32), multiplier(32), stride=1)
            self.conv6 = ExpandedConv(6, multiplier(32), multiplier(32), stride=1)
            self.conv7 = ExpandedConv(6, multiplier(32), multiplier(64), stride=2)
            self.conv8 = ExpandedConv(6, multiplier(64), multiplier(64), stride=1)
            self.conv9 = ExpandedConv(6, multiplier(64), multiplier(64), stride=1)
            self.conv10 = ExpandedConv(6, multiplier(64), multiplier(64), stride=1)
            self.conv11 = ExpandedConv(6, multiplier(64), multiplier(96), stride=1)
            self.conv12 = ExpandedConv(6, multiplier(96), multiplier(96), stride=1)
            self.conv13 = ExpandedConv(6, multiplier(96), multiplier(96), stride=1)
            self.conv14 = ExpandedConv(6, multiplier(96), multiplier(160), stride=2)
            self.conv15 = ExpandedConv(6, multiplier(160), multiplier(160), stride=1)
            self.conv16 = ExpandedConv(6, multiplier(160), multiplier(160), stride=1)
            self.conv17 = ExpandedConv(6, multiplier(160), multiplier(320), stride=1)

    def __call__(self, x):
        h = self.conv0(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        h = self.conv8(h)
        h = self.conv9(h)
        h = self.conv10(h)
        h = self.conv11(h)
        h = self.conv12(h)
        h = self.conv13(h)
        h = self.conv14(h)
        h = self.conv15(h)
        h = self.conv16(h)
        h = self.conv17(h)
        return h

    @staticmethod
    def prepare(image):
        return image
