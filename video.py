import configparser
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
import time

import chainer
from chainercv.utils import non_maximum_suppression
import cv2
import numpy as np
from PIL import ImageDraw, Image

from predict import COLOR_MAP
from predict import estimate, draw_humans, create_model
from utils import parse_size


def main():
    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = create_model(config)

    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)

    logger.info('camera will capture {} FPS'.format(cap.get(cv2.CAP_PROP_FPS)))
    if os.path.exists('mask.png'):
        mask = Image.open('mask.png')
        mask = mask.resize((200, 200))
    else:
        mask = None

    fps_time = 0
    degree = 0
    while cap.isOpened():
        degree += 5
        degree = degree % 360
        ret_val, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, model.insize)
        with chainer.using_config('autotune', True):
            humans = estimate(model,
                              image.transpose(2, 0, 1).astype(np.float32))
        pilImg = Image.fromarray(image)
        pilImg = draw_humans(
            model.keypoint_names,
            model.edges,
            pilImg,
            humans,
            mask=mask.rotate(degree) if mask else None
        )
        img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
        msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
        msg += ' ' + config.get('model_param', 'model_name')
        cv2.putText(img_with_humans, 'FPS: % f' % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        img_with_humans = cv2.resize(img_with_humans, (3 * model.insize[0], 3 * model.insize[1]))
        cv2.imshow('Pose Proposal Network' + msg, img_with_humans)
        fps_time = time.time()
        # press Esc to exit
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()
