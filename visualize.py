import numpy as np
from PIL import Image, ImageDraw


def plot(filename, array, keypoints, bbox, is_labeled, skeleton):
    array = array.astype('uint8').transpose((1, 2, 0))
    image = Image.fromarray(array)

    draw = ImageDraw.Draw(image)
    for x, y, w, h in bbox:
        draw.rectangle([x, y, x + w, y + h], outline='red')
    for points, labeled in zip(keypoints, is_labeled):
        for y, x in points[labeled]:
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill='cyan')

    image.save(filename)
