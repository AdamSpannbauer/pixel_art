import pkg_resources
import numpy as np
import imutils
from skimage import io
import cv2


def imread(path):
    """util function to read images 2 BGR from local or remote path

    :param path: (str) path to image local path or url path to image
    :return: BGR OpenCV image as 3D numpy array

    >>> local_image = imread('path/to/image.jpg')
    >>> remote_image = imread('https://path/to/image.png')
    """
    image = io.imread(path)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def st_watermark(image, watermark_path=None, alpha=.8):
    """place a transparent watermark in the bottom right corner of an image

    :param image: (numpy array) OpenCV image to be watermarked
    :param watermark_path: (str) path to watermark image
    :param alpha: (float) alpha value [0-1] for setting watermark transparency
    :return: input image with transparent watermark in bottom right corner

    >>> my_image = cv2.imread('path/to/image.png')
    >>> watermarked_image = watermark(my_image)
    """
    h, w = image.shape[:2]
    image_overlay = image.copy()

    if watermark_path is None:
        watermark_path = pkg_resources.resource_filename('pixel_art', 'data/st_watermark.png')

    st_watermark = cv2.imread(watermark_path)
    st_watermark = imutils.resize(st_watermark, width=w // 4)

    overlay = np.zeros((h, w, 3), dtype="uint8")
    overlay[h - st_watermark.shape[0] - 5:h - 5,
            w - st_watermark.shape[1] - 5:w - 5] = st_watermark

    watermark_inds = np.where((overlay[:, :, :3] > 20) & (overlay[:, :, :3] < 220))
    image_overlay[watermark_inds] = overlay[watermark_inds]

    cv2.addWeighted(image_overlay, alpha, image, 1 - alpha, 0, image)

    return image
