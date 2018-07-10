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
