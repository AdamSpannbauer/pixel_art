from collections import Counter
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .utils import imread


def tint_recolor(image, color, alpha=.5):
    """tint an image with a given 3 channel color

    :param image: (numpy array) OpenCV image to be recolored
    :param color: (list/tuple/np.array) 3 channel color specification
    :param alpha: (float) alpha value [0-1] for recoloring
    :return: image tinted to be more like input color

    >>> my_image = cv2.imread('path/to/image.png')
    >>> whiter_image = tint_recolor(my_image, [255, 255, 255])
    """
    h, w = image.shape[:2]
    tint_color_patch = np.zeros(h * w * 3, dtype='uint8').reshape(h, w, 3)
    tint_color_patch += np.array(color, dtype='uint8').reshape((1, 1, 3))

    cv2.addWeighted(tint_color_patch, alpha, image, 1 - alpha, 0, image)

    return image


def get_dominant_color(image, k=3, image_processing_size=None):
    """takes an image as input and returns the dominant color in the image as a list

    dominant color is found by performing k means on the pixel colors and returning the centroid
    of the largest cluster processing time is sped up by working with a smaller image; this can be done with the
    image_processing_size param which takes a tuple of image dims as input

    :param image: (numpy array) color OpenCV image to find dominant color of
    :param k: (int) number of color clusters to use in analysis
    :param image_processing_size: (tuple of ints) resize image to this size before processing
    :return: dominant color of image as a len 3 list of floats corresponding to the color space of the input image

    >>> my_image = cv2.imread('path/to/image.png')
    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixels and assign labels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)


def get_dominant_color_hsv(bgr_image, k=3, image_processing_size=(100, 100)):
    """utility wrapper function around get_dominant color for getting hsv dominant color from bgr image

    :param bgr_image: (numpy array) OpenCV image in bgr space to find dominant color of in hsv space
    :param k: (int) number of color clusters to use in analysis
    :param image_processing_size: (tuple of ints) resize image to this size before processing
    :return: dominant color of image as a len 3 list of floats corresponding to [h, s, v]
    """
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    return get_dominant_color(hsv_image, k, image_processing_size)


def get_dominant_color_paths(paths, max_icons=1000, verbose=True):
    """find dir of images dominant colors and return as a pandas df

    :param paths: iterable of string paths to icons
    :param max_icons: (int) max number of icons to process (for debug)
    :param verbose: (bool) should progress messages be printed to screen
    :return: a pandas dataframe with columns: ['path', 'h', 's', 'v'];
            where path is the path to the icon image, hsv are the values of the icon's dominant color

    >>> from imutils.paths import list_images
    >>> icon_paths = list(list_images('my/icon/dir'))
    >>> color_stats_df = get_dominant_color_dir(icon_paths)
    """
    icon_stats_list = []
    for i, path in enumerate(paths):
        if verbose:
            print('processing icon #{}'.format(i))
        if i >= max_icons:
            break
        h, s, v = get_dominant_color_hsv(imread(path))
        icon_stats_list.append((path, h, s, v))

    icon_stat_df = pd.DataFrame(icon_stats_list)
    icon_stat_df.columns = ['path', 'h', 's', 'v']

    return icon_stat_df
