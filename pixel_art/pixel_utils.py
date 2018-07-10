import random
import cv2
import imutils
import numpy as np
from .color_utils import tint_recolor
from .utils import imread


def pixelate(image, ncol=10, out_width=500):
    """apply a pixelated effect to an image

    :param image: (numpy array) OpenCV image to be pixelated
    :param ncol: (int) number of 'pixel' columns in final output image
    :param out_width: width of output image
    :return: tuple of pixelated image, (nrow, ncol);
             nrow & ncol correspond to the grid size of 'pixels' in outputted pixelated image

    >>> my_image = cv2.imread('path/to/image.png')
    >>> pixelated_image, (nrow, ncol) = pixelate(image)
    """
    shrunk = imutils.resize(image, width=ncol)
    nrow, ncol = shrunk.shape[:2]

    pixelated = imutils.resize(shrunk, width=out_width)
    return pixelated, (nrow, ncol)


def create_icon_grid(icon_paths, dims, icon_size):
    """create a grid of icon images given a path to the icon image dir

    :param icon_paths: list of paths for images of icons
    :param dims: (tuple) integers giving the size of the grid in (rows, cols)
    :param icon_size: (int) height/width of icons in grid; assumed to be square
    :return: a numpy array OpenCV image showing a grid of random icons

    >>> from imutils.paths import list_images
    >>> icon_paths = list(list_images('my/icon/dir'))
    >>> icon_grid = create_icon_grid(icon_paths, (10, 20), 6)
    """
    nrow, ncol = dims

    cols = []
    for col in range(ncol):
        print(col)
        col = []
        for row in range(nrow):
            icon = imread(random.choice(icon_paths))
            resized_icon = cv2.resize(icon, (icon_size, icon_size))

            col.append(resized_icon)
        cols.append(np.vstack(col))

    grid = np.hstack(cols)

    return grid


def pixel_icon_recolor(target_image, icon_paths, ncol=200, out_width=500, recolor_alpha=0.7):
    """convert an image to a 'pixelated' image where the pixels are icons

    :param target_image: (numpy array) OpenCV image to be pixelated with icons
    :param icon_paths: list of paths for images of icons
    :param ncol: (int) how many icons wide should output image be
    :param out_width: (int) width of output image; aspect ratio of target_image will be preserved
    :param recolor_alpha: (float) alpha value [0-1] for recoloring icons to resemble pixelated target_image
    :return: a pixelated version of the target image where the pixels are icons from icon_path dir

    >>> from imutils.paths import list_images
    >>> icon_paths = list(list_images('my/icon/dir'))
    >>> image_to_process = cv2.imread('path/to/image.png')
    >>> icon_pixel_image = pixel_icon_recolor(image_to_process, icon_paths=icon_paths)
    """
    pixelated_target, (nrow, ncol) = pixelate(target_image, ncol=ncol, out_width=out_width)
    icon_size = pixelated_target.shape[1] // ncol
    pixelated_target = imutils.resize(pixelated_target, width=icon_size * ncol + 1)

    # create oversize grid to avoid mismatch dims with pixelated target
    icon_grid = create_icon_grid(icon_paths, (nrow * 2, ncol * 2), icon_size)
    icon_grid_crop = icon_grid[0:pixelated_target.shape[0], 0:pixelated_target.shape[1]]

    cv2.addWeighted(pixelated_target, recolor_alpha, icon_grid_crop, 1 - recolor_alpha, 0, icon_grid_crop)
    icon_grid_crop = imutils.resize(icon_grid_crop, width=out_width)

    return icon_grid_crop


def pixel_icon_match(target_image, icon_stat_df,
                     ncol=200, out_width=500, color_tolerance=30,
                     matched_alpha=0.2, unmatched_alpha=0.8,
                     show_progress=True):
    """convert an image to a 'pixelated' image where the pixels are icons

    will attempt to match icons to pixel colors to avoid having to change icons colors

    :param target_image: (numpy array) OpenCV image to be pixelated with icons
    :param icon_stat_df: (pandas df) dataframe created by describe_icons with columns: ['path', 'h', 's', 'v']
    :param ncol: (int) how many icons wide should output image be
    :param out_width: (int) width of output image; aspect ratio of target_image will be preserved
    :param color_tolerance: (float) for icons to match a pixel in target image their dominant color
                            must be plus/minus color_tolerance away from pixel's color (in hsv)
    :param matched_alpha: (float) alpha value [0-1] for recoloring icons to resemble pixelated target_image;
                          this will only be applied to icons within color_tolerance
    :param unmatched_alpha: (float) alpha value [0-1] for recoloring icons to resemble pixelated target_image;
                            this will only be applied to icons not within color_tolerance
    :param show_progress: (bool) should an image/status messages be displayed showing progress of pixelation process
    :return: a pixelated version of the target image where the pixels are icons from icon_path dir
    """

    pixelated_target, (nrow, ncol) = pixelate(target_image, ncol=ncol, out_width=out_width)

    icon_size = pixelated_target.shape[1] // ncol
    pixelated_target = imutils.resize(pixelated_target, width=icon_size * ncol + 1)

    icon_pixel_output = pixelated_target.copy()

    i = 0
    for col in range(ncol):
        col_offset = icon_size * col
        for row in range(nrow):
            i += 1
            row_offset = icon_size * row

            pixel_block = pixelated_target[0 + row_offset:icon_size + row_offset,
                                           0 + col_offset:icon_size + col_offset]

            pixel_color = pixel_block[0, 0]

            np_pixel_color = np.array(pixel_color, dtype='uint8').reshape((1, 1, 3))
            h, s, v = cv2.cvtColor(np_pixel_color, cv2.COLOR_BGR2HSV)[0, 0]

            filtered_icons = icon_stat_df.loc[(h - color_tolerance <= icon_stat_df.h) &
                                              (s - color_tolerance <= icon_stat_df.s) &
                                              (v - color_tolerance <= icon_stat_df.v) &
                                              (icon_stat_df.h <= h + color_tolerance) &
                                              (icon_stat_df.s <= s + color_tolerance) &
                                              (icon_stat_df.v <= v + color_tolerance)]

            sample_paths = filtered_icons['path'].tolist()
            alpha_i = matched_alpha

            if not sample_paths:
                sample_paths = icon_stat_df['path'].tolist()
                alpha_i = unmatched_alpha

            icon = imread(random.choice(sample_paths))
            replacement_pixel = tint_recolor(icon, pixel_color, alpha=alpha_i)
            replacement_pixel = cv2.resize(replacement_pixel, (icon_size, icon_size))

            icon_pixel_output[0 + row_offset:icon_size + row_offset,
                              0 + col_offset:icon_size + col_offset] = replacement_pixel

            if show_progress:
                print('pixel block {} of {}'.format(i, ncol * nrow))
                cv2.imshow('Icon Progress', icon_pixel_output)
                cv2.waitKey(5)

    if show_progress:
        cv2.destroyWindow('Icon Progress')

    icon_pixel_output = imutils.resize(icon_pixel_output, width=out_width)

    return icon_pixel_output
