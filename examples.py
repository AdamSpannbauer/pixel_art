import cv2
from imutils.paths import list_images
import pixel_art

image = cv2.imread('images/github_logo.png')
# icon_paths = list(list_images('path/to/icons/dir'))
icon_paths = list(list_images('icons'))

# create pixelated version of input image
pixelated_image, (nrow, ncol) = pixel_art.pixelate(image, ncol=40, out_width=500)

# create pixelated version of input image from 'icons'
# where the icons are recolored to match the pixelated input image
recolor_icon_image = pixel_art.pixel_icon_recolor(image,
                                                  icon_paths=icon_paths,
                                                  ncol=40,
                                                  out_width=500)

# extract dominant colors from icons for color
# matching process in pixel_art.pixel_icon_match
icon_stats = pixel_art.get_icon_colors(icon_paths)

# create pixelated version of input image from 'icons'
# where the icons are attempted to be matched to the
# color of the pixelated input image
match_icon_image = pixel_art.pixel_icon_match(image,
                                              icon_stats,
                                              ncol=40,
                                              out_width=500)
