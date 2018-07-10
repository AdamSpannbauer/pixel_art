import pixel_art
import cv2

image = cv2.imread('images/github_logo.png')

# create pixelated version of input image
pixelated_image = pixel_art.pixelate(image, ncol=40, out_width=500)

# create pixelated version of input image from 'icons' where the icons are recolored
# to match the pixelated input image
recolor_icon_image = pixel_art.pixel_icon_recolor(image,
                                                  icon_path='path/to/icons/dir',
                                                  ncol=40,
                                                  out_width=500)

# extract dominant colors from icons for color matching process in pixel_art.pixel_icon_match
icon_stats = pixel_art.get_icon_colors('path/to/icons/dir')

# create pixelated version of input image from 'icons' where the icons are attempted
# to be matched to the color of the pixelated input image
match_icon_image = pixel_art.pixel_icon_match(image,
                                              icon_stats,
                                              ncol=40,
                                              out_width=500)
