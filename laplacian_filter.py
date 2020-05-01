import numpy as np
import imageio
import math


def image_convolution(img, convolution_filter):
    n, m = convolution_filter.shape
    N, M = img.shape

    a = int((n-1)/2)
    b = int((m-1)/2)

    # creating a zero padded image
    img_padded = np.pad(img, int(n/2))

    # image after filter
    image_filtered = np.zeros(img.shape, dtype=np.uint8)

    # for all pixels in the original image
    for x in range(a, N+a):
        for y in range(b, M+b):
            # region centered at x,y
            region_to_be_filtered = img_padded[x-a:x+a+1, y-b:y+b+1]

            # convoluting filter and image region
            image_filtered[x-a][y-b] = np.sum(np.multiply(
                region_to_be_filtered, convolution_filter)).astype(np.uint8)

    return image_filtered


# Normalizing a image
# max_scale_val is equal to the maximum value in a certain scale
# 8 bit greyscale maximum is 255
def image_normalization(img, max_scale_val):
    normilzed_image = ((img - np.min(img)) * max_scale_val) / np.max(img)
    return normilzed_image


# Adds two images values, multiplying the filtered one by c
def filter_addition(img, filtered, c):
    image_added = img + filtered * c
    return image_added


def laplacian_filter(img, c, kernel):
    # Image padded convolution with given kernel
    convoluted_image = image_convolution(img, kernel)

    # Imag scaling with normalization function
    normalized_image = image_normalization(convoluted_image, 255)

    # Adding filtered image * c with original image
    image_added = filter_addition(img, normalized_image, c)

    # Scaling final image and returning it
    return image_normalization(image_added, 255)
