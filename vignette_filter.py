import numpy as np
import imageio
from bilateral_filter import gaussian_kernel_equation
from laplacian_filter import image_normalization


def generate_1d_gaussian(sigma, size):
    w = np.zeros(size)
    for x in range(0, size):
        w[x] = (gaussian_kernel_equation(x - ((size-1)/2), sigma))

    return w


def vignette_filter(img, sigma_row, sigma_col):
    N, M = img.shape
    
    w_row = np.array([generate_1d_gaussian(sigma_row, N)])
    w_col = np.array([generate_1d_gaussian(sigma_col, M)]).T

    w = w_col.dot(w_row)

    filtred = np.multiply(img, w.T)
    
    return image_normalization(filtred, 255).astype(np.uint8)
