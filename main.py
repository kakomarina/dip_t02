# Marcelo de Moraes - 
# Marina Sleutjes Kako - 9763151
# Processamento de Imagens - scc0251 2020-1

import numpy as np
import imageio
import math

from bilateral_filter import *
from laplacian_filter import *
from vignette_filter import *

def RSE(input_img, output_img):
    return np.sqrt(np.sum(np.power(output_img.astype(np.float32) - input_img.astype(np.float32), 2)))


filename = str(input()).rstrip()
input_img = imageio.imread(filename)
method = int(input())
save = int(input())

if method == 1:
    n = int(input())
    sigmaS = float(input())
    sigmaR = float(input())
    spatial_gaussian = create_spatial_gaussian_component(sigmaS, n)
    output_image = bilateral_filter(input_img, spatial_gaussian, sigmaR)

elif method == 2:
    c = float(input())
    kernel_type = int(input())

    # creating kernel1
    if kernel_type == 1:
        kernel = np.matrix([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # creating kernel2
    elif kernel_type == 2:
        kernel = np.matrix([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    output_image = laplacian_filter(input_img, c, kernel)

elif method == 3:
    sigma_row = float(input())
    sigma_col = float(input())

    output_image = vignette_filter(input_img, sigma_row, sigma_col)

if save == 1:
    imageio.imwrite("output_img.png", output_image)

print(RSE(input_img, output_image))
