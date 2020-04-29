import numpy as np
import imageio
import math


def RSE(m, r):
    err = 0
    for i in range(len(m)):
        for j in range(len(m[i])):
            err = err + (m[i][j] - r[i][j])**2
    return math.sqrt(err)


def gaussianKernelEquation(x, sigma):
    return 1/(math.pi * (sigma**2)) * math.exp(- x**2 / 2*sigma**2)


def createRangeGaussianComponent(img, n, sigmaR):
    return img


def bilateralFilter(img, n, sigmaS, sigmaR):
    # output_image = img

    # Compute spatial Gaussian component

    # Apply the convolution
    # 1
    # Compute the value of Range Gaussian component
    # Using Gaussian kernel equation
    # sigma = sigmaR
    # x = diference of intensity of neighbor pixel and central pixel
    # gri = G(Ii - I(x,y), sigmaR)
    # i = index of central pixel

    return output_image


filename = str(input()).rstrip()
input_img = imageio.imread(filename)
method = int(input())
save = int(input())

if method == 1:
    n = int(input())
    sigmaS = float(input())
    sigmaR = float(input())
    output_image = bilateralFilter(input_img, n, sigmaS, sigmaR)


elif method == 2:
    c = float(input())
    kernel = int(input())

elif method == 3:
    sigmaRow = float(input())
