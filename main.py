import numpy as np
import imageio
import math


def RSE(input_img, output_img):
    return np.sqrt(np.sum(np.power(output_img.astype(np.float32) - input_img.astype(np.float32), 2)))


def gaussianKernelEquation(x, sigma):
    return (1.0 / (2.0 * np.pi * (np.power(sigma, 2)))) * (np.exp(-(np.power(x, 2)) / (2.0 * (np.power(sigma, 2)))))


def createSpatialGaussianComponent(sigmaS, n):
    spatial_gaussian = np.zeros((n, n), dtype=np.float32)

    for x in range(0, n):
        for y in range(0, n):
            spatial_gaussian[x][y] = gaussianKernelEquation(
                euclidian_distance(int((n-1)/2), int((n-1) / 2), x, y), sigmaS
            )

    return spatial_gaussian


def euclidian_distance(cx, cy, x, y):
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def bilateral_filter(f, spatial_gaussian, sigmaR):
    N, M = f.shape
    n, m = spatial_gaussian.shape

    a = int((n - 1) / 2)
    b = int((m - 1) / 2)

    # new image to store filtered pixels
    g = np.zeros(f.shape, dtype=np.uint8)

    # calculating padded image
    padded = np.pad(f, max(a, b))

    # for every pixel
    for x in range(a, N+a):
        for y in range(b, M+b):
            # passo 1: calcular a range Gaussian
            range_gaussian = np.zeros((n, m), dtype=np.float32)
            Wp = 0
            If = 0.0
            # para cada vizinho
            for xi in range(x - a, x + a + 1):
                for yi in range(y - b, y + b + 1):
                    # calcular a range gaussian
                    range_gaussian[xi - (x - a)][yi - (y - b)] = gaussianKernelEquation(
                        (padded[xi][yi]*1.0) - (padded[x][y]*1.0), sigmaR)
                    # calcular wi para o pixel correspondente
                    wi = range_gaussian[xi - (x - a)][yi -
                                                      (y - b)] * spatial_gaussian[xi - (x - a)][yi - (y - b)]
                    Wp = float(Wp + wi)
                    If = If + float(wi * padded[xi][yi])
            g[x-a][y-b] = int(If / Wp)

    return g


filename = str(input()).rstrip()
input_img = imageio.imread("imgs/" + filename)
method = int(input())
save = int(input())

if method == 1:
    n = int(input())
    sigmaS = float(input())
    sigmaR = float(input())
    spatial_gaussian = createSpatialGaussianComponent(sigmaS, n)
    output_image = bilateral_filter(input_img, spatial_gaussian, sigmaR)


elif method == 2:
    c = float(input())
    kernel = int(input())

elif method == 3:
    sigmaRow = float(input())

if save == 1:
    imageio.imwrite("output_img.png", output_image)

print(RSE(input_img, output_image))
