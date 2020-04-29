import numpy as np
import imageio
import math


def RSE(m, r):
    err = 0
    for i in range(len(m)):
        for j in range(len(m[i])):
            err = err + (m[i][j] - r[i][j]) ** 2
    return math.sqrt(err)


def gaussianKernelEquation(x, sigma):
    return (1.0 / (2.0 * math.pi * (sigma ** 2.0))) * (math.exp(-(x ** 2) / 2.0 * (sigma ** 2)))


def createSpatialGaussianComponent(sigmaS, n):
    spatial_gaussian = np.zeros((n, n), dtype=np.float32)

    for x in range(0, n):
        for y in range(0, n):
            spatial_gaussian[x][y] = gaussianKernelEquation(
                euclidian_distance(n / 2, n / 2, x, y), sigmaS
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
    g = np.zeros(f.shape, dtype=np.float32)

    # calculating padded image
    padded = np.pad(f, max(a, b))

    # for every pixel
    for x in range(a, N+a):
        for y in range(b, M+b):
            # passo 1: calcular a range Gaussian
            range_gaussian = np.zeros((n, m), dtype=np.float64)
            Wp = 0

            # para cada vizinho
            for xi in range(x - a, x + a + 1):
                for yi in range(y - b, y + b + 1):
                    # calcular a range gaussian
                    range_gaussian[xi - (x - a)][yi - (y - b)] = gaussianKernelEquation(
                        (padded[xi][yi]*1.0) - (padded[x][y]*1.0), sigmaR)
                    wi = range_gaussian[xi - (x - a)][yi -
                                                      (y - b)] * spatial_gaussian[xi - (x - a)][yi - (y - b)]
                    Wp = Wp + wi
                    g[x-a][y-b] = g[x-a][y-b] + wi * padded[xi][yi]
            g[x-a][y-b] = int(g[x-a][y-b]*1.0 / Wp)

    return g.astype(np.uint8)


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


print(RSE(input_img.astype(np.float32), output_image.astype(np.float32)))
