from PIL import Image
import numpy as np
import math
from scipy import signal

def boxfilter(n):
    assert n % 2 == 1, "Dimension must be odd"
    return np.full((n,n), 1/(n*n))

def gauss1d(sigma):
    # l is the length of the gaussian filter
    l = math.ceil(sigma * 6)
    if l % 2 == 0:
        l += 1
    edge = math.floor(l/2)
    a = np.arange(-edge, edge+1, 1)
    f = np.exp(-np.power(a, 2)/(2*math.pow(sigma,2)))
    f /= np.sum(f)
    return f

def gauss2d(sigma):
    d1 = gauss1d(sigma)
    d2 = d1[np.newaxis]
    d2t = np.transpose(d2)
    return signal.convolve2d(d2,d2t)

def gaussconvolve2d(array, sigma):
    filt = gauss2d(sigma)
    return signal.convolve2d(array, filt, 'same')

im = Image.open("maru.jpg").convert('L')
#im.show()

im2 = Image.fromarray(np.uint8(gaussconvolve2d(np.asarray(im), 3)))
im2.show()
