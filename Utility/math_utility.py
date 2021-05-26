import numpy as np


def distance_euclid_coord(x1, y1, x2, y2):
    return (((x1-x2)**2) + ((y1-y2)**2)) ** (1/2)


def distance_euclid_pos(pos1, pos2):
    return distance_euclid_coord(pos1[0], pos1[1], pos2[0], pos2[1])


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1 : size//2 + 1, -size//2 + 1 : size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def covariance(x, y):
    return np.average((x-np.average(x)) * (y-np.average(y)))