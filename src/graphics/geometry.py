"""Homogeneous transforms use image-edge coordinates, not invented character boxes."""
import numpy as np


def rotation_matrix(width, height, degrees):
    degrees %= 360
    return np.asarray({
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, -1, height], [1, 0, 0], [0, 0, 1]],
        180: [[-1, 0, width], [0, -1, height], [0, 0, 1]],
        270: [[0, 1, 0], [-1, 0, width], [0, 0, 1]],
    }[degrees], dtype=float)


def transform_points(points, matrix):
    points = np.c_[np.asarray(points, dtype=float), np.ones(len(points))] @ np.asarray(matrix).T
    return (points[:, :2] / points[:, 2:]).tolist()


def box_polygon(box):
    x0,y0,x1,y1 = box
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


def bounds(points):
    p=np.asarray(points)
    return [float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]


def iou(a,b):
    area=lambda r:max(0,r[2]-r[0])*max(0,r[3]-r[1])
    inter=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    return inter/max(1e-9,area(a)+area(b)-inter)


def coverage(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1]))
