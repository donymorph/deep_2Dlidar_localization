
import numpy as np
from scipy import io
from math import pi, cos, sin

import csv

def pol2cart(angles, ranges):
    cart = []
    xs = ranges * -np.cos(angles)
    ys = ranges * np.sin(angles)
    cart = np.array([xs,ys]).T    #N*2 array

    return cart


def ReadData():
    lidar = Lidar(pi,-pi,360)
    f = open("dataset/scan_data.csv", 'r')
    rdr = csv.reader(f)
    ranges = []
    next(rdr)
    for line in rdr:
        ranges_list = list(map(float, line[2].split()))  # Convert ranges to a list of floats
        ranges.append(ranges_list)
    lidar_data = np.array(ranges).reshape(-1,360)
    #lidar_data = np.array(ranges)
    f.close()
    return lidar, lidar_data

class Lidar():
    def __init__(self, angle_min, angle_max, npoints, range_min = 0.23, range_max = 60):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = (angle_max-angle_min)/npoints
        self.npoints = npoints
        self.range_min = range_min
        self.range_max = range_max
        self.scan_time = 0.025
        self.time_increment = 1.736112e-05
        self.angles = np.arange(self.angle_min, self.angle_max, self.angle_increment)

    def ReadAScan(self,lidar_data, scan_id, usableRange):
        ranges = lidar_data[scan_id]

        #Remove points whose range is not so trustworthy
        maxRange = min(self.range_max, usableRange)
        angle = self.angles[(self.range_min<ranges) & (ranges<maxRange)]
        range = ranges[(self.range_min<ranges) & (ranges<maxRange)]

        #Convert from polar coordinates to cartesian coordinates
        scan = pol2cart(angle,range)

        return scan


def v2t(pose):
    # from vector to transform
    tx = pose[0]
    ty = pose[1]
    theta = pose[2]
    transform = np.array([[np.cos(theta), -np.sin(theta), tx],
                          [np.sin(theta), np.cos(theta), ty],
                          [0, 0, 1]])

    return transform

def t2v(T):
    # from transform to vector
    v = np.zeros((3,))
    v[:2] = T[:2,2]
    v[2] = np.arctan2(T[1,0], T[0,0])
    return v

def localToGlobal(pose, scan):
    scanT = np.copy(scan.T)
    frame = np.ones((3, scan.shape[0]))
    frame[0, :] = scanT[0, :]
    frame[1, :] = scanT[1, :]

    transform = v2t(pose)

    scan_global = np.dot(transform, frame)[:2, :] # (2, N) matrix
    scan_global = scan_global.T # (N, 2) matrix

    return scan_global




