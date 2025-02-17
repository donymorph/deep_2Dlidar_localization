
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
    """
    Converts a 2D pose [x, y, theta] into a homogeneous transformation matrix (3x3).
    
    Args:
        pose (array-like): [x, y, theta] where
            x, y: Translation components
            theta: Rotation angle (in radians)
    
    Returns:
        numpy.ndarray: 3x3 homogeneous transformation matrix.
    """
    x, y, theta = pose
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    transform = np.array([
        [cos_theta, -sin_theta, x],
        [sin_theta,  cos_theta, y],
        [0,         0,         1]
    ])

    return transform

def t2v(T):
    """
    Converts a homogeneous transformation matrix (3x3) back to a 2D pose [x, y, theta].
    
    Args:
        T (numpy.ndarray): 3x3 transformation matrix
    
    Returns:
        numpy.ndarray: [x, y, theta] pose
    """
    x, y = T[:2, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])  # More robust than atan(y/x)

    return np.array([x, y, theta])

def localToGlobal(pose, scan):
    scanT = np.copy(scan.T)
    frame = np.ones((3, scan.shape[0]))
    frame[0, :] = scanT[0, :]
    frame[1, :] = scanT[1, :]

    transform = v2t(pose)

    scan_global = np.dot(transform, frame)[:2, :] # (2, N) matrix
    scan_global = scan_global.T # (N, 2) matrix

    return scan_global




