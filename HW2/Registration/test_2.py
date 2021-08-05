# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:38:34 2021

@author: admin
"""


from scipy.interpolate import interpn
import numpy as np

def value_func_3d(x, y, z):
    return 2 * x + 3 * y - z
x = np.linspace(0, 5)
y = np.linspace(0, 5)
z = np.linspace(0, 5)
points = (x, y, z)
values = value_func_3d(*np.meshgrid(*points))

point_x = np.matrix([2.21, 4.85, 3.64])
point_y = np.matrix([3.12, 1.93, 3.25])
point_z = np.matrix([1.15, 1.54, 2.92])
point = np.array([2.21, 3.12, 1.15])
point = (point_x, point_y, point_z)
print(interpn(points, values, point))