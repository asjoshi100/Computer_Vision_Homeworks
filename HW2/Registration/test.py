# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:31:51 2021

@author: admin
"""
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import pandas as pd
import time


# Old Warp code
# # To do
# img_warped = np.zeros((output_size))
# yedge = np.arange(0, img.shape[0])
# xedge = np.arange(0, img.shape[1])
# xx, yy = np.meshgrid(xedge, yedge)
# z = img[yy, xx]
# f = interpolate.interp2d(xedge, yedge, z, kind='linear')
# start_time = time.time()
# for i in range(output_size[0]):
#     for j in range(output_size[1]):
#         img_location = A @ np.matrix([[j], [i], [1]])
#         img_warped[i, j] = f(img_location.item(0), img_location.item(1))
# print('Time', time.time()- start_time)
# C = A @ np.matrix([[0], [0], [1]])
# x1 = C.item(0)
# y1 = C.item(1)
# C = A @ np.matrix([[output_size[1]], [0], [1]])
# x2 = C.item(0)
# y2 = C.item(1)
# C = A @ np.matrix([[output_size[1]], [output_size[0]], [1]])
# x3 = C.item(0)
# y3 = C.item(1)
# C = A @ np.matrix([[0], [output_size[0]], [1]])
# x4 = C.item(0)
# y4 = C.item(1)
# plt.imshow(img)
# plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'r')
# plt.axis('off')
# plt.plot()
# plt.show()

# # Step 1
# p0 = A - np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
# p = p0
# # Step 2
# temp_pad = np.ones((template.shape[0]+2, template.shape[1]+2)) * np.average(template)
# filter_y = np.reshape(np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]), -1)
# filter_x = np.reshape(np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]), -1)
# x_offset = 1
# y_offset = 1
# temp_pad[x_offset:template.shape[0]+x_offset,y_offset:template.shape[1]+y_offset] = template
# temp_filter_x = np.zeros(template.shape)
# temp_filter_y = np.zeros(template.shape)
# for i in range(template.shape[0]):
#     for j in range(template.shape[1]):
#         b = np.reshape(temp_pad[i:i+3, j:j+3], -1)
#         c_x = np.dot(filter_x, b)
#         c_y = np.dot(filter_y, b)
#         temp_filter_x[i][j] = c_x
#         temp_filter_y[i][j] = c_y
# # Step 3
# dw_dp_u = np.tile(np.arange(template.shape[1]), (template.shape[0], 1))
# dw_dp_v = np.transpose([np.arange(template.shape[0])] * template.shape[1])
# dw_dp_1 = np.ones(template.shape)
# dw_dp_0 = np.zeros(template.shape)
# # Step 4
# temp_x_u = temp_filter_x * dw_dp_u
# temp_x_v = temp_filter_x * dw_dp_v
# temp_x_1 = temp_filter_x * dw_dp_1
# temp_y_u = temp_filter_y * dw_dp_u
# temp_y_v = temp_filter_y * dw_dp_v
# temp_y_1 = temp_filter_y * dw_dp_1
# temp_x_y_w_p = np.array([temp_x_u, temp_x_v, temp_x_1, temp_y_u, temp_y_v, temp_y_1])
# # Step 5
# H = np.zeros((6,6))
# for i in range(6):
#     for j in range(6):
#         H[i,j] = np.sum(temp_x_y_w_p[i] * temp_x_y_w_p[j])
# # Step 6
# initial_warped_img = warp_image(target, A, template.shape)
# error_img = (initial_warped_img - template)
# F = np.array([[np.sum(temp_x_u * error_img)], [np.sum(temp_x_v * error_img)], [np.sum(temp_x_1 * error_img)], [np.sum(temp_y_u * error_img)], [np.sum(temp_y_v * error_img)], [np.sum(temp_y_1 * error_img)]])
# delta_p = np.linalg.inv(H) @ F
# delta_p_w = np.array([[delta_p.item(0)+1, delta_p.item(1), delta_p.item(2)], [delta_p.item(3), delta_p.item(4)+1, delta_p.item(5)], [0,0,1]])
# delta_p = abs(np.sum(delta_p))
# # delta_p = np.linalg.inv(H) @ np.array([[np.sum(temp_x_u)], [np.sum(temp_x_v)], [np.sum(temp_x_1)], [np.sum(temp_y_u)], [np.sum(temp_y_v)], [np.sum(temp_y_1)]])
# # delta_p = abs(np.sum([np.array([np.sum(delta_p[k] * (template - initial_warped_img))]) for k in range(6)]))
# epsilon = 0.0015
# A_p = A
# print(delta_p)
# all_errors = []
# while delta_p > epsilon:
#     print(delta_p)
#     # Step 7
#     target_warp = warp_image(target, A_p, template.shape)
#     # Step 8
#     error_img = (target_warp - template)
#     error_img = (template - target_warp)
#     error_value = np.sum(abs(error_img))
#     print(error_value)
#     # Step 9
#     F = np.array([[np.sum(temp_x_u * error_img)], [np.sum(temp_x_v * error_img)], [np.sum(temp_x_1 * error_img)], [np.sum(temp_y_u * error_img)], [np.sum(temp_y_v * error_img)], [np.sum(temp_y_1 * error_img)]])
#     # Step 10
#     delta_p = np.linalg.inv(H) @ F
#     # Step 11
#     delta_p_w = np.array([[delta_p.item(0)+1, delta_p.item(1), delta_p.item(2)], [delta_p.item(3), delta_p.item(4)+1, delta_p.item(5)], [0,0,1]])        
#     A_p = A_p @ np.linalg.inv(delta_p_w)
#     delta_p = np.sum(abs(delta_p))
#     all_errors.append(error_value)
# A_refined = A_p
# all_errors = np.array(all_errors)
# return A_refined, all_errors

template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
target_list = []
for i in range(4):
    target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
    target_list.append(target)

img = target_list[0]
'''
Let img be the image before warping. We define the "limits" of the data grid for scipy interpolation
from the shape of img
 '''
lim_H, lim_W = img.shape 
'''
With that in place, we create two linear spaces to create a coordinate grid
'''
x = np.linspace(0, lim_W-1, lim_W)
y = np.linspace(0, lim_H-1, lim_H)
points = (y, x) # coordinate grid
 
'''
Now this is where the magic happens. 
Let proj_coord be a (out_H, out_W, 2) array, such that 
proj_coord[i,j,:] returns the (2,) array of x,y coordinates of the original image
that will be mapped to the pixel at location (i,j) of the output image.

That is, proj_coord is the result of applying the inverse mapping.

Flatten proj_coord so that it becomes an (n,2) array, with n being the
number of pixels in the output image.

There remains one last step, which is finding the indices of proj_coord
that DON'T have a corresponding target, i.e. the values that fall outside
the ranges 0<= x <= lim_W-1 and 0<= y <=lim_H-1. 

Suppose you have a list of indices idx of all the indices that satisfy the
inverse mapping. Then, you can just do:
'''

output_size = template.shape

rows, cols = output_size
canvas = np.zeros(output_size)

x_out = np.linspace(0, cols-1, cols)
y_out = np.linspace(0, rows-1, rows)

# meshgrid for projection
yy, xx = np.meshgrid(y_out,x_out)

xy = np.hstack([xx.reshape(rows*cols, 1), yy.reshape(rows*cols, 1), np.ones((rows*cols, 1))])

A = np.matrix([[1.00836, -0.0469219, 539.694], [0.0103494, 1.08298, 107.869], [0, 0, 1]])
proj_coords = A @ xy.T
proj_coords = np.delete(proj_coords, 2, 0)
proj_coords = (proj_coords[1], proj_coords[0])

interp_img = interpolate.interpn(points, img, proj_coords, method='linear') # the idnexing trick [::-1] reverses x,y for image indexing
interp_img = interp_img.reshape(292, 452)
interp_img = interp_img.T
plt.imshow(interp_img, cmap='gray')
plt.axis('off')
plt.show()
# plt.imshow(np.abs(template-img_warped), cmap='jet')
# plt.axis('off')
# plt.show()
