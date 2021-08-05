# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 01:24:47 2021

@author: admin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

bounding_box_supre = np.empty((0,3), int)
bounding_boxes = np.load('bounding_boxes_2.npy')
# while bounding_boxes.size != 0:
while bounding_boxes.size != 0:
    print('True')
    current_max = bounding_boxes[:,2].max()
    max_index = np.where(bounding_boxes == current_max)[0][0]
    max_x = bounding_boxes[max_index][0]
    max_y = bounding_boxes[max_index][1]
    print(bounding_boxes[max_index])
    bounding_box_supre = np.vstack((bounding_box_supre, bounding_boxes[max_index]))
    bounding_boxes = np.delete(bounding_boxes, (max_index), axis=0)
    copy_bounding_boxes = bounding_boxes.copy()
    for i in range(len(bounding_boxes)):
        x_dist = min((max_x+50), (bounding_boxes[i][0]+50)) - max(max_x, bounding_boxes[i][0])
        y_dist = min((max_y+50), (bounding_boxes[i][1]+50)) - max(max_y, bounding_boxes[i][1])
        if x_dist > 0 and y_dist > 0:
            area = x_dist * y_dist
            if area > 1000:
                area_index = np.where(copy_bounding_boxes == bounding_boxes[i])[0][0]
                copy_bounding_boxes = np.delete(copy_bounding_boxes, (area_index), axis=0)
                # print('Deleted row')
    bounding_boxes = copy_bounding_boxes.copy()
# bounding_box_supre = np.delete(bounding_box_supre, (3,4), axis=0)
    