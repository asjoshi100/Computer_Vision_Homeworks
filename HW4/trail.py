# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:02:27 2021

@author: admin
"""

import numpy as np

def im2col(x,ww,hh,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    w,h,c = x.shape
    new_h = (h-hh) // stride + 1
    print('new_h is', new_h)
    new_w = (w-ww) // stride + 1
    print('new_w is', new_w)
    col = np.zeros([new_h*new_w,c*hh*ww])
    col = np.zeros([c*hh*ww, new_h*new_w])

    for i in range(new_w):
       for j in range(new_h):
           # patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           # print('shape of patch is', patch.shape)
           # col[i*new_w+j,:] = np.reshape(patch,-1)
           patch = x[i*stride:i*stride+ww, j*stride:j*stride+hh,...]
           col[:,i*new_h+j] = np.reshape(patch, (1, patch.shape[0]*patch.shape[1]))
    return col


X = np.ones((16,16,1))
Y = im2col(X, 3, 3, 1)