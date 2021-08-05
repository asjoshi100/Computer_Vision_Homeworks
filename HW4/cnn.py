import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main
from scipy.special import softmax


def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    if im_train.shape[1]%batch_size == 0:
        size = im_train.shape[1]/batch_size
    else:
        size = np.floor(im_train.shape[1]/batch_size) + 1
    (size1,size2) = im_train.shape
    mini_batch_x = np.zeros((int(size), size1, batch_size))
    mini_batch_y = np.zeros((int(size), 10, batch_size))
    train_size = size2
    labels_all = np.zeros((10, train_size))
    for i in range(train_size):
        labels_all[label_train[0,i],i] = 1
    rand_num = np.random.permutation(train_size)
    for i in range(int(size)):
        mini_batch_x[i] = im_train[:, rand_num[i*batch_size:(i+1)*batch_size]]
        mini_batch_y[i] = labels_all[:, rand_num[i*batch_size:(i+1)*batch_size]]
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    y = w@x + b
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_dx = w.T @ dl_dy
    dl_dw = dl_dy @ (x.T)
    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    l = np.square(np.linalg.norm(y - y_tilde))
    dl_dy = -(2*(y - y_tilde))
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
    soft_x = softmax(x)
    l = (-y.T) @ np.log(soft_x)
    dl_dy = (soft_x - y)
    return l, dl_dy

def relu(x):
    # TO DO
    zeros = np.zeros(x.shape)
    y = np.maximum(zeros, x)
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy * (y >= 0)
    return dl_dx

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
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])
    col = np.zeros([c*hh*ww, new_h*new_w])

    for i in range(new_w):
       for j in range(new_h):
           patch = x[i*stride:i*stride+ww, j*stride:j*stride+hh,...]
           col[:,i*new_h+j] = np.reshape(patch, (1, patch.shape[0]*patch.shape[1]), order='F')
    return col

def conv(x, w_conv, b_conv):
    # TO DO
    Hei, Wid, Dep = x.shape
    w_conv_Hei, w_conv_Wid, w_conv_dep1, w_conv_dep2 = w_conv.shape
    y = np.zeros((Hei, Wid, w_conv_dep2))
    pad_lay = 1
    X = np.reshape(np.pad(x, (pad_lay, pad_lay))[:,:,1], (16,16,1), order='F')
    X_cool = im2col(X, 3, 3, 1)
    w_new = np.reshape(w_conv, (9,3), order='F')
    y_cool = X_cool.T @ w_new
    y = np.reshape(y_cool, (14, 14, 3), order='F')
    for num_Bias in range(w_conv_dep2):
        y[:,:,num_Bias] = y[:,:,num_Bias] + b_conv[num_Bias]
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    Hei, Wid, Dep = x.shape
    w_conv_Hei, w_conv_Wid, w_conv_dep1, w_conv_dep2 = w_conv.shape
    pad_lay = 1
    X = np.reshape(np.pad(x, (pad_lay, pad_lay))[:,:,1], (16,16,1), order='F')
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)
    X_cool = im2col(X, 3, 3, 1)
    new_dl_dy = np.reshape(dl_dy, (196, 3), order='F')
    new_dl_dw = X_cool @ new_dl_dy
    dl_dw = np.reshape(new_dl_dw, (3, 3, 1, 3), order='F')
    for i in range(w_conv_dep2):
        dl_db[i, 0] = np.sum(np.sum(dl_dy[:,:,i])) 
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    Hei, Wid, Dep = x.shape
    stride = 2
    y = np.zeros((Hei//stride, Wid//stride, Dep))
    for i in range(0, Hei, stride):
        for j in range(0, Wid, stride):
            for k in range(Dep):
                batch = [x[i,j,k], x[i+stride-1,j,k], x[i,j+stride-1,k], x[i+stride-1,j+stride-1,k]]
                y[(i+stride-1)//2,(j+stride-1)//2,k] = np.max(batch)
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    Hei, Wid, Dep = x.shape
    dl_dx = np.zeros((Hei, Wid, Dep))
    stride = 2
    for i in range(0, Hei, stride):
        for j in range(0, Wid, stride):
            for k in range(Dep):
                if x[i,j,k] == y[(i+stride-1)//2,(j+stride-1)//2,k]:
                    dl_dx[i,j,k] = dl_dy[(i+stride-1)//2, (j+stride-1)//2,k]
                elif x[i+stride-1,j,k] == y[(i+stride-1)//2,(j+stride-1)//2,k]:
                    dl_dx[i+stride-1,j,k] = dl_dy[(i+stride-1)//2,(j+stride-1)//2,k]
                elif x[i,j+stride-1,k] == y[(i+stride-1)//2,(j+stride-1)//2,k]:
                    dl_dx[i,j+stride-1,k] = dl_dy[(i+stride-1)//2,(j+stride-1)//2,k]
                else:
                    dl_dx[i+stride-1,j+stride-1,k] = dl_dy[(i+stride-1)//2,(j+stride-1)//2,k]
    return dl_dx


def flattening(x):
    # TO DO
    y = x.flatten('F')
    y = np.reshape(y, (y.shape[0], -1), order='F')
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    dl_dx = np.reshape(dl_dy, x.shape, order='F')
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    w = np.random.normal(0,1,[10,196])
    b = np.random.normal(0,1,[10,1])
    batches = mini_batch_x.shape[0]
    batch_size = mini_batch_x[0].shape[1]
    learning_rate = 0.01
    decay_rate = 0.5
    decay_interval = 1000
    num_iteration = 10000
    btc = 0
    loss_iter = 0
    for iteration in range(num_iteration):
        if iteration % decay_interval == 0:
            learning_rate = decay_rate * learning_rate
        dL_dW = 0
        dL_dB = 0
        for i in range(batch_size):
            x = np.reshape(mini_batch_x[btc,:,i], (196, -1), order='F')
            y = fc(x, w, b)
            l, dl_dy = loss_euclidean(y, np.reshape(mini_batch_y[btc,:,i], (10, -1), order='F'))
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            dL_dW = dL_dW + dl_dw
            dL_dB = dL_dB + dl_db
        btc = btc + 1
        if btc >= batches:
            btc = 0
        w = w - (learning_rate*dL_dW)/batch_size
        b = b - (learning_rate*dL_dB)/batch_size
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    w = np.random.normal(0,1,[10,196])
    b = np.random.normal(0,1,[10,1])
    batches = mini_batch_x.shape[0]
    batch_size = mini_batch_x[0].shape[1]
    learning_rate = 0.2
    decay_rate = 0.9
    decay_interval = 1000
    num_iteration = 10000
    btc = 0
    for iteration in range(num_iteration):
        if iteration % decay_interval == 0:
            learning_rate = decay_rate * learning_rate
        dL_dW = 0
        dL_dB = 0
        for i in range(batch_size):
            x = np.reshape(mini_batch_x[btc,:,i], (196, -1), order='F')
            y = fc(x, w, b)
            l, dl_dy = loss_cross_entropy_softmax(y, np.reshape(mini_batch_y[btc,:,i], (10, -1), order='F'))
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            dL_dW = dL_dW + dl_dw
            dL_dB = dL_dB + dl_db
        btc = btc + 1
        if btc >= batches:
            btc = 0
        w = w - (learning_rate*dL_dW)/batch_size
        b = b - (learning_rate*dL_dB)/batch_size
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    w1 = np.random.rand(30, 196)
    b1 = np.random.rand(30, 1)
    w2 = np.random.rand(10, 30)
    b2 = np.random.rand(10, 1)
    batches = mini_batch_x.shape[0]
    batch_size = mini_batch_x[0].shape[1]
    learning_rate = 0.05
    decay_rate = 0.86
    decay_interval = 2000
    num_iteration = 10000
    num_iteration = 15000
    btc = 0
    for iteration in range(num_iteration):
        if iteration % decay_interval == 0:
            learning_rate = decay_rate * learning_rate 
        dL_dW1 = 0
        dL_dB1 = 0
        dL_dW2 = 0
        dL_dB2 = 0
        for i in range(batch_size):
            x = np.reshape(mini_batch_x[btc,:,i], (196, -1), order='F')
            y_fc_1 = fc(x, w1, b1)
            y_relu = relu(y_fc_1)
            y_fc_2 = fc(y_relu, w2, b2)
            l, dl_dy = loss_cross_entropy_softmax(y_fc_2, np.reshape(mini_batch_y[btc,:,i], (10, -1), order='F'))
            dl_dx2, dl_dw2, dl_db2 = fc_backward(dl_dy, y_relu, w2, b2, y_fc_2)
            dl_dy1 = relu_backward(dl_dx2, y_fc_1, y_relu)
            dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dy1, x, w1, b1, y_fc_1)
            dL_dW1 = dL_dW1 + dl_dw1
            dL_dB1 = dL_dB1 + dl_db1
            dL_dW2 = dL_dW2 + dl_dw2
            dL_dB2 = dL_dB2 + dl_db2  
        btc = btc + 1
        if btc >= batches:
            btc = 0
        w1 = w1 - (learning_rate*dL_dW1)/batch_size
        b1 = b1 - (learning_rate*dL_dB1)/batch_size
        w2 = w2 - (learning_rate*dL_dW2)/batch_size
        b2 = b2 - (learning_rate*dL_dB2)/batch_size
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    w_conv = np.random.rand(3,3,1,3)
    b_conv = np.random.rand(3,1)
    w_fc = np.random.rand(10,147)
    b_fc = np.random.rand(10,1)
    btc = 0
    batches = mini_batch_x.shape[0]
    batch_size = mini_batch_x[0].shape[1]
    learning_rate = 0.1
    decay_rate = 0.8
    decay_interval = 1000
    num_iteration = 20000
    for iteration in range(num_iteration):
        print('no of iteration', iteration)
        if iteration % decay_interval == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw_conv = 0
        dL_db_conv = 0
        dL_dw_fc = 0
        dL_db_fc = 0
        loss_iter = 0
        for i in range(batch_size):
            x = np.reshape(mini_batch_x[btc,:,i], (196, -1), order='F')
            X = np.reshape(x, (14, 14, 1), order='F')
            convoluted_X = conv(X, w_conv, b_conv)
            y_relu = relu(convoluted_X)
            pooled_y = pool2x2(y_relu)
            flattened = flattening(pooled_y)
            y_fc = fc(flattened, w_fc, b_fc)
            L, dl_dy_FC = loss_cross_entropy_softmax(y_fc, np.reshape(mini_batch_y[btc,:,i], (10, -1), order='F'))
            dl_dflattened, dl_dw_fc, dl_db_fc = fc_backward(dl_dy_FC, flattened, w_fc, b_fc, y_fc)
            dl_dx_pooled = flattening_backward(dl_dflattened, pooled_y, flattened)
            dl_dx_pooled_back = pool2x2_backward(dl_dx_pooled, y_relu, pooled_y)
            dl_dx_relu_back = relu_backward(dl_dx_pooled_back, convoluted_X, y_relu)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dx_relu_back, X, w_conv, b_conv, convoluted_X)
            dL_dw_conv = dL_dw_conv + dl_dw_conv
            dL_db_conv = dL_db_conv + dl_db_conv
            dL_dw_fc = dL_dw_fc + dl_dw_fc
            dL_db_fc = dL_db_fc + dl_db_fc
            loss_iter = loss_iter + 1
        btc = btc + 1
        if btc >= batches:
            btc = 0
        w_conv = w_conv - (learning_rate*dL_dw_conv)/batch_size
        b_conv = b_conv - (learning_rate*dL_db_conv)/batch_size
        w_fc = w_fc - (learning_rate*dL_dw_fc)/batch_size
        b_fc = b_fc - (learning_rate*dL_db_fc)/batch_size
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



