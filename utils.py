#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: utils.py 
@time: 2019/7/4 19:48 
"""

import numpy as np

def im2col(input_data,f_height,f_width,stride = 1,padding = 0):
    """优化卷积算法 将原来矩阵展开"""
    out_channel,in_channel,height,width = input_data.shape

    out_height = 1 + (height + 2 * padding - f_height) // stride
    out_width = 1 + (width + 2 * padding - f_width) // stride

    img = np.pad(input_data,[(0,0),(0,0),(padding,padding),(padding,padding)],'constant')
    col = np.zeros((out_channel,in_channel,f_height,f_width,out_height,out_width))

    for y in range(f_height):
        y_max = y + stride * out_height
        for x in range(f_width):
            x_max = x + stride * out_width
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 坐标轴的变换
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(out_channel * out_height * out_width, -1)

    return col

def col2im(col,input_shape,f_height,f_width,stride = 1,padding = 0):
    """将展开的矩阵还原"""

    N,C,height,width = input_shape

    out_height = (height + 2 * padding - f_height) // stride + 1
    out_width = (width + 2 * padding - f_width) // stride + 1
    col = col.reshape(N, out_height, out_width, C, f_height, f_width).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, height + 2*padding + stride - 1, width + 2*padding + stride - 1))
    for y in range(f_height):
        y_max = y + stride*out_height
        for x in range(f_width):
            x_max = x + stride*out_width
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:height + padding, padding:width + padding]

def translate(y):
    train_y = np.zeros([y.shape[0], 10])
    for i in range(y.shape[0]):
        for j in range(10):
            if y[i] == j:
                train_y[i][j] = 1
    return train_y
