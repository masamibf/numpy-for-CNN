#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: layers.py 
@time: 2019/7/4 19:44 
"""

import numpy as np
from CNN.losses import mean_squared_loss
import pandas as pd

class Convolution():
    """卷积层"""
#                  权重   偏差  步幅        填充
    def __init__(self,weight,bias,stride = 1,padding = 0):

        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding

        self.x = None

    def _convolution_(self,img,weight,bias = False):

        # 输入数据形状(样本数 通道 宽度 高度)
        batch_size, in_channel, in_height, in_width = img.shape

        # 卷积核形状(输入通道 输出通道 宽度 高度)
        in_channel, f_out_channel, f_height, f_width = weight.shape

        # 输出数据的高和宽
        out_height = 1 + int((in_height + 2 * self.padding - f_height) / self.stride)
        out_width = 1 + int((in_width + 2 * self.padding - f_width) / self.stride)

        # 卷积运算的输出
        out =  np.zeros((batch_size,f_out_channel,out_height,out_width))

        # 输出的形状 batch_size, f_out_channel, out_height, out_width
        for b in np.arange(batch_size):
            for c in np.arange(f_out_channel):
                for h in np.arange(out_height):
                    for w in np.arange(out_width):
                        if bias == True:
                            out[b,c,h // self.stride,w // self.stride] = np.sum(img[b,:,h:h + f_height,w:w + f_width] *
                                                                            weight[:,c]) + self.bias[c]
                        elif bias == False:
                            out[b, c, h // self.stride, w // self.stride] = np.sum(
                                img[b, :, h:h + f_height, w:w + f_width] * weight[:, c])
        return out

    def con_forward(self, x):
        """前向传播 输出卷积结果"""

        # 填充图像
        img = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')

        # 计算卷积 加上偏置
        out = self._convolution_(img,self.weight,bias = True)

        self.x = x

        return out

    def con_backward(self,dout):
        """反向传播"""
        # 卷积核形状(输入通道 输出通道 宽度 高度)
        in_channel, f_out_channel, f_height, f_width = self.weight.shape

        # 卷积核翻转180°
        weight_180 = np.flip(self.weight,(2,3))

        # 交换通道
        weight_180 = np.swapaxes(weight_180,0,1)

        # 填充
        pad_out = np.pad(dout, [(0, 0), (0, 0), (f_height-1,f_height-1), (f_width-1,f_width-1)], 'constant')

        # 前一层的δ
        dout_last = self._convolution_(pad_out,weight_180,bias=False)

        # 交换 输入x的通道 变为 channel batch H W
        x = np.swapaxes(self.x,0,1)

        # 权重的梯度
        dw = self._convolution_(x,dout)

        # 偏置的梯度
        db = np.sum(np.sum(np.sum(dout, axis=-1), axis=-1),axis=0)

        batch = self.x.shape[0]


        return dw/batch,db/batch,dout_last

class Pooling():
    """池化层"""
    def __init__(self,pool,stride = 2,padding = 0):

        # 池化窗口的高和宽
        self.pool_h = pool[0]
        self.pool_w = pool[1]
        self.padding = padding
        self.stride = stride

        self.x = None
        self.list = []

    def pool_forward(self,x):
        """前向传播"""

        # 输入数据的形状
        batch_size,channel,height,width = x.shape

        # 填充
        img = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        self.x = img

        # 输出数据的高和宽
        out_h = (height - self.pool_h) // self.stride + 1
        out_w = (width - self.pool_w) // self.stride + 1

        # 输出数据
        pool_out = np.zeros((batch_size,channel,out_h,out_w))

        for b in np.arange(batch_size):
            for c in np.arange(channel):
                for h in np.arange(out_h):
                    for w in np.arange(out_w):
                        pool_out[b,c,h,w] = np.max(img[b,c,
                                                   self.stride * h : self.stride * h + self.pool_h,
                                                   self.stride * w : self.stride * w + self.pool_w])

        return pool_out

    def pool_backward(self,dout):

        # 前一层数据形状
        batch_size, channel, height, width = self.x.shape

        # 池化层输出数据形状
        _,_,out_h,out_w = dout.shape

        # 前一层的δ
        dout_last = np.zeros((batch_size,channel,height,width))

        for b in np.arange(batch_size):
            for c in np.arange(channel):
                for h in np.arange(out_h):
                    for w in np.arange(out_w):
                        max_index = np.argmax(self.x[b,c,
                                              self.stride * h: self.stride * h + self.pool_h,
                                              self.stride * w: self.stride * w + self.pool_w])

                        h_index = self.stride * h + max_index // self.pool_w
                        w_index = self.stride * w + max_index % self.pool_w

                        dout_last[b,c,h_index,w_index] += dout[b,c,h,w]

        return dout_last

class FullConnect():
    """全连接层"""
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):

        #初始化输入数据：输入层神经元个数、隐藏层神经元个数、输出层神经元个数、学习率
        self.input = input_nodes
        self.hidden = hidden_nodes
        self.output = output_nodes
        self.lr = learning_rate

        # 输入层和隐藏层之间的权重            #高斯分布的均值           #标准差                #大小
        self.weight_i_h = np.random.normal(0.0,             pow(self.hidden,- 0.5),(self.input,self.hidden))
        # 隐藏层和输出层之间的权重
        self.weight_h_o = np.random.normal(0.0,             pow(self.output,- 0.5),(self.hidden,self.output))
        # sigmoid激活函数
        self.sigmoid = lambda x: 1.0/(1 + np.exp(-x*1.0))

    def fc_forward(self, input_data):
        self.input_data = input_data
        #计算隐藏层输入
        hidden_input = np.dot(self.input_data, self.weight_i_h)
         #计算隐藏层输出
        self.hidden_output = self.sigmoid(hidden_input)
        #计算输出层输入
        final_input = np.dot(self.hidden_output, self.weight_h_o)
        #计算输出层输出
        self.final_output = self.sigmoid(final_input)

        return self.final_output

    def fc_backward(self,target):

        #计算在输出层的损失
        delta_h_o = (target -self.final_output) * self.final_output * (1-self.final_output)
        #计算在隐藏层的损失
        delta_i_h = delta_h_o.dot(self.weight_h_o.T) * self.hidden_output * (1-self.hidden_output)
        #计算输入层(展平层flatten)的损失
        delta_flatten = delta_i_h.dot(self.weight_i_h.T)
        #隐藏层_输出层权重更新
        delta_weight_h_o = self.lr * self.hidden_output.T.dot(delta_h_o)
        self.weight_h_o += delta_weight_h_o
        #输入层_隐藏层权重更新
        delta_weight_i_h = self.lr * self.input_data.T.dot(delta_i_h)
        self.weight_i_h += delta_weight_i_h

        return delta_flatten

class Activation():
    """激活层"""

    def relu_forward(self,input):

        self.input = input
        return np.maximum(0,input)

    def relu_backward(self,next_dout):

        dout = np.where(np.greater(self.input, 0), next_dout, 0)
        return dout


# act = Activation()
# input = np.random.randn(2,1,4,4)
# out = act.relu_forward(input)
# print(out,'\n\n')
#
# next_dout = np.random.randn(2,1,4,4)
# dout = act.relu_backward(next_dout)
# print(next_dout,'\n\n',dout)



# fc = FullConnect(784,1000,10,0.01)
#
# # 读取训练数据
#
# mnist_train = pd.read_csv("D://my_python_code//python_with_deep_learning//example//mnist_dataset//mnist_train_100.csv",header=None)
#
# train_x = np.array(mnist_train.values[:,1:785])
#
# y_train = np.array(mnist_train.values[:,0])
#
# def translate(y):
#     train_y = np.zeros([y.shape[0],10])
#     for i in range(y.shape[0]):
#         for j in range(10):
#           if y[i]==j:
#             train_y[i][j] = 1
#     return train_y
#
# train_y = translate(y_train)
#
# for i in range(40):
#     correct_rate = 0.0
#     for j in range(train_x.shape[0]):
#
#         fc_input = (train_x[j]/255).reshape(1,-1)
#
#         out = fc.fc_forward(fc_input)
#         # print(out)
#         y = np.argmax(out,axis=1)
#         if y==y_train[j]:
#             correct_rate += 1
#         fc.fc_backward(train_y[j])
#
#
#     print('预测正确率：',correct_rate/100)
#
#



# img = np.random.randn(1,1,4,4)
#
# pool = Pooling((2,2))
# out = pool.pool_forward(img)
# print(img,'\n\n',out)
# index = pool.pool_backward(out)
# print(index)



# img = np.random.randn(3,3,28,28)
# w = np.random.randn(3,4,3,3)
# b = np.zeros(4)
# y_true = np.ones((3,4,26,26))
# conv = Convolution(w,b)
# #
# for i in range(1000):
#     # 前向
#     next_z = conv.con_forward(img)
#
#     # 反向
#     loss, dy = mean_squared_loss(next_z, y_true)
#     dw,db,_ = conv.con_backward(dy)
#     # 更新梯度
#     w -= 0.0014 * dw
#     b -= 0.0015 * db
#
#     # 打印损失
#     print("step:{},loss:{}".format(i, loss))
#
#     if np.allclose(y_true, next_z):
#         print("yes")
#         break

    # print("min:{},max:{},avg:{}".format(np.min(next_z),np.max(next_z),np.average(next_z)))
