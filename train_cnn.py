#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: train_cnn.py 
@time: 2019/7/6 14:07 
"""

from CNN.layers import *
from CNN.utils import *
from CNN.losses import *
import pandas as pd

# ==================== 参数初始化 ====================
batch = 1
in_channel = 1
input_size = (28,28)

out_channel = 4
filter_size = (3,3)
stride = 1
padding = 0

pool_size = (2,2)

fc_lr = 0.01
conv_lr = 0.001

conv_weight = np.random.randn(in_channel,out_channel,filter_size[0],filter_size[1])
conv_bias = np.zeros(out_channel)

flatten_h = (1 + (input_size[0] + 2*padding - filter_size[0]) // stride) // pool_size[0]
flatten_w = (1 + (input_size[1] + 2*padding - filter_size[1]) // stride) // pool_size[1]
flatten_shape = (batch,out_channel,flatten_h,flatten_w)

fc_input = flatten_shape[1] * flatten_shape[2] * flatten_shape[3]

fc_hidden = 800
fc_output = 10
# ===================================================

conv = Convolution(conv_weight,conv_bias,stride,padding)
act = Activation()
pool = Pooling(pool_size)
fc = FullConnect(fc_input,fc_hidden,fc_output,fc_lr)

def cnn_forward(input):
	"""前向传播"""

	conv_out = conv.con_forward(input)

	relu_out = act.relu_forward(conv_out)

	pool_out = pool.pool_forward(relu_out)
	pool_out = pool_out

	flatten_out = pool_out.reshape(pool_out.shape[0],pool_out.shape[1]*
	                               pool_out.shape[2] * pool_out.shape[3]
	                               )

	fc_out = fc.fc_forward(flatten_out)

	return fc_out

def cnn_backward(target):
	"""反向传播"""

	d_flatten = fc.fc_backward(target)

	d_pool = d_flatten.reshape(flatten_shape)

	d_relu = pool.pool_backward(d_pool)

	d_conv = act.relu_backward(d_relu)

	dw,db,_ = conv.con_backward(d_conv)

	return dw,db,_


if __name__ == '__main__':

	# 读取训练数据

	mnist_train = pd.read_csv("D://my_python_code//python_with_deep_learning//example//mnist_dataset//mnist_train_100.csv",header=None)
	train_x = np.array(mnist_train.values[ : ,1:785])
	y_train = np.array(mnist_train.values[ : ,0])
	train_y = translate(y_train)

	# 读取测试数据
	mnist_test = pd.read_csv('D://my_python_code//python_with_deep_learning//example//mnist_dataset//mnist_test.csv',header=None)
	test_x = np.array(mnist_test.values[ : ,1:785])
	y_test = np.array(mnist_test.values[ : ,0])
	test_y = translate(y_test)

	# 训练

	for i in range(20):
		correct_rate = 0.0
		for j in range(train_x.shape[0]):

			input = (train_x[j]/255).reshape(1,-1)
			img = input.reshape(batch,in_channel,input_size[0],input_size[1])

			out = cnn_forward(img)
			y = np.argmax(out,axis=1)

			if y == y_train[j]:
				correct_rate += 1
			dw,db,_ = cnn_backward(train_y[j])
			conv_weight -= conv_lr * dw
			conv_bias -= conv_lr * db

		print('step',i,'训练精度:', correct_rate / 100)

	# 测试
	test_acc = 0.0
	for i in range(100):
		test_img = (test_x[i]/255).reshape(batch,in_channel,input_size[0],input_size[1])
		test_out = cnn_forward(test_img)
		y_predict = np.argmax(test_out,axis=1)
		if y_predict == y_test[i]:
			test_acc += 1
		print('预测值',y_predict,'真实值',y_test[i])

	print('测试精度:',test_acc/100)

