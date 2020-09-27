# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:36:37 2020

@author: asus
"""

#from __future__ import print_function
#import torch
## =============================================================================
## Basic tensor
## 
## x = torch.empty(4,4)
## y = torch.rand(4,4)
## z = torch.zeros(4,4)
## a = torch.tensor([12,35])
## b = torch.ones(4,5,dtype = torch.double)
## c = torch.randn_like(x, dtype=torch.float)
## 
## d = torch.add(y,z)
## print(x.add_(y+z),'\n',d) 
## #x.view(2,2,4)  reshape the matrix
## 
## #tensor and numpy, change at the same time
## e = d.numpy()
## f = torch.from_numpy(e)
## 
## =============================================================================
##Autograd
#x = torch.ones(4,4,requires_grad = True)
#print(x,'\n')
#y = (x+1)*5
#print(y)
##backward()用于计算倒数

#一个简单的神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F

class simple_net(nn.Module):
#一个nn.Module包含各个层和一个forward(input)方法，该方法返回output。
    def __init__(self):
        super(simple_net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)#Linear为全连接层
    
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = simple_net()
print(net)

#学习参数返回：net.parameters()
        
input = torch.randn(1,1,32,32)
out = net(input)
print(out)

#清零所有参数的梯度缓存，然后进行随机梯度的反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

#损失函数
output = net(input)
target = torch.randn(10)  # 本例子中使用模拟数据
target = target.view(1, -1)  # 使目标值与数据值尺寸一致,output不改变数据，但是改变数据的形状
criterion = nn.MSELoss()

loss = criterion(output, target)
#print(loss)

#反向传播
#net.zero_grad()     # 清零所有参数(parameter）的梯度缓存
#
#print('conv1.bias.grad before backward')
#print(net.conv1.bias.grad)
#
#loss.backward()
#
#print('conv1.bias.grad after backward')
#print(net.conv1.bias.grad)

#更新权重
import torch.optim as optim

# 创建优化器(optimizer），还包括SGD、Nesterov-SGD、Adam、RMSProp
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练的迭代中：
optimizer.zero_grad()   # 清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新参数
print(output)




