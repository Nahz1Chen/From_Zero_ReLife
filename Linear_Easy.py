import numpy as np
import torch
from torch.utils import data
from torch import nn


def synthetic_data(w, b, num_examples):
    """生成 y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个Pytoch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

# nn代表Neural Networks的缩写，里面包含着很多定义好的层
net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)  # [0]访问到线性层，weight访问到w，data是真实data，normal_()使用正态分布替换掉data的值
net[0].bias.data.fill_(0)  # bias访问到偏差值，data真实值，fill_(0)全部替换为0

loss = nn.MSELoss()  # 均方误差

trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 实例化SGD示例

num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1},loss {l:f}')
