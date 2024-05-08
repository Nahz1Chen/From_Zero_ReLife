import torch
from torch import nn

import img_dataset
import Softmax_Regress

# pytorch不会隐式地调整输入的形状
# 因此，我们定义了平展层(flatten)在线性层前调整网络输入的形状
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = img_dataset.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(),lr=0.1)

    num_epochs = 10
    Softmax_Regress.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
