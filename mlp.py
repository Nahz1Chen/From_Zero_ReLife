import torch
from torch import nn

import img_dataset
import Softmax_Regress


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, num_inputs))  # -1的意思是把列变成num_input,自动求行数
    H = relu(X @ W1 + b1)  # 是否转置根据变量声明
    return (H @ W2 + b2)


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = img_dataset.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True
    ) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True
    ) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    loss = nn.CrossEntropyLoss(reduction='none')

    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    Softmax_Regress.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    Softmax_Regress.predict_ch3(net, test_iter)
