import torch
from torch import nn

import img_dataset
import Softmax_Regress


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == "__main__":
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = img_dataset.load_data_fashion_mnist(batch_size)
    Softmax_Regress.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
