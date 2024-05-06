import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    transform=trans,
    download=True,
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    transform=trans,
    download=True,
)

print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)  # channel = 1,长:宽 = 28:28


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()