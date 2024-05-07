import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import time


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
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
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


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
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    # plt.show()
    return axes


# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

def get_dataloader_workers():
    """使用n个进程来读取数据"""
    return 4


# train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
#                              num_workers=get_dataloader_workers())
#
# start = time.time()
# for X, y in train_iter:
#     continue
# end = time.time()
# runtime = end - start
# print(f"{runtime:.2f}sec")

if __name__ == "__main__":
    train_iter, test_iter = load_data_fashion_mnist(8, resize=28)

    X, y = next(iter(train_iter))
    show_images(X.reshape(8, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    start = time.time()
    for X, y in train_iter:
        continue
    end = time.time()
    runtime = end - start
    print(f"{runtime:.2f}sec")
