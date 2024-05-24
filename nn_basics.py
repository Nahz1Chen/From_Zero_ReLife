import torch
from torch import nn
from torch.nn import functional as F

'''
自定义块
'''


class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义模型的前向传播，即如何根据输入x返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义
        return self.out(F.relu(self.hidden(X)))


'''
自定义顺序块
'''


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module时Module子类的一个示例。我们把它保存在“Module”类的成员变量_modules
            # _module的类型是OrderedDict（顺序字典）
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历
        for block in self._modules.values():
            X = block(X)
        return X


'''
在前向传播函数中执行代码
'''


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数，因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数，mm用于矩阵相乘的函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层，这相当于两个全连接共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


'''
混合搭配各种块
'''


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), )
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


'''
从嵌套块收集参数
'''


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU(), )


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block{i}', block1())
    return net


'''
参数初始化
默认情况下，PyTorch会根据一个范围均匀地初始化权重和偏置矩阵， 这个范围是根据输入和输出维度计算出的。 PyTorch的nn.init模块提供了多种预置初始化方法。
'''


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)  # 将权重初始化为从均匀分布[−10,10] 中采样的值。
        m.weight.data *= m.weight.data.abs() >= 5


'''
自定义层
'''


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


class MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


if __name__ == '__main__':
    """
    层和块
    """
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

    X = torch.rand(2, 20)
    print(net(X))

    net = MLP()
    print(net(X))

    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    print(net(X))

    net = FixedHiddenMLP()
    print(net(X))

    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    print(chimera(X))
    """
    参数管理
    """
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    print(net(X))

    print(net[2].state_dict())

    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    print(net[2].bias.grad)
    print(net[2].weight.grad == None)

    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])

    print(net.state_dict()['2.bias'].data)

    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print(rgnet(X))
    print(rgnet)
    print(rgnet[0][1][0].bias.data)

    net.apply(init_normal)
    print(net[0].weight.data[0], net[0].bias.data[0])

    net.apply(init_constant)
    print(net[0].weight.data[0], net[0].bias.data[0])

    net[0].apply(init_xavier)
    net[2].apply(init_42)
    print(net[0].weight.data[0])
    print(net[2].weight.data)

    net.apply(my_init)
    print(net[0].weight[:2])

    net[0].weight.data[:] += 1
    net[0].weight.data[0, 0] = 42
    print(net[0].weight.data[0])

    # 我们需要给共享层一个名称，以便可以引用它的参数
    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(),
                        nn.Linear(8, 1))
    print(net(X))
    # 检查参数是否相同
    print(net[2].weight.data[0] == net[4].weight.data[0])
    net[2].weight.data[0, 0] = 100
    # 确保它们实际上是同一个对象，而不只是有相同的值
    print(net[2].weight.data[0] == net[4].weight.data[0])
    """
    自定义层
    """
    layer = CenteredLayer()
    layer(torch.FloatTensor([1, 2, 3, 4, 5]))

    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    Y.mean()

    linear = MyLinear(5, 3)
    print(linear.weight)
    print(linear(torch.rand(2, 5)))

    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))
    """
    读写文件
    """
    x = torch.arange(4)
    torch.save(x, 'x-file')

    x2 = torch.load('x-file')
    print(x2)

    y = torch.zeros(4)
    torch.save([x, y], 'x-files')
    x2, y2 = torch.load('x-files')
    print(x2, y2)

    mydict = {'x': x, 'y': y}
    torch.save(mydict, 'mydict')
    mydict2 = torch.load('mydict')
    print(mydict2)

    net = MLP2()
    X = torch.randn(size=(2, 20))
    Y = net(X)

    torch.save(net.state_dict(), 'mlp.params')

    clone = MLP2()
    clone.load_state_dict(torch.load('mlp.params'))
    clone.eval()

    Y_clone = clone(X)
    Y_clone == Y
