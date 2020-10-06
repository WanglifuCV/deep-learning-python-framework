# -*- coding:utf-8 -*-
import numpy as np
import mnist_loader


class Network(object):
    def __init__(self, layer_sizes):
        """
        列表layer_sizes包含了每一层对应神经元的个数。如果列表是[2, 3, 1]， 那么就是一个3层神经网络。
        :param layer_sizes: 每一层的隐藏层的的宽度
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(self.layer_sizes)
        # 全用矩阵的尺寸来写
        # 目前self.weights和self.bias都是list， 以后可以考虑优化为dict或者ordereddict?
        """
        初始化工作放在这里。最开始的初始化用了np.random.random()，是[0,1]之间的均匀分布，导致加权之后的z一直是正的，
        经过映射之后就进入饱和区，权重无法更新。因此初始化是非常重要的。
        """
        self.weights = [np.random.randn(s1, s2) for s1, s2 in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.bias = [np.random.randn(s, 1) for s in layer_sizes[1:]]

    def feed_forward(self, inputs):
        """
        前向传播
        """
        # TODO: 这一部分和反向传播时的前向推理可以考虑复用
        for W, b in zip(self.weights, self.bias):
            inputs = sigmoid(np.dot(W, inputs) + b)
        return inputs

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        利用随机梯度下降算法进行网络学习。
        学习过程包含如下几部分：
        1. 更新权重
        2. 每个epoch测试精度。
        """
        # 获得training_data的数据量
        training_data = list(training_data)
        training_data_num = len(training_data)
        # 计算一个epoch要迭代多少次
        # epoch的迭代
        for epoch in range(epochs):
            # 每个epoch都重新shuffle一次
            np.random.shuffle(training_data)
            # epoch内的迭代,
            for k in range(0, training_data_num, mini_batch_size):
                mini_batch_data = training_data[k: k + mini_batch_size]
                self.update_mini_batch(mini_batch_data=mini_batch_data,
                                       eta=eta)
            if test_data is not None:
                test_data = list(test_data)
                test_data_num = len(test_data)
                print('Epoch {}, {} / {}'.format(epoch, self.eval(test_data), test_data_num))

    def update_mini_batch(self, mini_batch_data, eta):
        """
        权重更新模块。在这里利用反向传播来计算各个权重的梯度，然后更新权重。
        :param mini_batch_data: 每个mini_batch
        :param eta: 学习率
        """
        nabla_W = [np.zeros(weights.shape) for weights in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.bias]

        for x, y in mini_batch_data:
            delta_nabla_W, delta_nabla_b = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_W = [nw + dnw for nw, dnw in zip(nabla_W, delta_nabla_W)]

        self.weights = [w - (eta / len(mini_batch_data)) * nw for w, nw in zip(self.weights, nabla_W)]
        self.bias = [b - (eta / len(mini_batch_data)) * nb for b, nb in zip(self.bias, nabla_b)]

    def eval(self, test_data):
        """
        TODO: 优化
        用来统计多少个结果是对的
        :param test_data:
        :return:
        """
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def summary(self):
        """
        展示各个层的宽度和尺寸
        """
        print('Input layer, input size {}'.format(self.weights[0].shape[-1]))
        for idx, (W, b) in enumerate(zip(self.weights, self.bias)):
            print('Layer {}, Weights size {}, bias size {}.'.format(idx + 1, W.shape, b.shape))
        print('Output layer, output size {}'.format(self.bias[-1].shape[0]))

    def backprop(self, x, y):
        """
        返回一个表示代价函数C_x的梯度元组(nabla_W, nabla_b). nabla_W和nabla_b是一层接一层的numpy数组列表,类似于self.bias和
        self.weights.
        :param x:
        :param y:
        :return:
        """
        nabla_W = [np.zeros(weights.shape) for weights in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.bias]

        # Feedforward
        # 可以考虑直接用self.feedforward来代替？
        activation = x
        activations = [x]
        # 名称以后再优化
        zs = []
        for weights, bias in zip(self.weights, self.bias):
            z = np.dot(weights, activation) + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backpropgation
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_W[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_W[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_W, nabla_b

    def cost_derivative(self, output_activations, y):
        """
        返回关于输出激活值的偏导数的向量
        :param output_activations:
        :param y:
        :return:
        """
        return (output_activations - y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':
    trainint_data, validation_datga, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(trainint_data, 30, 10, 3.0, test_data=test_data)
