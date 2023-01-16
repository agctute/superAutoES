import numpy as np
import sapai as sp


def relu(x):
    return max(0, x)


def generate_wt(x, y):
    l = np.random.rand(x*y)
    l.reshape([x, y])
    return l


class AI:
    def __init__(self, w1=None, w2=None, w3=None):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        if not self.w1:
            self.w1 = generate_wt(89, 30)
        if not self.w2:
            self.w2 = generate_wt(30, 30)
        if not self.w3:
            self.w3 = generate_wt(30, 47)

    def forward(self, input_data):
        m1 = np.dot(input_data, self.w1)
        m2 = np.dot(m1, self.w2)
        o = np.dot(m2, self.w3)
        return o

