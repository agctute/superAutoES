import numpy as np
import sapai as sp
import math

PETS_AVAIL = 10
ITEMS_AVAIL = 2
TEAM_SLOTS = 5
STATUSES = 1

def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))


def relu(x):
    return np.maximum(np.zeros([1, x.shape[1]]), x)


def generate_wt(x, y):
    l = np.random.rand(x * y)
    l = l.reshape(x, y)
    # print(l.shape)
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
        m1 = sigmoid(m1)
        m2 = np.dot(m1, self.w2)
        m2 = sigmoid(m2)
        o = np.dot(m2, self.w3)
        # to check what pets exist/don't exist, look at input_data[0, 11 + 13n] for 0 <= n <= 4
        empty_slot = False
        for i in range(5):
            if not input_data[0, 11 + 13 * i]:  # if slot on team is empty
                empty_slot = True
                o[0, 41 + i] = 0  # can't sell pet that isn't there
                for j in range(2): # can't buy food for pet that isn't there
                    o[0, PETS_AVAIL*2 + (TEAM_SLOTS + 1) * i] = 0

        for i in range(10):
            if not empty_slot and not input_data[0, 65 + 2 * i]:  # if no slots or upgrades exist
                o[0, 2 * i] = 0  # can't buy this pet

        return o


