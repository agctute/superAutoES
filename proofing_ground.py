import scipy
import ai
import numpy as np

one = np.ones([94, 1])
soft = scipy.special.softmax(one)
print(one)