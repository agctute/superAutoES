import numpy as np
import sapai as sp
from ai import AI

test = np.random.rand(89)
test = test[np.newaxis]
for i in range(test.shape[1]):
    if test[0, i] > 0.5:
        test[0, i] = 1
    else:
        test[0, i] = 0

fido = AI()

print(fido.forward(test))
