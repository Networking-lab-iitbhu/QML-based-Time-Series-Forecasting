import numpy as np

from data import get_data_upto
from model import Model

model = Model(9, 32)
x, y, xx, yy = get_data_upto(8, 64)

theta = np.random.rand(3, 3)

print(model.circuit(x[0], theta))
