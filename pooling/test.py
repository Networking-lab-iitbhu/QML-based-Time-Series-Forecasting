import numpy as np

from data import get_data_upto
from model import Model

x, y, xx, yy = get_data_upto(2, 16)
model = Model(wires=4, batch_size=256)
print(Model.circuit(x[0], model.weights))
print(len(x), len(xx))
