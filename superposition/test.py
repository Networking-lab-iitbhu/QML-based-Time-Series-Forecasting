# import pennylane as qml
# import numpy as np
# dev = qml.device("default.qubit", wires=1)

# @qml.qnode(dev)
# def foo():
#     qml.RY(np.arccos(0.5), wires=0)
#     # qml.Hadamard(wires=0)
#     return qml.expval(qml.PauliZ(0))
# print(foo())
from test2 import model
import sys

sys.stdout = open("output", "w")

model().loop()
