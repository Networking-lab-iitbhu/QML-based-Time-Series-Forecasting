from pennylane import numpy as np
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
from tqdm import tqdm

TOTAL_WIRES = 4
TRAINABLE_WIRES = 2

class Model:
    dev = qml.device("default.qubit", wires=TOTAL_WIRES)
    def __init__(self, wires, batch_size):
        '''
        4 + 2 qubit system
        '''
        self.weights = 3.14 * np.random.rand(12, requires_grad=True)
        self.bias = np.array(0.0, requires_grad=True)
        self.optimizer = NesterovMomentumOptimizer(0.5)
        self.batch_size = batch_size
    

    @qml.qnode(dev)
    def circuit(x, theta):

        data = [0, 1, 2, 3]

        qml.AmplitudeEmbedding(x, wires=data, normalize=True)
        
        for i in data:
            qml.Hadamard(i)

        # 6 trainable args for pooling
        qml.RZ(-np.pi, wires=[0])
        qml.CNOT([2, 0])
        qml.RZ(theta[0], wires=[0])
        qml.RY(theta[1], wires=[2])
        qml.CNOT([0, 2])
        qml.RY(theta[2], wires=[2])

        qml.RZ(-np.pi, wires=[1])
        qml.CNOT([3, 1])
        qml.RZ(theta[3], wires=[1])
        qml.RY(theta[4], wires=[3])
        qml.CNOT([1, 3])
        qml.RY(theta[5], wires=[3])

        # variational part
        qml.Rot(theta[6], theta[7], theta[8], wires=[2])
        qml.CNOT([2, 3])
        qml.Rot(theta[9], theta[10], theta[11], wires=[3])
        qml.CNOT([3, 2])

        return qml.expval(qml.PauliZ(3))

    def draw(self):
        qml.draw_mpl(Model.circuit)(np.ndarray((64)), np.ndarray((9, 3)))
        plt.show()

    def classif(self, weights, bias, x):
        return ((Model.circuit(x, weights) + bias) + 1) / 2


    def loss(labels, prediction):
        return np.mean((labels - qml.math.stack(prediction))**2)


    def accuracy(labels, prediction):
        tr = lambda x: 1 if x>0.5 else 0
        return np.sum(tr(p) == l for l, p in zip(labels, prediction)) / len(labels)


    def cost(self, weights, bias, X, Y):
        prediction = [self.classif(weights, bias, x) for x in X]
        return Model.loss(Y, prediction)


    def train(self, X, Y, epochs=10):
        weights, bias = self.weights, self.bias
        for it in range(epochs):
            batch_index = np.random.randint(0, len(X), (self.batch_size,))
            X_batch = X[batch_index]
            Y_batch = Y[batch_index]
            weights = self.optimizer.step(lambda w: self.cost(w, bias, X_batch, Y_batch), weights)

            predictions = [self.classif(weights, bias, x) for x in X]

            current_cost = self.cost(weights, bias, X, Y)
            acc = Model.accuracy(Y, predictions)
            print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | acc: {acc:0.7f}")
