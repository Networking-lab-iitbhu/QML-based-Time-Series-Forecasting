from pennylane import numpy as np
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt


class Model:
    dev = qml.device("default.qubit", wires=12)
    def __init__(self, wires, batch_size):
        '''
        8 + 4 qubit system
        '''
        self.weights = 0.01 * np.random.randn(wires, 3, requires_grad=True)
        self.bias = np.array(0.0, requires_grad=True)
        self.optimizer = NesterovMomentumOptimizer(0.5)
        self.batch_size = batch_size
    

    @qml.qnode(dev)
    def circuit(x, theta):
        
        def add_k_fourier(k, wires):
            for j in range(len(wires)):
                qml.RZ(k * np.pi / (2**j), wires=wires[j])  

        def zfeaturemap(x, wires):
            for j in wires:
                qml.Hadamard(wires=j)
                qml.RZ(2*x[j], wires=j)
                qml.Hadamard(wires=j)
                qml.RZ(2*x[j], wires=j)

        data = [0, 1, 2, 3, 4, 5, 6, 7]
        combined = [8, 9, 10, 11]

        zfeaturemap(x, wires=data)

        # qml.BasisEmbedding(x, wires=data)

        #linear combination part
        qml.QFT(wires=combined)
        for i in range(len(data)):
            qml.ctrl(add_k_fourier, control=data[i])(2 **(len(data) - i - 1), combined)
        qml.adjoint(qml.QFT)(wires=combined)

        # variational part
        for wire in combined:
            qml.Rot(*theta[wire], wires=wire)
        
        for idx in range(len(combined)):
            qml.CNOT([combined[idx], combined[(idx + 1) % len(combined)]])

        return qml.expval(qml.PauliZ(combined[-1]))

    def draw(self):
        qml.draw_mpl(Model.circuit)(np.ndarray((8)), np.ndarray((12, 3)))
        plt.show()

    def classif(self, weights, bias, x):
        return Model.circuit(x, weights) + bias


    def loss(labels, prediction):
        return np.mean((labels - qml.math.stack(prediction))**2)


    def accuracy(labels, prediction):
        pred = lambda x: -1 if x < 0 else 1
        return np.sum( (l != pred(p)) for l, p in zip(labels, prediction)) / len(labels)


    def cost(self, weights, bias, X, Y):
        prediction = [self.classif(weights, bias, x) for x in X]
        return Model.loss(Y, prediction)


    def train(self, X, Y, epochs=10):
        weights, bias = self.weights, self.bias
        for it in range(epochs):
            batch_index = np.random.randint(0, len(X), (self.batch_size,))
            X_batch = X[batch_index]
            Y_batch = Y[batch_index]
            weights, bias = self.optimizer.step(self.cost, weights, bias, X=X_batch, Y=Y_batch)

            predictions = [np.sign(self.classif(weights, bias, x)) for x in X]

            current_cost = self.cost(weights, bias, X, Y)
            acc = Model.accuracy(Y, predictions)
            print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | acc: {acc:0.7f}")



