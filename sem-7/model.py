from pennylane import numpy as np
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
from tqdm import tqdm

TOTAL_WIRES = 6 + 3
TRAINABLE_WIRES = 3

class Model:
    dev = qml.device("default.qubit", wires=TOTAL_WIRES)
    def __init__(self, wires, batch_size):
        '''
        6 + 3 qubit system
        '''
        self.weights = 3.14 * np.random.rand(TRAINABLE_WIRES, 3, requires_grad=True)
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

        data = [0, 1, 2, 3, 4, 5]
        combined = [6, 7, 8]

        # zfeaturemap(x, wires=data)

        qml.AmplitudeEmbedding(x, wires=data, normalize=True)

        #linear combination part
        qml.QFT(wires=combined)
        for i in range(len(data)):
            qml.ctrl(add_k_fourier, control=data[i])(2 **(len(data) - i - 1), combined)
        qml.adjoint(qml.QFT)(wires=combined)

        # variational part
        for idx, wire in enumerate(combined):
            qml.Rot(*theta[idx], wires=wire)
            qml.CNOT([combined[idx], combined[(idx + 1) % len(combined)]])

        return np.array([qml.expval(qml.PauliZ(i)) for i in combined])

    def draw(self):
        qml.draw_mpl(Model.circuit)(np.ndarray((64)), np.ndarray((9, 3)))
        plt.show()

    def classif(self, weights, bias, x):
        return (Model.circuit(x, weights) + 1) / 2


    def loss(labels, prediction):
        return np.mean((labels - qml.math.stack(prediction))**2)


    def accuracy(labels, prediction):
        tr = lambda x: [1 if i>0.5 else 0 for i in x]
        return np.sum(np.array_equal(l, tr(p)) for l, p in zip(labels, prediction)) / len(labels)


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
            print(np.array(predictions[50]), Y[50])
            current_cost = self.cost(weights, bias, X, Y)
            acc = Model.accuracy(Y, predictions)
            print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | acc: {acc:0.7f}")
