from pennylane import numpy as np
from data import get_train_data
from model import Model

# constants
N = 100
WIRES = 6
BATCH_SIZE = 32


def main():
    # trainX, trainY, testX, testY = get_data_for_1v10(label=4)
    trainX, trainY = get_train_data(N)

    print("data loading done ✅")

    model = Model(wires=WIRES, batch_size=BATCH_SIZE)
    model.draw()
    model.train(trainX, trainY)


if __name__ == "__main__" :
    main()