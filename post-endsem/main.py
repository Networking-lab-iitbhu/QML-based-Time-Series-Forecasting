from pennylane import numpy as np
from data import generate_dataset
from model import Model

# constants
N = 100
WIRES = 12
BATCH_SIZE = 32


def main():
    # trainX, trainY, testX, testY = get_data_for_1v10(label=4)
    images, labels = generate_dataset(N)

    print("data loading done ✅")

    model = Model(wires=WIRES, batch_size=BATCH_SIZE)
    model.draw()
    model.train(images, labels)


if __name__ == "__main__" :
    main()