# dummy data for training
import numpy as np

def get_train_data(n):
    X = np.array([np.random.randint(0, 16) for _ in range(n)])
    Y = np.array([x%2 for x in X])
    return X, Y