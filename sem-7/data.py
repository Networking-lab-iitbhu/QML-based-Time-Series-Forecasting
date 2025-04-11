import numpy as np
from mnist import MNIST
from sklearn.decomposition import PCA
from progress.spinner import Spinner

def transform_bitwise(x, bits: int = 3):
    pad = lambda arr: [0] * (bits - len(arr)) + arr
    return np.array([pad(list(map(int, list(bin(w)[2:])))) for w in x])

def get_data(pca_comp: int):
    mndata = MNIST('./dataset')
    
    trainX, trainY = mndata.load_training()
    testX, testY = mndata.load_testing()

    pca = PCA(n_components=pca_comp)
    trainX = pca.fit_transform(trainX)
    testX = pca.transform(testX)
    
    return (np.array(x) for x in (trainX, trainY, testX, testY))

def get_data_upto(num: int, pca_comp: int = 32):
    trainX, trainY, testX, testY = get_data(pca_comp)
    trainX = [trainX[i] for i, x in enumerate(trainY) if x < num]
    testX = [testX[i] for i, x in enumerate(testY) if x < num]
    trainY = [x for i, x in enumerate(trainY) if x < num]
    testY = [x for i, x in enumerate(testY) if x < num]

    trainY = transform_bitwise(trainY) # could have been made inplace
    testY = transform_bitwise(testY)

    return (np.array(trainX), trainY, np.array(testX), testY)