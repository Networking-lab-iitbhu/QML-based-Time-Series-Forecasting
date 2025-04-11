from data import get_data_upto
from model import Model

WIRES = 6 + 3
BATCH_SIZE = 128

trainX, trainY, testX, testY = get_data_upto(num=8, pca_comp=64)
model = Model(wires=WIRES, batch_size=BATCH_SIZE)
print("size: ", len(trainX))
# model.draw()
model.train(trainX[:1000], trainY[:1000])