from data import get_data_upto
from model import Model

WIRES = 4
BATCH_SIZE = 128

trainX, trainY, testX, testY = get_data_upto(num=2, pca_comp=16) # binary classification
model = Model(wires=WIRES, batch_size=BATCH_SIZE)
print("size: ", len(trainX))
# model.draw()
model.train(trainX[:], trainY[:], epochs=30)
