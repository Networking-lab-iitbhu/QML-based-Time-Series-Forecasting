from data import get_data_upto
from model import Model
import sys

WIRES = 4
BATCH_SIZE = 128

with open("./trial_count", "r+") as f:
    trial_count = int(f.read().strip())
    f.seek(0)
    f.write(str(trial_count + 1))

print(f"running model version: v{trial_count}")

sys.stdout = open(f"./output_{trial_count}", "w")

with open("./model.py", "r") as f:
    print(f.read(), end="\n" + "-"*100 + "\n\n", flush=True)

trainX, trainY, testX, testY = get_data_upto(num=2, pca_comp=16) # binary classification
model = Model(wires=WIRES, batch_size=BATCH_SIZE)
print("size: ", len(trainX), flush=True)
# model.draw()
model.train(trainX[:], trainY[:], epochs=30)
