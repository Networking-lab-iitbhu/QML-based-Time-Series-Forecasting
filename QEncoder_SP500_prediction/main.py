import os
import argparse
from QEncoder_SP500_prediction.train_loop import train
from .encoder import train_encoder
from .classification_model import Classifier
from .dataset import (
    X, Y,flattened,tX,tY,split_features_labels
)
from .test import test
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--loss", dest="loss", type=str, default="MSE")
parser.add_argument("--eval_every", dest="eval_every", type=int, default=1)
parser.add_argument("--test_size", dest="test_size", type=int, default=500)
parser.add_argument("--dataset", dest="dataset", type=str, default="yelp")
parser.add_argument("--train_iter", dest="train_iter", type=int, default=200)
parser.add_argument("--n_cells", dest="n_cells", type=int, default=5) #number of EIP cells
parser.add_argument("--depth", dest="depth", type=int, default=2)
parser.add_argument("--mode", dest="mode", type=str, default="train")
parser.add_argument("--num_latent", dest="num_latent", type=int, default=4)
parser.add_argument("--num_trash", dest="num_trash", type=int, default=6)
parser.add_argument("--lr", dest="lr", type=float, default=0.01)
parser.add_argument("--bs", dest="batch_size", type=int, default=256)
args = parser.parse_args()

print(f"{os.getpid()=}")

print(f"{args.num_latent=}", flush=True)
print(f"{args.num_trash=}", flush=True)

trained_encoder = train_encoder(flattened, args)

model = Classifier(trained_encoder, args)  
    

# train_split = int(len(X) * 0.7)

# train_set, labels_train = X[:train_split], Y[:train_split] #labels_train are training set target values

total_dataset = np.array(X)
labels_dataset = np.array(Y)

train_set, validation_set, labels_train, labels_val = split_features_labels(total_dataset, labels_dataset, 0.2)

test_set, labels_test = tX, tY







if args.mode == "train":
    train(
        model,
        train_set,
        labels_train,
        validation_set,
        labels_val,
        args,
    )
    print(len(test_set)) #debugging
  
elif args.mode == "test":
    BASE_DIR = "./QEncoder_SP500_prediction/"
    test_dir = os.path.join(BASE_DIR,'evaluation_results/weights/')
    test(
        model,
        args,
        test_dir,
        test_set,
        labels_test
    )

