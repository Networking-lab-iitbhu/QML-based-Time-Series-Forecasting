import os
import argparse
from QEncoder_SP500_prediction.train_loop import train
from .encoder import train_encoder
from .classification_model import Classifier
from .dataset import (
    X, Y,flattened,tX,tY
)

parser = argparse.ArgumentParser()
parser.add_argument("--norm", dest="norm", type=bool, default=True)
parser.add_argument("--loss", dest="loss", type=str, default="MSE")
parser.add_argument("--eval_every", dest="eval_every", type=int, default=1)
parser.add_argument("--test_size", dest="test_size", type=int, default=500)
parser.add_argument("--model", dest="model", type=str, default="default_model")
parser.add_argument("--ansatz", dest="ansatz", type=int, default=13)
parser.add_argument("--dataset", dest="dataset", type=str, default="yelp")
parser.add_argument("--pca_dims", dest="pca_dims", type=int, default=5)
parser.add_argument("--train_iter", dest="train_iter", type=int, default=200)
parser.add_argument("--n_cells", dest="n_cells", type=int, default=5) #number of EIP cells
parser.add_argument("--depth", dest="depth", type=int, default=2)
parser.add_argument("--remove_sw", dest="remove_sw", type=str, default="False")
parser.add_argument("--mode", dest="mode", type=str, default="train")
parser.add_argument("--v", dest="version", type=int, default=1)
parser.add_argument("--num_latent", dest="num_latent", type=int, default=4)
parser.add_argument("--num_trash", dest="num_trash", type=int, default=6)
parser.add_argument("--lr", dest="lr", type=float, default=0.01)
parser.add_argument("--bs", dest="batch_size", type=int, default=256)
parser.add_argument("--machine", dest="machine", type=str, default='local')
parser.add_argument("--train_machine", dest="train_machine", type=str, default='None')
parser.add_argument("--data_eval_cat", dest="data_eval_cat", type=str, default='val')
parser.add_argument("--n_members", dest="n_members", type=int, default=15)
parser.add_argument("--pad_mode", dest="pad_mode", type=str, default='selective')
parser.add_argument("--s3_bucket", dest="s3_bucket", type=str, default='symphoniq')
parser.add_argument("--aws", dest="aws", type=str, default='off')
parser.add_argument("--anz_set", dest="anz_set", type=int, default=1)
args = parser.parse_args()

print(f"{os.getpid()=}")

print(f"{args.num_latent=}", flush=True)
print(f"{args.num_trash=}", flush=True)

trained_encoder = train_encoder(flattened, args)

if args.model == "default_model":
    model = Classifier(trained_encoder, args)  
    

train_split = int(len(X) * 0.7)

train_set, labels_train = X[:train_split], Y[:train_split] #labels_train are training set target values
test_set, labels_test = tX, tY
validation_set, labels_val = X[train_split + int(0.1*len(X)): ], Y[train_split + int(0.1*len(X)): ]


if args.mode == "train":
    train(
        model,
        train_set,
        labels_train,
        test_set,
        labels_test,
        validation_set,
        labels_val,
        args,
    )
    print(len(test_set)) #debugging
    

