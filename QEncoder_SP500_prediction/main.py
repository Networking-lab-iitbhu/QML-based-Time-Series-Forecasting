import os
import argparse
from QEncoder_SP500_prediction.train_loop import train
from .encoder import train_encoder
from .classification_model import Classifier
from .dataset import load_dataset, split_features_labels
from .test import test

parser = argparse.ArgumentParser()
parser.add_argument("--loss", dest="loss", type=str, default="MSE")
parser.add_argument("--eval_every", dest="eval_every", type=int, default=1)
parser.add_argument("--test_size", dest="test_size", type=int, default=500)
parser.add_argument("--dataset", dest="dataset", type=str, default="sp500")
parser.add_argument("--train_iter", dest="train_iter", type=int, default=200)
parser.add_argument("--n_cells", dest="n_cells", type=int, default=5)  # Number of EIP cells
parser.add_argument("--depth", dest="depth", type=int, default=2)
parser.add_argument("--mode", dest="mode", type=str, default="train")
parser.add_argument("--num_latent", dest="num_latent", type=int, default=4)
parser.add_argument("--num_trash", dest="num_trash", type=int, default=6)
parser.add_argument("--lr", dest="lr", type=float, default=0.01)
parser.add_argument("--bs", dest="batch_size", type=int, default=256)
parser.add_argument("--encoder_train_iter",dest="encoder_train_iter",type=int,default=300)
parser.add_argument("--test_ratio",dest="test_ratio",type=int,default=0.5)
parser.add_argument("--val_ratio",dest="val_ratio",type=int,default=0.2)
args = parser.parse_args()

print(f"{os.getpid()=}")
print(f"{args.num_latent=}", flush=True)
print(f"{args.num_trash=}", flush=True)

# Load dataset
X, Y, tX, tY, flattened = load_dataset(args)

print(tX.shape)

# Train the encoder
trained_encoder = train_encoder(flattened, args)

# Initialize the classifier model
model = Classifier(trained_encoder, args)
print("Model State Dict Keys:", model.state_dict().keys())

# Split the data into training, validation, and test sets

train_set, validation_set, labels_train, labels_val = split_features_labels(X, Y, args)
test_set, labels_test = tX, tY





# Perform training or testing based on the mode
if args.mode == "train":
    train(
        model,
        train_set,
        labels_train,
        validation_set,
        labels_val,
        args,
    )
    print(len(test_set))  # Debugging output

elif args.mode == "test":
    BASE_DIR = "./QEncoder_SP500_prediction"
    test_dir = os.path.join(BASE_DIR, 'evaluation_results', 'weights')
    print(test_dir)
    #print(test_set)
    test(
        model,
        args,
        test_dir,
        test_set,
        labels_test
    )
