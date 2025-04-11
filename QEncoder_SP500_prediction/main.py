import os
import argparse
from QEncoder_SP500_prediction.train_loop import train
from QEncoder_SP500_prediction.modules.encoder import train_encoder
from QEncoder_SP500_prediction.models import (
    amplitude_mean_ablation,
    encoder_ablation,
    optimized_xyz,
    rnn_same_p,
    symphoniq,
)
from QEncoder_SP500_prediction.dataset import (
    X, Y, flattened, tX, tY
)

parser = argparse.ArgumentParser()
parser.add_argument("--norm", dest="norm", type=bool, default=True)
parser.add_argument("--loss", dest="loss", type=str, default="MSE")
parser.add_argument("--eval_every", dest="eval_every", type=int, default=1)
parser.add_argument("--test_size", dest="test_size", type=int, default=500)
parser.add_argument("--model", dest="model", type=str, default="diff_p")
parser.add_argument("--ansatz", dest="ansatz", type=int, default=13)
parser.add_argument("--dataset", dest="dataset", type=str, default="yelp")
parser.add_argument("--pca_dims", dest="pca_dims", type=int, default=5)
parser.add_argument("--train_iter", dest="train_iter", type=int, default=200)
parser.add_argument("--sentence_len", dest="sentence_len", type=int, default=5)
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
args.resume_training = True  # 🔹 Add this before calling train_encoder
args.start_iteration = 21    # 🔹 Ensure a valid starting iteration (default is 21)

def get_circuit(args, flattened=None):

    trained_encoder = train_encoder(flattened, args)
    if args.model == "same_p":
        return rnn_same_p.SentimentAnalysisSameP(trained_encoder, args)
    elif args.model == "diff_p" or args.model == 'pair_encoding' or args.model =='machine_aware':
        return symphoniq.SymphoniQ(trained_encoder, args)
    elif args.model == "amp_mean":
        return amplitude_mean_ablation.SentimentAnalysisAmpMean(args)
    elif args.model == "ablation_angle":
        return encoder_ablation.SentimentAnalysisEncoderAblation(args)
    elif args.model == "ablation_angle_amp":
        return encoder_ablation.SentimentAnalysisEncoderAblation(args)
    elif args.model == "xyz":
        return optimized_xyz.SentimentAnalysisXYZ(trained_encoder, args)

train_split = int(len(X) * 0.7)
test_split = int(len(X) * 0.1)

pca_glove_train, labels_train = X[:train_split], Y[:train_split]
pca_glove_test, labels_test = tX, tY
pca_glove_val, labels_val = X[train_split + test_split: ], Y[train_split + test_split: ]

sentiment_circuit = get_circuit(args, flattened=flattened)
if args.mode == "train":
    train(
        sentiment_circuit,
        pca_glove_train,
        labels_train,
        pca_glove_test,
        labels_test,
        pca_glove_val,
        labels_val,
        args,
    )
    


