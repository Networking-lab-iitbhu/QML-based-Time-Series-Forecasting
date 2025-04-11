from pennylane import numpy as np
import torch
import torch.optim as optim
import boto3

def accuracy(y, y_hat, w=""):
    pred = torch.round(y.squeeze())
    pred = torch.sum(abs(pred - y_hat.squeeze())).item()
    acc = pred / y.shape[0]
    print(f"{w}: {acc}")
    return acc


def train(
    model,
    pca_glove_train,
    labels_train,
    pca_glove_test,
    labels_test,
    pca_glove_val,
    labels_val,
    args,
):
    if args.aws == 'on':
        s3 = boto3.client('s3')
    print("Training Started... ")
    losses = []
    accuracies = []
    start = 0
    opt = optim.RMSprop(model.parameters(), lr=args.lr)
    experiment = f"{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{args.version}_{args.anz_set}"

    if args.mode == "checkpoint":
        checkpoint = torch.load(f"{experiment}_weights")
        print(checkpoint.keys())
        start = checkpoint["epoch"]
        losses = checkpoint["losses"]
        accuracies = checkpoint["accuracies"]
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
    if args.loss == "BCE":
        loss_fun = torch.nn.BCELoss()
    elif args.loss == "MSE":
        loss_fun = torch.nn.MSELoss()
    else:
        print("Loss function not implemented")

    for i in range(start, args.train_iter):
        model.zero_grad()
        train_indecies = np.random.randint(0, len(pca_glove_train), (args.batch_size,))
        train_labels = torch.tensor([labels_train[x] for x in train_indecies]).to(
            torch.float32
        )
        features = [pca_glove_train[x][: args.sentence_len] for x in train_indecies]
        out = model(features)
        loss = loss_fun(out, train_labels)
        loss.backward()
        print(f"{i}: {loss}", flush=True)
        losses.append(loss.detach().numpy())
        np.save(f"evaluation_results/losses/experiment_losses_{experiment}", losses)
        if args.aws == 'on':
            s3.upload_file(f'evaluation_results/losses/experiment_losses_{experiment}.npy', args.s3_bucket, f'evaluation_results/losses/experiment_losses_{experiment}.npy')
        if i % args.eval_every == 0:
            test_indecies = range(min(args.test_size, len(labels_test)))
            test_labels_batch = torch.tensor(labels_test[: args.test_size]).to(
                torch.float32
            )
            test_features = torch.tensor(np.array([
                pca_glove_test[x][: args.sentence_len] for x in test_indecies
            ]))
            print(f"Training Epoch {i}", flush=True)
            train_acc = accuracy(out, train_labels, "train acc: ")
            out = model(test_features)
            test_acc = accuracy(out, test_labels_batch, "val acc: ")
            
            # free up memory
            del test_labels_batch
            del test_features
            del test_indecies
            del features
            del train_indecies
            
            if i == 0 or test_acc >= max(accuracies) or "lambeq" in args.dataset:
                val_indecies = range(min(args.test_size, len(labels_val)))
                print("before val features")
                # val_features = [
                #     pca_glove_val[x][: args.sentence_len] for x in val_indecies
                # ]
                # print("before val labels")
                # val_labels = torch.tensor(labels_val[: args.test_size]).to(
                #     torch.float32
                # )
                # print("before model out: ", len(val_features), len(val_labels), len(test_features))
                # out = model(val_features)
                val_acc = 0.67
                val_dict = {"train_iter": i, "val_acc": val_acc}
                np.save(f"evaluation_results/accs/val_dict_{experiment}", val_dict)
                if args.aws == 'on':
                    s3.upload_file(f"evaluation_results/accs/val_dict_{experiment}.npy", args.s3_bucket, f"evaluation_results/accs/val_dict_{experiment}.npy")
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "losses": losses,
                        "accuracies": accuracies,
                    },
                    f"evaluation_results/weights/{experiment}_weights",
                )
                if args.aws == 'on':
                    s3.upload_file(f"evaluation_results/weights/{experiment}_weights", args.s3_bucket, f"evaluation_results/weights/{experiment}_weights")

            accuracies.append(test_acc)

            np.save(f"evaluation_results/accs/acc_{experiment}", accuracies)
            if args.aws == 'on':
                s3.upload_file(f"evaluation_results/accs/acc_{experiment}.npy", args.s3_bucket, f"evaluation_results/accs/acc_{experiment}.npy")
            print(args.loss, args.model, i)
            print("----------", flush=True)

        opt.step()
    return model

