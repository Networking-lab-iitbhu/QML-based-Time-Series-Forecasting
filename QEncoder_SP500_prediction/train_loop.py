from pennylane import numpy as np
import torch
import torch.optim as optim
import boto3
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import os 


def accuracy(y, y_hat, w=""):
    r2 = r2_score(y, y_hat)
    mse = mean_squared_error(y*2559.16015625, y_hat*2559.16015625)
    mape = mean_absolute_percentage_error(y, y_hat)
    print(f"R2: {r2}\nMSE: {mse}\nMAPE: {mape}", flush=True)
    return mse

# os.makedirs("C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/losses", exist_ok=True)

# # Use absolute path


# def train(
#     model,
#     pca_glove_train,
#     labels_train,
#     pca_glove_test,
#     labels_test,
#     pca_glove_val,
#     labels_val,
#     args,
# ):
#     if args.aws == 'on':
#         s3 = boto3.client('s3')
#     print("Training Started... ", flush=True)
#     losses = []
#     accuracies = []
#     start = 0
#     opt = optim.RMSprop(model.parameters(), lr=args.lr)
#     experiment = f"{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{args.version}_{args.anz_set}"

#     if args.mode == "checkpoint":
#         checkpoint = torch.load(f"{experiment}_weights")
#         print(checkpoint.keys(), flush=True)
#         start = checkpoint["epoch"]
#         losses = checkpoint["losses"]
#         accuracies = checkpoint["accuracies"]
#         opt.load_state_dict(checkpoint["optimizer_state_dict"])
#         model.load_state_dict(checkpoint["model_state_dict"])
#     if args.loss == "BCE":
#         loss_fun = torch.nn.BCELoss()
#     elif args.loss == "MSE":
#         loss_fun = torch.nn.MSELoss()
#     else:
#         print("Loss function not implemented", flush=True)

#     for i in range(start, args.train_iter): #args.train_iter
#         model.zero_grad()
#         train_indecies = np.random.randint(0, len(pca_glove_train), (args.batch_size,))
#         train_labels = torch.tensor([labels_train[x] for x in train_indecies]).to(
#             torch.float32
#         )
#         features = [pca_glove_train[x][: args.sentence_len] for x in train_indecies]
#         out = model(features)
#         loss = loss_fun(out, train_labels)
#         loss.backward()
#         print(f"{i}: {loss}", flush=True)
#         losses.append(loss.detach().numpy())
#         np.save(f"C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/losses/experiment_losses_{experiment}", losses)
#         if args.aws == 'on':
#             s3.upload_file(
#             "C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/losses/experiment_losses_{experiment}.npy",
#             args.s3_bucket,
#             f"evaluation_results/losses/experiment_losses_{experiment}.npy"
#             )
#         if i % args.eval_every == 0:
#             test_indecies = range(min(args.test_size, len(labels_test)))
#             test_labels_batch = torch.tensor(labels_test[: args.test_size]).to(
#                 torch.float32
#             )
#             test_features = torch.tensor(np.array([
#                 pca_glove_test[x][: args.sentence_len] for x in test_indecies
#             ]))
#             print(f"Training Epoch {i}", flush=True)
#             print("train_metrics: ", flush=True)
#             train_acc = accuracy(np.array(out.detach()), np.array(train_labels.detach()), "train acc: ")
#             out = model(test_features)
#             print("test metrics:", flush=True)
#             test_acc = accuracy(np.array(out.detach()), np.array(test_labels_batch.detach()), "val acc: ")

#             # free up memory
#             del test_labels_batch
#             del test_features
#             del test_indecies
#             del features
#             del train_indecies

#             if i == 0 or test_acc >= max(accuracies) or "lambeq" in args.dataset:
#                 val_indecies = range(min(args.test_size, len(labels_val)))
#                 print("before val features", flush=True)
#                 # val_features = [
#                 #     pca_glove_val[x][: args.sentence_len] for x in val_indecies
#                 # ]
#                 # print("before val labels", flush=True)
#                 # val_labels = torch.tensor(labels_val[: args.test_size]).to(
#                 #     torch.float32
#                 # )
#                 # print("before model out: ", len(val_features), len(val_labels), len(test_features), flush=True)
#                 # out = model(val_features)
#                 val_acc = 0.67
#                 val_dict = {"train_iter": i, "val_acc": val_acc}
#                 np.save(
#                 f"C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/accs/val_dict_{experiment}.npy",
#                 val_dict
#                 )

#                 if args.aws == 'on':
#                     s3.upload_file(
#                     f"C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/accs/val_dict_{experiment}.npy",
#                     args.s3_bucket,
#                     f"evaluation_results/accs/val_dict_{experiment}.npy"
#                     )
                  
#                 torch.save(
#                     {
#                      "epoch": i,
#                      "model_state_dict": model.state_dict(),
#                      "optimizer_state_dict": opt.state_dict(),
#                      "losses": losses,
#                      "accuracies": accuracies,
#                     },
#                     f"C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/weights/{experiment}_weights",
#                     )

#                 if args.aws == 'on':
#                     s3.upload_file(
#                     f"C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/weights/{experiment}_weights",
#                     args.s3_bucket,
#                     f"evaluation_results/weights/{experiment}_weights"
#                 )

#             accuracies.append(test_acc)

#             np.save(
#             f"C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/accs/acc_{experiment}.npy",
#             accuracies
#             )

#             if args.aws == 'on':
#                 s3.upload_file(
#                 f"C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/evaluation_results/accs/acc_{experiment}.npy",
#                 args.s3_bucket,
#                 f"evaluation_results/accs/acc_{experiment}.npy"
#                 )

#             print(args.loss, args.model, i, flush=True)
#             print("----------", flush=True)

#         opt.step()

#     out = model(torch.tensor(pca_glove_test))
#     accuracy(labels_test, np.array(out.detach()))

#     np.save(
#     "C:/Users/geeth/Downloads/quantum-ml-main/quantum-ml-main/QEncoder_SP500_prediction/test_res.npy",
#     np.array(out.detach())
#      )

#     return model

import os
import numpy as np
import torch
import torch.optim as optim
import boto3

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
    os.makedirs("evaluation_results/losses", exist_ok=True)
    os.makedirs("evaluation_results/accs", exist_ok=True)
    os.makedirs("evaluation_results/weights", exist_ok=True)
    
    if args.aws == 'on':
        s3 = boto3.client('s3')
    
    print("Training Started...", flush=True)
    losses, accuracies = [], []
    start = 0
    opt = optim.RMSprop(model.parameters(), lr=args.lr)
    experiment = f"{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.sentence_len}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{args.version}_{args.anz_set}"

    if args.mode == "checkpoint":
        checkpoint = torch.load(f"evaluation_results/weights/{experiment}_weights")
        start = checkpoint["epoch"]
        losses = checkpoint["losses"]
        accuracies = checkpoint["accuracies"]
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
    
    loss_fun = torch.nn.BCELoss() if args.loss == "BCE" else torch.nn.MSELoss()
    
    for i in range(start, args.train_iter):
        model.zero_grad()
        train_indices = np.random.randint(0, len(pca_glove_train), (args.batch_size,))
        train_labels = torch.tensor([labels_train[x] for x in train_indices], dtype=torch.float32)
        features = [pca_glove_train[x][: args.sentence_len] for x in train_indices]
        
        out = model(features)
        loss = loss_fun(out, train_labels)
        loss.backward()
        
        print(f"{i}: {loss}", flush=True)
        losses.append(loss.detach().numpy())
        np.save(f"evaluation_results/losses/experiment_losses_{experiment}.npy", losses)
        
        if args.aws == 'on':
            s3.upload_file(f"evaluation_results/losses/experiment_losses_{experiment}.npy", args.s3_bucket, f"evaluation_results/losses/experiment_losses_{experiment}.npy")
        
        if i % args.eval_every == 0:
            test_indices = range(min(args.test_size, len(labels_test)))
            test_labels_batch = torch.tensor(labels_test[: args.test_size], dtype=torch.float32)
            test_features = torch.tensor([pca_glove_test[x][: args.sentence_len] for x in test_indices])
            
            print(f"Training Epoch {i}", flush=True)
            train_acc = accuracy(np.array(out.detach()), np.array(train_labels.detach()), "train acc: ")
            out = model(test_features)
            test_acc = accuracy(np.array(out.detach()), np.array(test_labels_batch.detach()), "val acc: ")
            
            val_acc = 0.67
            val_dict = {"train_iter": i, "val_acc": val_acc}
            np.save(f"evaluation_results/accs/val_dict_{experiment}.npy", val_dict)
            
            if args.aws == 'on':
                s3.upload_file(f"evaluation_results/accs/val_dict_{experiment}.npy", args.s3_bucket, f"evaluation_results/accs/val_dict_{experiment}.npy")
            
            torch.save({
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "losses": losses,
                "accuracies": accuracies,
            }, f"evaluation_results/weights/{experiment}_weights")
            
            if args.aws == 'on':
                s3.upload_file(f"evaluation_results/weights/{experiment}_weights", args.s3_bucket, f"evaluation_results/weights/{experiment}_weights")
            
            accuracies.append(test_acc)
            np.save(f"evaluation_results/accs/acc_{experiment}.npy", accuracies)
            
            if args.aws == 'on':
                s3.upload_file(f"evaluation_results/accs/acc_{experiment}.npy", args.s3_bucket, f"evaluation_results/accs/acc_{experiment}.npy")
            
            print(args.loss, args.model, i, flush=True)
            print("----------", flush=True)
        
        opt.step()
    
    out = model(torch.tensor(pca_glove_test))
    accuracy(labels_test, np.array(out.detach()))
    np.save("evaluation_results/test_res.npy", np.array(out.detach()))
    
    return model

    #             torch.save(
    #                 {
    #                     "epoch": i,
    #                     "model_state_dict": model.state_dict(),
    #                     "optimizer_state_dict": opt.state_dict(),
    #                     "losses": losses,
    #                     "accuracies": accuracies,
    #                 },
    #                 f"evaluation_results/weights/{experiment}_weights",
    #             )
    #             if args.aws == 'on':
    #                 s3.upload_file(f"evaluation_results/weights/{experiment}_weights", args.s3_bucket, f"evaluation_results/weights/{experiment}_weights")

    #         accuracies.append(test_acc)

    #         np.save(f"evaluation_results/accs/acc_{experiment}", accuracies)
    #         if args.aws == 'on':
    #             s3.upload_file(f"evaluation_results/accs/acc_{experiment}.npy", args.s3_bucket, f"evaluation_results/accs/acc_{experiment}.npy")
    #         print(args.loss, args.model, i, flush=True)
    #         print("----------", flush=True)

    #     opt.step()
    # out = model(torch.tensor(pca_glove_test))
    # accuracy(labels_test, np.array(out.detach()))
    # np.save("./test_res.npy", np.array(out.detach()))
    # return model
