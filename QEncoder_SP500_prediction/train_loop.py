from pennylane import numpy as np
import torch
import torch.optim as optim
import boto3
import os 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

BASE_DIR="./QEncoder_SP500_prediction/"

def accuracy(y, y_hat):
    r2 = r2_score(y, y_hat)
    mse = mean_squared_error(y * 2559.16015625, y_hat * 2559.16015625)
    mae = mean_absolute_error(y * 2559.16015625, y_hat * 2559.16015625)
    mape = mean_absolute_percentage_error(y, y_hat)
    
    return r2, mse, mae, mape


def train(
    model,
    train_set,
    labels_train,
    test_set,
    labels_test,
    validation_set,
    labels_val,
    args,
):
    if args.aws == 'on':
        s3 = boto3.client('s3')

    print("Training Started... ", flush=True)

    losses, r2_scores, mses, maes, mapes = [], [], [], [], []
    start = 0
    opt = optim.RMSprop(model.parameters(), lr=args.lr)

    # Define the experiment directory
    experiment_dir = os.path.expanduser("~/quantum-ml-main/QEncoder_SP500_prediction")

    # Ensure the directory exists
    os.makedirs(experiment_dir, exist_ok=True)
    evaluation_results_dir = os.path.join(experiment_dir, "evaluation_results")
    os.makedirs(evaluation_results_dir, exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, "accs"), exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, "losses"), exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, "weights"), exist_ok=True)


    experiment = f"{args.dataset}_{args.model}_{args.loss}_{args.depth}_{args.n_cells}_{args.num_latent}_{args.num_trash}_{args.pad_mode}_{args.version}_{args.anz_set}"

    if args.loss == "BCE":
        loss_fun = torch.nn.BCELoss()
    elif args.loss == "MSE":
        loss_fun = torch.nn.MSELoss()
    else:
        print("Loss function not implemented", flush=True)

    for i in range(start, args.train_iter):
        model.zero_grad()
        train_indices = np.random.randint(0, len(train_set), (args.batch_size,))
        #batch_size=256 ,train_indices = random 256 indices from training set.
        train_labels = torch.tensor([labels_train[x] for x in train_indices]).to(torch.float32)
        features = [train_set[x][: args.n_cells] for x in train_indices]
        out = model(features) #out = probability of 0th qubit collapsing to 0 (refer to classification_model.py)

        loss = loss_fun(out, train_labels)
        loss.backward()

        # Training set metrics
        r2_train, mse_train, mae_train, mape_train = accuracy(np.array(train_labels.detach()), np.array(out.detach()))
        print(f"Training {i}: Loss={loss}, R2={r2_train}, MSE={mse_train}, MAE={mae_train}, MAPE={mape_train}", flush=True)

        losses.append(loss.detach().numpy())

        if args.aws == 'on':
            s3.upload_file(f'{experiment_dir}/evaluation_results/losses/experiment_losses_{experiment}.npy', args.s3_bucket, f'evaluation_results/losses/experiment_losses_{experiment}.npy')

        if i % args.eval_every == 0:
            # Evaluate on validation set
            val_indices = np.random.randint(0, len(validation_set), (args.batch_size,))
            val_labels = torch.tensor([labels_val[x] for x in val_indices]).to(torch.float32)
            val_features = [validation_set[x][: args.n_cells] for x in val_indices]

            # Get model output for validation
            out_val = model(val_features)

            # Validation set metrics
            r2_val, mse_val, mae_val, mape_val = accuracy(np.array(val_labels.detach()), np.array(out_val.detach()))
            print(f"Validation {i}: R2={r2_val}, MSE={mse_val}, MAE={mae_val}, MAPE={mape_val}", flush=True)

            r2_scores.append(r2_val)
            mses.append(mse_val)
            maes.append(mae_val)
            mapes.append(mape_val)

            # Save best model based on validation MSE
            if i == 0 or mse_val <= min(mses):  # Corrected comparison for mse
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "losses": losses,
                        "r2_scores": r2_scores,
                        "mses": mses,
                        "maes": maes,
                        "mapes": mapes
                    },
                    f"{experiment_dir}/evaluation_results/weights/{experiment}_weights",
                )
                if args.aws == 'on':
                    s3.upload_file(f"{experiment_dir}/evaluation_results/weights/{experiment}_weights", args.s3_bucket, f"evaluation_results/weights/{experiment}_weights")

            # Save metrics for plotting
            np.save(f"{experiment_dir}/evaluation_results/accs/r2_{experiment}.npy", r2_scores)
            np.save(f"{experiment_dir}/evaluation_results/accs/mse_{experiment}.npy", mses)
            np.save(f"{experiment_dir}/evaluation_results/accs/mae_{experiment}.npy", maes)
            np.save(f"{experiment_dir}/evaluation_results/accs/mape_{experiment}.npy", mapes)

        opt.step()

    # Load the best model based on validation performance:
    
    
    # best_checkpoint = torch.load(f"{experiment_dir}/evaluation_results/weights/{experiment}_weights",weights_only=False)
    # model.load_state_dict(best_checkpoint["model_state_dict"])

    # #Final evaluation on the test set (unseen data)
    # out_test = model(torch.tensor(test_set))
    
    # r2_test, mse_test, mae_test, mape_test = accuracy(labels_test, np.array(out_test.detach()))
   
    # print(f"Final Test Accuracy:\nR2: {r2_test}\nMSE: {mse_test}\nMAE: {mae_test}\nMAPE: {mape_test}", flush=True)
    # np.save(f"{experiment_dir}/evaluation_results/test_results/test_res.npy", np.array(out_test.detach()))
   

   
    return model


