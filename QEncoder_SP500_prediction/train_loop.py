from pennylane import numpy as np
import torch
import torch.optim as optim
import os 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error



def accuracy(y, y_hat):
    r2 = r2_score(y, y_hat)
    mse = mean_squared_error(y , y_hat )
    mae = mean_absolute_error(y , y_hat)
    mape = mean_absolute_percentage_error(y, y_hat)
    
    return r2, mse, mae, mape


def train(
    model,
    train_set,
    labels_train,
    validation_set,
    labels_val,
    args,
):
    print("Training Started... ", flush=True)

    losses, r2_scores, mses, maes, mapes = [], [], [], [], []

    opt = optim.RMSprop(model.parameters(), lr=args.lr)

    # Define the experiment directory
    experiment_dir = "./QEncoder_SP500_prediction/"
    evaluation_results_dir = os.path.join(experiment_dir, "evaluation_results")
    weights_dir = os.path.join(evaluation_results_dir, "weights")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, "accs"), exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, "losses"), exist_ok=True)

    experiment = f"{args.dataset}_{args.loss}_{args.depth}_{args.n_cells}_{args.num_latent}_{args.num_trash}"
    latest_path = os.path.join(weights_dir, f"{experiment}_latest.pt")
    best_path = os.path.join(weights_dir, f"{experiment}_weights_iteration_1")

    # Resume training if checkpoint exists
    start = 0
    if os.path.exists(latest_path):
        checkpoint_path = latest_path
    elif os.path.exists(best_path):
        checkpoint_path = best_path
    else:
        checkpoint_path = None

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        if 'p_weights' in state_dict:
            state_dict['model_weights'] = state_dict.pop('p_weights')
        model.load_state_dict(state_dict, strict=False)
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        losses = checkpoint.get("losses", [])
        r2_scores = checkpoint.get("r2_scores", [])
        mses = checkpoint.get("mses", [])
        maes = checkpoint.get("maes", [])
        mapes = checkpoint.get("mapes", [])
        start = checkpoint.get("iteration", checkpoint.get("epoch", 0)) + 1
        print(f"Resuming training from iteration {start}")
    else:
        print("No checkpoint found, starting from scratch.")

    if args.loss == "BCE":
        loss_fun = torch.nn.BCELoss()
    elif args.loss == "MSE":
        loss_fun = torch.nn.MSELoss()
    else:
        raise ValueError("Loss function not implemented")

    if start >= args.train_iter:
        print("Model has already been trained, use --mode test to run it on test set.")
    else:
        for i in range(start, args.train_iter):
            model.zero_grad()

            train_indices = np.random.randint(0, len(train_set), (args.batch_size,))
            train_labels = torch.tensor([labels_train[x] for x in train_indices]).to(torch.float32)
            features = [train_set[x][: args.n_cells] for x in train_indices]

            out = model(features)

            loss = loss_fun(out, train_labels)
            loss.backward()

            # Training metrics
            r2_train, mse_train, mae_train, mape_train = accuracy(
                np.array(train_labels.detach()), np.array(out.detach())
            )
            print(f"Training {i}: Loss={loss}, R2={r2_train}, MSE={mse_train}, MAE={mae_train}, MAPE={mape_train}", flush=True)

            losses.append(loss.detach().numpy())

            if i % args.eval_every == 0:
                # Validation metrics
                val_indices = np.random.randint(0, len(validation_set), (args.batch_size,))
                val_labels = torch.tensor([labels_val[x] for x in val_indices]).to(torch.float32)
                val_features = [validation_set[x][: args.n_cells] for x in val_indices]

                out_val = model(val_features)

                r2_val, mse_val, mae_val, mape_val = accuracy(
                    np.array(val_labels.detach()), np.array(out_val.detach())
                )
                print(f"Validation {i}: R2={r2_val}, MSE={mse_val}, MAE={mae_val}, MAPE={mape_val}", flush=True)

                r2_scores.append(r2_val)
                mses.append(mse_val)
                maes.append(mae_val)
                mapes.append(mape_val)

                # Save best model
                if i == 0 or mse_val <= min(mses):
                    torch.save(
                        {
                            "epoch": i,
                            "iteration": i,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "losses": losses,
                            "r2_scores": r2_scores,
                            "mses": mses,
                            "maes": maes,
                            "mapes": mapes,
                        },
                        best_path,
                    )

            # Always save latest checkpoint
            torch.save(
                {
                    "epoch": i,
                    "iteration": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "losses": losses,
                    "r2_scores": r2_scores,
                    "mses": mses,
                    "maes": maes,
                    "mapes": mapes,
                },
                latest_path,
            )

            # Save metrics
            np.save(os.path.join(evaluation_results_dir, f"accs/r2_{experiment}.npy"), r2_scores)
            np.save(os.path.join(evaluation_results_dir, f"accs/mse_{experiment}.npy"), mses)
            np.save(os.path.join(evaluation_results_dir, f"accs/mae_{experiment}.npy"), maes)
            np.save(os.path.join(evaluation_results_dir, f"accs/mape_{experiment}.npy"), mapes)

            opt.step()

    return model
