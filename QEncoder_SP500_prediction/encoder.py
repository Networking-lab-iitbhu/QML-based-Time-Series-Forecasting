import os
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pennylane import numpy as np
import copy
BASE_DIR = "./QEncoder_SP500_prediction/encoder_details/"


print("encoder.py is running...")

#  Quantum Circuit for Autoencoder

def construct_autoencoder_circuit(args, weights, features=None):
    
    dev = qml.device("default.qubit", wires=args.num_latent + 2* args.num_trash + 1)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def autoencoder_circuit(weights, features=None):
        # Debugging: Print the shape before reshaping
        print(f"Shape of weights before reshaping: {weights.shape}")
        
        # Reshape weights correctly
        try:
            weights = weights.reshape(-1, args.num_latent + args.num_trash)
        except RuntimeError as e:
            raise ValueError(f"Weight reshaping error: {e}, Expected shape: (-1, {args.num_latent + args.num_trash})")

        print(f"Shape of weights after reshaping: {weights.shape}")

        # Encode features if provided
        if features is not None:
            qml.AngleEmbedding(
                features[:, :args.num_trash + args.num_latent],
                wires=range(args.num_latent + args.num_trash),
                rotation="X",
            )

        qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))
       

        # Trash register and auxiliary qubit swap test
        aux_qubit = args.num_latent + 2 * args.num_trash
        qml.Hadamard(wires=aux_qubit)
        for i in range(args.num_trash):
            qml.CSWAP(wires=[aux_qubit, args.num_latent + i, args.num_latent + args.num_trash + i])
        qml.Hadamard(wires=aux_qubit)

        # Get the probability distribution
        return qml.probs(wires=[aux_qubit])

    # Execute the quantum circuit directly
    result = autoencoder_circuit(weights, features)

    print(f"Quantum circuit executed. Result:  Type: {type(result)}")

    # Ensure the result is a numerical tensor
    if isinstance(result, torch.Tensor):
        return result
    else:
        raise TypeError(f"Expected a tensor but got {type(result)}")

# Define AutoEncoder Model in PyTorch
class AutoEncoder(nn.Module):
    def __init__(self, args, weights=None):
        super().__init__()
        self.n_qubits = args.num_latent + args.num_trash
        self.args = args
        self.weights = (
            weights if weights is not None else
            nn.Parameter(0.01 * torch.rand(args.depth * self.n_qubits), requires_grad=True)
        )

    def forward(self, features):
        # Call the quantum circuit function
        return construct_autoencoder_circuit(self.args, self.weights, features).to(
           torch.float32
      )[:, 1]


def autoencoder_circuit_trained(weights, args):
    weights = weights.reshape(-1, args.num_latent + args.num_trash)
    qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))


def train_encoder(flattened, args):
    print("\n Starting Training for Encoder...\n", flush=True)

    dataset = args.dataset
    qubit_config = f"{args.num_latent}_{args.num_trash}"

    best_model_path = os.path.join(BASE_DIR, f"{dataset}_best_encoder_weights_{qubit_config}_{args.encoder_train_iter}.pth")
    latest_model_path = os.path.join(BASE_DIR, f"{dataset}_latest_encoder_checkpoint_{qubit_config}_{args.encoder_train_iter}.pt")
    weights_text_path = os.path.join(BASE_DIR, f"{dataset}_all_encoder_weights_{qubit_config}_{args.encoder_train_iter}.txt")

    enc = AutoEncoder(args)
    opt = optim.SGD(enc.parameters(), lr=args.lr)

    best_loss = float("inf")
    losses = []
    all_weights = {}
    start = 1  # iteration to begin from

    # Resume training if checkpoint exists
    if os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path,weights_only=False)
        enc.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        losses = checkpoint.get("losses", [])
        best_loss = checkpoint.get("best_loss", best_loss)
        start = checkpoint.get("iteration", 0) + 1
        print(f"Resuming encoder training from iteration {start}...\n", flush=True)
    elif os.path.exists(best_model_path):
        print("Best encoder already exists. Skipping training...\n")
        state_dict = torch.load(best_model_path,weights_only=False)
        model_state_dict=state_dict['model_state_dict']
        enc.load_state_dict(model_state_dict)
        enc.eval()
        print(enc.weights)
        return enc

    # Open weights file in append mode if resuming, else write header
    mode = 'a' if start > 1 else 'w'
    with open(weights_text_path, mode) as f:
        if start == 1:
            f.write("Epoch | Weights\n")

        for i in range(start, args.encoder_train_iter):
            train_indices = np.random.randint(0, len(flattened), (args.batch_size,))
            features = torch.tensor(np.array([flattened[x] for x in train_indices]), dtype=torch.float32)

            enc.zero_grad()
            out = enc(features)
            loss = torch.sum(out)
            loss.backward()
            opt.step()

            current_loss = round(float(loss / args.batch_size), 3)
            losses.append(current_loss)

            all_weights[f"epoch_{i}"] = copy.deepcopy(enc.state_dict())
            print(f" Loss: {current_loss} | Iteration: {i}", flush=True)

            # Write current weights
            weights_str = str(enc.state_dict())
            f.write(f"{i} | {weights_str}\n")

            # Save best model
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(enc.state_dict(), best_model_path)
                print(f" New BEST model saved at iteration {i} with loss {best_loss}")

            # Always save latest checkpoint
            torch.save({
                "iteration": i,
                "model_state_dict": enc.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "losses": losses,
                "best_loss": best_loss
            }, latest_model_path)

    # Save final model and training history
    final_path = os.path.join(BASE_DIR, f"{dataset}_trained_encoder_final_{qubit_config}_{args.encoder_train_iter}.pth")
    torch.save(enc.state_dict(), final_path)
    print(f"\n Final encoder model saved as '{os.path.basename(final_path)}'\n")
    print(f" Best model had loss {best_loss} and was saved as '{os.path.basename(best_model_path)}'\n")

    np.save(os.path.join(BASE_DIR, f"{dataset}_encoder_losses_{qubit_config}_{args.encoder_train_iter}.npy"), losses)
    torch.save(all_weights, os.path.join(BASE_DIR, f"{dataset}_all_encoder_weights_{qubit_config}_{args.encoder_train_iter}.pt"))

    print(f"All losses and weights saved successfully.")
    return enc
