import os
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pennylane import numpy as np
import copy
BASE_DIR = "./QEncoder_SP500_prediction/encoder_details_latent_4_trash_6/"


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
    print("\n Starting Fresh Training for Encoder...\n", flush=True)
     
     
    dataset = args.dataset
    # Check if the best model already exists, if so, load it
    best_model_path = os.path.join(BASE_DIR, f"{dataset}_best_encoder_weights.pth")
    weights_text_path = os.path.join(BASE_DIR, f"{dataset}_all_encoder_weights.txt")

    if os.path.exists(best_model_path):
        print("Best model already exists. Skipping training...\n")
        enc = AutoEncoder(args)
        enc.load_state_dict(torch.load(best_model_path))
        enc.eval()  # Set the model to evaluation mode
        print(enc.weights)
        return enc  # Return the trained encoder
    else:
        # If the model doesn't exist, start fresh training
        enc = AutoEncoder(args)
        opt = optim.SGD(enc.parameters(), lr=args.lr)

        best_loss = float("inf")  # Initialize with a very high loss
        losses = []
        all_weights = {}

        # Open the weights text file for writing (create if doesn't exist)
        with open(weights_text_path, "w") as f:
            f.write("Epoch | Weights\n")  # Header line

            for i in range(1, 3):  # Train for 300 iterations
                train_indices = np.random.randint(0, len(flattened), (args.batch_size,))
                features = torch.tensor(np.array([flattened[x] for x in train_indices]), dtype=torch.float32)

                enc.zero_grad()
                out = enc(features)
                # print(out)

                loss = torch.sum(out)
                # print(f"LOSS:{loss}")
                loss.backward()
                opt.step()

                current_loss = round(float(loss / args.batch_size),3)
                losses.append(current_loss)

                # Store weights in memory
                all_weights[f"epoch_{i}"] = copy.deepcopy(enc.state_dict())

                print(f" Loss: {current_loss} | Iteration: {i}", flush=True)

                # Write epoch and weights to the text file
                weights_str = str(enc.state_dict())  # Convert weights dictionary to string
                f.write(f"{i} | {weights_str}\n")

                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save(enc.state_dict(), best_model_path)
                    print(f" New BEST model saved at iteration {i} with loss {best_loss}")

        # Save final model
        torch.save(enc.state_dict(), os.path.join(BASE_DIR, f"{dataset}_trained_encoder_final.pth"))
        print(f"\n Final encoder model saved as '{dataset}_trained_encoder_final.pth'\n")
        print(f" Best model had loss {best_loss} and was saved as '{dataset}_best_encoder_weights.pth'\n")

        # Save losses
        np.save(os.path.join(BASE_DIR, f"{dataset}_encoder_losses.npy"), losses)
        print(f"All losses saved in '{dataset}_encoder_losses.npy'")

        # Save all weights dictionary (optional if size is okay)
        torch.save(all_weights, os.path.join(BASE_DIR, f"{dataset}_all_encoder_weights.pt"))
        print(f" All epoch-wise encoder weights saved in '{dataset}_all_encoder_weights.pt'")

        return enc

