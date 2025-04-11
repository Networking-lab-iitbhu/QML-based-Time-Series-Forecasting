# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# import pennylane as qml
# from pennylane import numpy as np
# from QEncoder_SP500_prediction.device_router import route_device
# import sys

# # Pytorch imports
# import torch
# import torch.nn as nn
# import torch.optim as optim

# print("encoder.py is running...")


# def construct_autoencoder_circuit(args, weights, features=None):
#     dev = route_device(args, args.num_trash * 2 + args.num_latent + 1)

#     @qml.qnode(dev, interface="torch", diff_method="backprop")
#     def autoencoder_circuit(args, weights, features=None):
#         weights = weights.reshape(-1, args.num_latent + args.num_trash)
#         if features != None:
#             # Embed Inputs
#             qml.AngleEmbedding(
#                 features[:, : args.num_trash + args.num_latent],
#                 wires=range(args.num_latent + args.num_trash),
#                 rotation="X",
#             )

#             # qml.AngleEmbedding(
#             #     features[
#             #         :,
#             #         (args.num_trash + args.num_latent) : (2 * args.num_trash + args.num_latent),
#             #     ],
#             #     wires=range(args.num_latent + args.num_trash),
#             #     rotation="Y",
#             # )

#         # Encoder Network
            
#         if args.model == 'machine_aware':
#             for j in range(args.depth):
#                 w_count = 0
#                 weights = torch.flatten(weights)
#                 for i in range(args.num_latent + args.num_trash):
#                     qml.RZ(weights[w_count], i)
#                     w_count += 1
#                 for i in range(args.num_latent + args.num_trash - 1):
#                     qml.CNOT([i, i+1])
#         else:
#             qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))

#         # Swap Test
#         aux_qubit = args.num_latent + 2 * args.num_trash
#         qml.Hadamard(aux_qubit)
#         for i in range(args.num_trash):
#             qml.CSWAP(
#                 [aux_qubit, args.num_latent + i, args.num_latent + args.num_trash + i]
#             )
#         qml.Hadamard(aux_qubit)

#         return qml.probs(wires=[aux_qubit])

#     x = autoencoder_circuit(args, weights, features)
#     # print(f"{weights=}")
#     # print(f"{features=}")
#     # print(f"autoencoder {weights.shape} {x.shape} {x}")
#     # sys.exit(-1)
#     return x


# class AutoEncoder(nn.Module):
#     def __init__(self, args, weights=None):
#         super().__init__()
#         self.n_qubits = args.num_latent + args.num_trash
#         self.args = args
#         self.weights = (
#             weights
#             if weights is not None
#             else nn.Parameter(
#                 0.01 * torch.rand(args.depth * self.n_qubits), requires_grad=True
#             )
#         )

#     def forward(self, features):
#         return construct_autoencoder_circuit(self.args, self.weights, features).to(
#             torch.float32
#         )[:, 1]


# # def train_encoder(flattened, args):
# #     print("\n Training Encoder...\n", flush=True)
# #     enc = AutoEncoder(args)
# #     opt = optim.SGD(enc.parameters(), lr=args.lr)
# #     if args.mode =='train' and args.model!='ablation_angle':
# #         for i in range(1, 301):
# #             train_indecies = np.random.randint(0, len(flattened), (args.batch_size,))
# #             features = torch.tensor(np.array([flattened[x] for x in train_indecies]))
# #             enc.zero_grad()
# #             # print(features.shape)
# #             out = enc(features)
# #             loss = torch.sum(out)
# #             loss.backward()
# #             print(f"Encoder Loss: {round(float(loss / args.batch_size),3)} Iteration: {i}", flush=True)
# #             opt.step()
# #     return enc





# def train_encoder(flattened, args):
#     print("\nResuming Training Encoder...\n", flush=True)

#     enc = AutoEncoder(args)  # Initialize model
#     opt = optim.SGD(enc.parameters(), lr=args.lr)

#     # 🔹 Try loading the saved model to resume training
#     try:
#         enc.load_state_dict(torch.load("trained_encoder.pth"))
#         print("✅ Loaded saved encoder model from 'trained_encoder.pth'")
#     except FileNotFoundError:
#         print("⚠️ No saved model found! Training from scratch...")

#     start_iter = 1  # Default start
#     if args.resume_training:  # Resume if the flag is set
#         start_iter = args.start_iteration if hasattr(args, 'start_iteration') else 71  # Continue from 21

#     # 🔹 Training loop (continues from saved state if available)
#     for i in range(start_iter, 301):  
#         train_indices = np.random.randint(0, len(flattened), (args.batch_size,))
#         features = torch.tensor(np.array([flattened[x] for x in train_indices]), dtype=torch.float32)

#         enc.zero_grad()
#         out = enc(features)
#         loss = torch.sum(out)
#         loss.backward()
        
#         print(f"Encoder Loss: {round(float(loss / args.batch_size), 3)} Iteration: {i}", flush=True)
#         opt.step()

#         # 🔹 Save model every 50 iterations
#         if i % 50 == 0:
#             torch.save(enc.state_dict(), "trained_encoder.pth")
#             print(f"💾 Model saved at iteration {i}")

#     # 🔹 Final save after completing training
#     torch.save(enc.state_dict(), "trained_encoder.pth")
#     print("\n✅ Encoder model saved as 'trained_encoder.pth' after 300 iterations.\n")

#     return enc

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pennylane import numpy as np
from QEncoder_SP500_prediction.device_router import route_device

print("encoder.py is running...")

# 🔹 Quantum Circuit for Autoencoder
import pennylane as qml
import torch

# def construct_autoencoder_circuit(args, weights, features=None):
#     dev, insert_noise = route_device(args, 'medium')  # Extract both device and noise transform
    
#     @qml.qnode(dev, interface="torch", diff_method="backprop")
#     @insert_noise  # Apply noise transformation
#     def autoencoder_circuit(weights, features=None):
#         weights = weights.reshape(-1, args.num_latent + args.num_trash)
        
#         # Encode features if provided
#         if features is not None:
#             qml.AngleEmbedding(
#                 features[:, : args.num_trash + args.num_latent],
#                 wires=range(args.num_latent + args.num_trash),
#                 rotation="X",
#             )

#         # Apply model-specific layers
#         if args.model == 'machine_aware':
#             w_count = 0
#             weights_flattened = torch.flatten(weights)  # Flatten once before loop
#             for i in range(args.num_latent + args.num_trash):
#                 qml.RZ(weights_flattened[w_count], wires=i)
#                 w_count += 1
#             for i in range(args.num_latent + args.num_trash - 1):
#                 qml.CNOT(wires=[i, i + 1])
#         else:
#             qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))

#         # Trash register and auxiliary qubit swap test
#         aux_qubit = args.num_latent + 2 * args.num_trash
#         qml.Hadamard(wires=aux_qubit)
#         for i in range(args.num_trash):
#             qml.CSWAP(wires=[aux_qubit, args.num_latent + i, args.num_latent + args.num_trash + i])
#         qml.Hadamard(wires=aux_qubit)

#         return qml.probs(wires=[aux_qubit])

#     return autoencoder_circuit(weights, features)

# def construct_autoencoder_circuit(args, weights, features=None):
#     dev, insert_noise = route_device(args, 'medium')  # Extract both device and noise transform
    
#     @qml.qnode(dev, interface="torch", diff_method="backprop")
#     def autoencoder_circuit(weights, features=None):
#         weights = weights.reshape(-1, args.num_latent + args.num_trash)
        
#         # Encode features if provided
#         if features is not None:
#             qml.AngleEmbedding(
#                 features[:, : args.num_trash + args.num_latent],
#                 wires=range(args.num_latent + args.num_trash),
#                 rotation="X",
#             )

#         # Apply model-specific layers
#         if args.model == 'machine_aware':
#             w_count = 0
#             weights_flattened = torch.flatten(weights)  # Flatten once before loop
#             for i in range(args.num_latent + args.num_trash):
#                 qml.RZ(weights_flattened[w_count], wires=i)
#                 w_count += 1
#             for i in range(args.num_latent + args.num_trash - 1):
#                 qml.CNOT(wires=[i, i + 1])
#         else:
#             qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))

#         # Trash register and auxiliary qubit swap test
#         aux_qubit = args.num_latent + 2 * args.num_trash
#         qml.Hadamard(wires=aux_qubit)
#         for i in range(args.num_trash):
#             qml.CSWAP(wires=[aux_qubit, args.num_latent + i, args.num_latent + args.num_trash + i])
#         qml.Hadamard(wires=aux_qubit)

#         return qml.probs(wires=[aux_qubit])

#     # Manually apply noise to the function
#     autoencoder_circuit = insert_noise(autoencoder_circuit)
    
#     result= autoencoder_circuit(weights, features)
#     return result

# def construct_autoencoder_circuit(args, weights, features=None):
#     dev, insert_noise = route_device(args, 'medium')  # Extract both device and noise transform
    
#     @qml.qnode(dev, interface="torch", diff_method="backprop")
#     def autoencoder_circuit(weights, features=None):
#         weights = weights.reshape(-1, args.num_latent + args.num_trash)
        
#         # Encode features if provided
#         if features is not None:
#             qml.AngleEmbedding(
#                 features[:, : args.num_trash + args.num_latent],
#                 wires=range(args.num_latent + args.num_trash),
#                 rotation="X",
#             )

#         # Apply model-specific layers
#         if args.model == 'machine_aware':
#             w_count = 0
#             weights_flattened = torch.flatten(weights)  # Flatten once before loop
#             for i in range(args.num_latent + args.num_trash):
#                 qml.RZ(weights_flattened[w_count], wires=i)
#                 w_count += 1
#             for i in range(args.num_latent + args.num_trash - 1):
#                 qml.CNOT(wires=[i, i + 1])
#         else:
#             qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))

#         # Trash register and auxiliary qubit swap test
#         aux_qubit = args.num_latent + 2 * args.num_trash
#         qml.Hadamard(wires=aux_qubit)
#         for i in range(args.num_trash):
#             qml.CSWAP(wires=[aux_qubit, args.num_latent + i, args.num_latent + args.num_trash + i])
#         qml.Hadamard(wires=aux_qubit)

#         result = qml.probs(wires=[aux_qubit])
#         return result

#     # Manually apply noise to the function
#     autoencoder_circuit = insert_noise(autoencoder_circuit)
    
#     result = autoencoder_circuit(weights, features)

#     # Convert result to a PyTorch tensor
#     try:
#         result_tensor = torch.tensor(result, dtype=torch.float32)
#     except Exception as e:
#         print(f"Error converting result to tensor: {e}")
#         # You can also return the result as is or handle it based on the needs
#         return result

#     return result_tensor





# # 🔹 AutoEncoder Model
# # class AutoEncoder(nn.Module):
# #     def __init__(self, args, weights=None):
# #         super().__init__()
# #         self.n_qubits = args.num_latent + args.num_trash
# #         self.args = args
# #         self.weights = (
# #             weights if weights is not None else
# #             nn.Parameter(0.01 * torch.rand(args.depth * self.n_qubits), requires_grad=True)
# #         )

# #     def forward(self, features):
# #         return construct_autoencoder_circuit(self.args, self.weights, features).to(torch.float32)[:, 1]


# class AutoEncoder(nn.Module):
#     def __init__(self, args, weights=None):
#         super().__init__()
#         self.n_qubits = args.num_latent + args.num_trash
#         self.args = args
#         self.weights = (
#             weights if weights is not None else
#             nn.Parameter(0.01 * torch.rand(args.depth * self.n_qubits), requires_grad=True)
#         )

#     def forward(self, features):
#         # Call the quantum circuit function and ensure result is a tensor
#         result = construct_autoencoder_circuit(self.args, self.weights, features)
        
#         # Ensure the result is in tensor format, handle the error if necessary
#         try:
#             result_tensor = torch.tensor(result, dtype=torch.float32)
#         except Exception as e:
#             print(f"Error converting result to tensor: {e}")
#             return result  # Returning result as-is if error occurs
        
#         # Return the second element (ensure slicing is valid)
#         return result_tensor[:, 1]




# # 🔹 Train & Save Model
# def train_encoder(flattened, args):
#     print("\n🔹 Starting Fresh Training for Encoder...\n", flush=True)

#     enc = AutoEncoder(args)  # New model instance
#     opt = optim.SGD(enc.parameters(), lr=args.lr)

#     for i in range(1, 301):  
#         train_indices = np.random.randint(0, len(flattened), (args.batch_size,))
#         features = torch.tensor(np.array([flattened[x] for x in train_indices]), dtype=torch.float32)

#         enc.zero_grad()
#         out = enc(features)
#         loss = torch.sum(out)
#         loss.backward()

#         print(f"Encoder Loss: {round(float(loss / args.batch_size), 3)} Iteration: {i}", flush=True)
#         opt.step()

#         # 🔹 Save model every 50 iterations
#         if i % 50 == 0:
#             torch.save(enc.state_dict(), "trained_encoder_new.pth")
#             print(f"💾 Model saved at iteration {i} as 'trained_encoder_new.pth'")

#     # 🔹 Final save
#     torch.save(enc.state_dict(), "trained_encoder_new.pth")
#     print("\n✅ Encoder model saved as 'trained_encoder_new.pth' after 300 iterations.\n")

#     return enc

# def construct_autoencoder_circuit(args, weights, features=None):
#     """Construct the autoencoder quantum circuit with the added noise model using route_device."""
#     # Integrate noise using route_device
#     dev = route_device(args, 'medium')  # Get the device with noise model

#     @qml.qnode(dev, interface="torch", diff_method="backprop")
#     def autoencoder_circuit(weights, features=None):
#         weights = weights.reshape(-1, args.num_latent + args.num_trash)

#         # Encode features if provided
#         if features is not None:
#             qml.AngleEmbedding(
#                 features[:, :args.num_trash + args.num_latent],
#                 wires=range(args.num_latent + args.num_trash),
#                 rotation="X",
#             )

#         # Apply model-specific layers
#         if args.model == 'machine_aware':
#             w_count = 0
#             weights_flattened = torch.flatten(weights)
#             for i in range(args.num_latent + args.num_trash):
#                 qml.RZ(weights_flattened[w_count], wires=i)
#                 w_count += 1
#             for i in range(args.num_latent + args.num_trash - 1):
#                 qml.CNOT(wires=[i, i + 1])
#         else:
#             qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))

#         # Trash register and auxiliary qubit swap test
#         aux_qubit = args.num_latent + 2 * args.num_trash
#         qml.Hadamard(wires=aux_qubit)
#         for i in range(args.num_trash):
#             qml.CSWAP(wires=[aux_qubit, args.num_latent + i, args.num_latent + args.num_trash + i])
#         qml.Hadamard(wires=aux_qubit)

#         # Get the probability distribution
#         return qml.probs(wires=[aux_qubit])

#     # Execute the quantum circuit directly
#     result = autoencoder_circuit(weights, features)

#     print(f"Quantum circuit executed. Result: {result}, Type: {type(result)}")

#     # Ensure the result is a numerical tensor
#     if isinstance(result, torch.Tensor):
#         return result
#     else:
#         raise TypeError(f"Expected a tensor but got {type(result)}")


# class AutoEncoder(nn.Module):
#     def __init__(self, args, weights=None):
#         super().__init__()
#         self.n_qubits = args.num_latent + args.num_trash
#         self.args = args
#         self.weights = (
#             weights if weights is not None else
#             nn.Parameter(0.01 * torch.rand(args.depth * self.n_qubits), requires_grad=True)
#         )

#     def forward(self, features):
#         # Call the quantum circuit function and ensure result is a tensor
#         result = construct_autoencoder_circuit(self.args, self.weights, features)
        
#         # Ensure the result is in tensor format
#         if isinstance(result, torch.Tensor):
#             return result  # Return if result is already a tensor
#         else:
#             raise TypeError(f"Expected a tensor but got {type(result)}")


# def train_encoder(flattened, args):
#     print("\n🔹 Starting Fresh Training for Encoder...\n", flush=True)

#     enc = AutoEncoder(args)  # New model instance
#     opt = optim.SGD(enc.parameters(), lr=args.lr)

#     for i in range(1, 301):  
#         train_indices = np.random.randint(0, len(flattened), (args.batch_size,))
#         features = torch.tensor(np.array([flattened[x] for x in train_indices]), dtype=torch.float32)

#         enc.zero_grad()
#         out = enc(features)

#         try:
#             # Compute the loss (sum of probabilities for simplicity)
#             loss = torch.sum(out)  # This can be modified depending on the desired loss function
#             loss.backward()
#         except Exception as e:
#             print(f"Error during loss calculation: {e}")
#             continue

#         print(f"Encoder Loss: {round(float(loss / args.batch_size), 3)} Iteration: {i}", flush=True)
#         opt.step()

#         # Save model every 50 iterations
#         if i % 50 == 0:
#             torch.save(enc.state_dict(), "trained_encoder_new.pth")
#             print(f"💾 Model saved at iteration {i} as 'trained_encoder_new.pth'")

#     # Final save
#     torch.save(enc.state_dict(), "trained_encoder_new.pth")
#     print("\n✅ Encoder model saved as 'trained_encoder_new.pth' after 300 iterations.\n")

#     return enc

import pennylane as qml
import torch
import numpy as np
from torch import nn, optim

# Define the quantum device with multiple noise types

# Define the quantum circuit with multiple noise types
def construct_autoencoder_circuit(args, weights, features=None):
    """Construct the autoencoder quantum circuit with an added noise model using route_device."""
    
    # Integrate noise using route_device (ensures noisy backend)
    dev = route_device(args, 'medium')

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

        # Apply model-specific layers
        if args.model == 'machine_aware':
            w_count = 0
            weights_flattened = torch.flatten(weights)
            for i in range(args.num_latent + args.num_trash):
                if w_count < len(weights_flattened):  # Prevent index out of bounds
                    qml.RZ(weights_flattened[w_count], wires=i)
                    w_count += 1
                else:
                    print(f"Warning: Not enough weights. Expected at least {i + 1}, got {w_count}.")
                
                #  Add Depolarizing Noise 
                qml.DepolarizingChannel(0.0025, wires=i)
                # Add Bit Flip Noise 
                qml.BitFlip(0.03, wires=i)
                #  Add Amplitude Damping Noise 
                qml.AmplitudeDamping(0.01, wires=i)

            for i in range(args.num_latent + args.num_trash - 1):
                if i < args.num_latent + args.num_trash - 1:
                    qml.CNOT(wires=[i, i + 1])
                    #  Add Phase Damping Noise 
                    qml.PhaseDamping(0.0125, wires=i)
                    #  Add Amplitude Damping Noise 
                    qml.AmplitudeDamping(0.01, wires=i)
                    #  Add Depolarizing Noise 
                    qml.DepolarizingChannel(0.0025, wires=i)

        else:
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
        #result = construct_autoencoder_circuit(self.args, features)(self.weights)
        return construct_autoencoder_circuit(self.args, self.weights, features).to(
           torch.float32
      )[:, 1]

# Training function with backpropagation
# def train_encoder(flattened, args):
#     print("\n🔹 Starting Fresh Training for Encoder...\n", flush=True)

#     enc = AutoEncoder(args)
#     opt = optim.SGD(enc.parameters(), lr=args.lr)

#     for i in range(1, 301): #301 actual
#         train_indices = np.random.randint(0, len(flattened), (args.batch_size,))
#         features = torch.tensor(np.array([flattened[x] for x in train_indices]), dtype=torch.float32)

#         enc.zero_grad()
#         out = enc(features)
#         print(f"Iteration {i}: Output - {out}")

#         #out = torch.tensor(out, dtype=torch.float32)  # Convert QNode output to tensor


#         # Compute the loss (sum of probabilities for simplicity)
#         loss = torch.sum(out)  # Example loss calculation
#         loss.backward()

#         print(f"Encoder Loss: {round(float(loss / args.batch_size), 3)} Iteration: {i}", flush=True)
#         opt.step()

#         if i % 50 == 0:
#             torch.save(enc.state_dict(), "trained_encoder_new.pth")
#             print(f"💾 Model saved at iteration {i} as 'trained_encoder_new.pth'")

#     # Final save
#     torch.save(enc.state_dict(), "trained_encoder_new.pth")
#     print("\n✅ Encoder model saved as 'trained_encoder_new.pth' after 300 iterations.\n")

#     return enc


def train_encoder(flattened, args, start_iter=151):
    print(f"\n🔹 Resuming Training from Iteration {start_iter}...\n", flush=True)

    enc = AutoEncoder(args)
    if os.path.exists("trained_encoder_new.pth"):
        enc.load_state_dict(torch.load("trained_encoder_new.pth"))
        print("✅ Loaded trained encoder from 'trained_encoder_new.pth'.")
    else:
        print("⚠️ No saved model found. Starting fresh training from iteration 1.")
        start_iter = 1  # If no saved model, start from scratch

    opt = optim.SGD(enc.parameters(), lr=args.lr)

    for i in range(start_iter, 301):  # Resume from `start_iter`
        train_indices = np.random.randint(0, len(flattened), (args.batch_size,))
        features = torch.tensor(np.array([flattened[x] for x in train_indices]), dtype=torch.float32)

        enc.zero_grad()
        out = enc(features)
        print(f"Iteration {i}: Output - {out}")

        loss = torch.sum(out)  # Example loss calculation
        loss.backward()

        print(f"Encoder Loss: {round(float(loss / args.batch_size), 3)} Iteration: {i}", flush=True)
        opt.step()

        if i % 50 == 0:
            torch.save(enc.state_dict(), "trained_encoder_new.pth")
            print(f"💾 Model saved at iteration {i} as 'trained_encoder_new.pth'")

    torch.save(enc.state_dict(), "trained_encoder_new.pth")
    print("\n✅ Encoder model saved as 'trained_encoder_new.pth' after 300 iterations.\n")

    return enc

# 🔹 Uncomment below lines to load model when needed
# trained_encoder = load_trained_encoder(args)
