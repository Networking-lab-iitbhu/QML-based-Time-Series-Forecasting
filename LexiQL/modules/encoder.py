import pennylane as qml
from pennylane import numpy as np
from device_router import route_device
import sys

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim



def construct_autoencoder_circuit(args, weights, features=None):
    dev = route_device(args, args.num_trash * 2 + args.num_latent + 1)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def autoencoder_circuit(args, weights, features=None):
        weights = weights.reshape(-1, args.num_latent + args.num_trash)
        if features != None:
            # Embed Inputs
            qml.AngleEmbedding(
                features[:, : args.num_trash + args.num_latent],
                wires=range(args.num_latent + args.num_trash),
                rotation="X",
            )

            # qml.AngleEmbedding(
            #     features[
            #         :,
            #         (args.num_trash + args.num_latent) : (2 * args.num_trash + args.num_latent),
            #     ],
            #     wires=range(args.num_latent + args.num_trash),
            #     rotation="Y",
            # )

        # Encoder Network
            
        if args.model == 'machine_aware':
            for j in range(args.depth):
                w_count = 0
                weights = torch.flatten(weights)
                for i in range(args.num_latent + args.num_trash):
                    qml.RZ(weights[w_count], i)
                    w_count += 1
                for i in range(args.num_latent + args.num_trash - 1):
                    qml.CNOT([i, i+1])
        else:
            qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))

        # Swap Test
        aux_qubit = args.num_latent + 2 * args.num_trash
        qml.Hadamard(aux_qubit)
        for i in range(args.num_trash):
            qml.CSWAP(
                [aux_qubit, args.num_latent + i, args.num_latent + args.num_trash + i]
            )
        qml.Hadamard(aux_qubit)

        return qml.probs(wires=[aux_qubit])

    x = autoencoder_circuit(args, weights, features)
    # print(f"{weights=}")
    # print(f"{features=}")
    # print(f"autoencoder {weights.shape} {x.shape} {x}")
    # sys.exit(-1)
    return x


class AutoEncoder(nn.Module):
    def __init__(self, args, weights=None):
        super().__init__()
        self.n_qubits = args.num_latent + args.num_trash
        self.args = args
        self.weights = (
            weights
            if weights is not None
            else nn.Parameter(
                0.01 * torch.rand(args.depth * self.n_qubits), requires_grad=True
            )
        )

    def forward(self, features):
        return construct_autoencoder_circuit(self.args, self.weights, features).to(
            torch.float32
        )[:, 1]


def train_encoder(flattened, args):
    print("\n Training Encoder...\n")
    enc = AutoEncoder(args)
    opt = optim.SGD(enc.parameters(), lr=args.lr)
    if args.mode =='train' and args.model!='ablation_angle':
        for i in range(1, 301):
            train_indecies = np.random.randint(0, len(flattened), (args.batch_size,))
            features = torch.tensor(np.array([flattened[x] for x in train_indecies]))
            enc.zero_grad()
            # print(features.shape)
            out = enc(features)
            loss = torch.sum(out)
            loss.backward()

            if i % 20 == 0:
                print(
                    f"Encoder Loss: {round(float(loss / args.batch_size),3)} Iteration: {i}"
                )
            opt.step()
    return enc
