import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))



import pennylane as qml
from pennylane import numpy as np
import torch


def autoencoder_circuit_no_swap(weights, args, features=None):
    weights = weights.reshape(-1, args.num_latent + args.num_trash)
    if features != None:
        # Embed Inputs
        qml.AngleEmbedding(
            features[:, : args.num_trash + args.num_latent],
            wires=range(args.num_latent + args.num_trash),
            rotation="X",
        )
        qml.AngleEmbedding(
            features[
                :,
                args.num_trash + args.num_latent : 2 * args.num_trash + args.num_latent,
            ],
            wires=range(args.num_latent + args.num_trash),
            rotation="Y",
        )
    # Encoder Network
    if args.model == 'machine_aware':
        for j in range(args.depth):
            weights = torch.flatten(weights)
            w_count = 0
            for i in range(args.num_latent + args.num_trash):
                qml.RZ(weights[w_count], i)
                w_count += 1
            for i in range(args.num_latent + args.num_trash - 1):
                qml.CNOT([i, i+1])
    else:
        qml.BasicEntanglerLayers(weights, wires=range(args.num_latent + args.num_trash))

# from https://arxiv.org/pdf/1905.10876.pdf
def circuit_14(weights, args):
    depth = args.depth
    n_qubits = args.num_latent + args.num_trash
    w_count = 0
    weights = torch.flatten(weights)
    for j in range(depth):
        for i in range(n_qubits):
            qml.RY(weights[w_count], i)
            w_count += 1
        qml.CRX(weights[w_count], [n_qubits-1, 0])
        w_count += 1
        for i in range(n_qubits - 1,0,-1):
            qml.CRX(weights[w_count], [i-1, i])
            w_count += 1
        for i in range(n_qubits):
            qml.RY(weights[w_count], i)
            w_count += 1
        qml.CRX(weights[w_count], [n_qubits-1, n_qubits-2])
        w_count += 1
        qml.CRX(weights[w_count], [0,n_qubits-1])
        w_count += 1
        for i in range(n_qubits - 2):
            qml.CRX(weights[w_count], [i, i + 1])
            w_count += 1

# from https://arxiv.org/pdf/1905.10876.pdf
def circuit_11(weights, args):
    depth = args.depth
    n_qubits = args.num_latent + args.num_trash
    w_count = 0
    weights = torch.flatten(weights)
    for j in range(depth):
        for i in range(n_qubits):
            qml.RY(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(0,n_qubits-1,2):
            qml.CNOT([i+1, i])
        if n_qubits%2 == 1:
            qml.CNOT([n_qubits-1, n_qubits-2])
        for i in range(1,n_qubits-1):
            qml.RY(weights[w_count], i)
            w_count += 1
        for i in range(1,n_qubits-1):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(1,n_qubits-2,2):
            qml.CNOT([i+1, i])
        if n_qubits%2 == 1:
            qml.CNOT([n_qubits-2, n_qubits-3])


# from https://arxiv.org/pdf/1905.10876.pdf
def circuit_2_new(weights, args):
    depth = args.depth
    n_qubits = args.num_latent + args.num_trash
    w_count = 0
    weights = torch.flatten(weights)
    for j in range(depth):
        for i in range(n_qubits):
            qml.RX(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits-1):
            qml.CNOT([i+1, i])
# from https://arxiv.org/pdf/1905.10876.pdf
def circuit_2(weights, args):
    depth = args.depth
    n_qubits = args.num_latent + args.num_trash
    w_count = 0
    weights = torch.flatten(weights)
    for j in range(depth):
        for i in range(n_qubits):
            qml.Hadamard(i)
        for i in range(n_qubits-1):
            qml.CNOT([i+1, i])
        for i in range(n_qubits):
            qml.RX(weights[w_count], i)
            w_count += 1


# from https://arxiv.org/pdf/1905.10876.pdf
def circuit_13(weights, args):
    depth = args.depth
    n_qubits = args.num_latent + args.num_trash
    w_count = 0
    weights = torch.flatten(weights)
    for j in range(depth):
        for i in range(n_qubits):
            qml.RY(weights[w_count], i)
            w_count += 1
        qml.CRZ(weights[w_count], [n_qubits-1, 0])
        w_count += 1
        for i in range(n_qubits - 1,0,-1):
            qml.CRZ(weights[w_count], [i-1, i])
            w_count += 1
        for i in range(n_qubits):
            qml.RY(weights[w_count], i)
            w_count += 1
        qml.CRZ(weights[w_count], [n_qubits-1, n_qubits-2])
        w_count += 1
        qml.CRZ(weights[w_count], [0,n_qubits-1])
        w_count += 1
        for i in range(n_qubits - 2):
            qml.CRZ(weights[w_count], [i, i + 1])
            w_count += 1


# from https://arxiv.org/pdf/1905.10876.pdf
def circuit_7(weights, args):
    depth = args.depth
    n_qubits = args.num_latent + args.num_trash
    w_count = 0
    weights = torch.flatten(weights)
    for j in range(depth):
        for i in range(n_qubits):
            qml.RX(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(0,n_qubits-1,2):
            qml.CRZ(weights[w_count],[i+1, i])
            w_count += 1
        for i in range(n_qubits):
            qml.RX(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(1,n_qubits-1,2):
            qml.CRZ(weights[w_count],[i+1, i])
            w_count += 1
# from https://arxiv.org/pdf/1905.10876.pdf
def circuit_8(weights, args):
    depth = args.depth
    n_qubits = args.num_latent + args.num_trash
    w_count = 0
    weights = torch.flatten(weights)
    for j in range(depth):
        for i in range(n_qubits):
            qml.RX(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(0,n_qubits-1,2):
            qml.CRX(weights[w_count],[i+1, i])
            w_count += 1
        for i in range(n_qubits):
            qml.RX(weights[w_count], i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(weights[w_count], i)
            w_count += 1
        for i in range(1,n_qubits-1,2):
            qml.CRX(weights[w_count],[i+1, i])
            w_count += 1
def p(weights, args):
    member_num = args.version
    # Design 13 from https://arxiv.org/pdf/1905.10876.pdf
    weights = torch.flatten(weights)
    if args.anz_set ==1:
        if member_num is None or member_num % 5 == 0:
            circuit_2(weights, args)
        elif member_num is None or member_num % 5 == 1: 
            circuit_7(weights, args)
        elif member_num is None or member_num % 5 == 2:
            circuit_11(weights, args)
        elif member_num is None or member_num % 5 == 3: 
            circuit_13(weights, args)   
        elif member_num is None or member_num % 5 == 4: 
            circuit_14(weights, args)  
    elif args.anz_set == 2:
        if member_num is None or member_num % 2 == 0:
            circuit_13(weights, args)
        if member_num is None or member_num % 2 == 1:
            circuit_14(weights, args)
    elif args.anz_set == 3:
        circuit_14(weights, args)
    elif args.anz_set == 4:
        circuit_2(weights, args)
    elif args.anz_set == 5:
        circuit_11(weights, args)
    elif args.anz_set == 6:
        circuit_7(weights, args)
    elif args.anz_set == 7:
        circuit_8(weights, args)
    elif args.anz_set == 9:
        circuit_2_new(weights, args)
    
