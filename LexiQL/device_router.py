import pennylane as qml

def route_device(args, n_qubits):
    return qml.device("default.qubit", args.num_trash * 2 + args.num_latent + 1)
    
