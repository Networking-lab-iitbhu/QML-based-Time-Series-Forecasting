import pennylane as qml
import numpy as np

# def route_device(args, n_qubits):
#     return qml.device("default.qubit", args.num_trash * 2 + args.num_latent + 1)
    
# import pennylane as qml

# def route_device(args, noise_level='medium'):
#     """Properly integrates noise into a QNode instead of the device."""
#     n_qubits = args.num_latent + 2 * args.num_trash + 1

#     # Noise parameters
#     NOISE_PARAMS = {
#         'high': {'single_qubit': 0.02, 'phase': 0.03, 'cnot': 0.06, 'readout': 0.03},
#         'medium': {'single_qubit': 0.01, 'phase': 0.015, 'cnot': 0.03, 'readout': 0.015},
#         'low': {'single_qubit': 0.005, 'phase': 0.008, 'cnot': 0.01, 'readout': 0.008}
#     }
#     params = NOISE_PARAMS[noise_level]

#     # Define device
#     dev = qml.device("default.qubit", wires=n_qubits)

#     # Define noise transformation
#     def insert_noise(fn):
#         def noisy_qfunc(*args, **kwargs):
#             with qml.tape.QuantumTape() as tape:
#                 fn(*args, **kwargs)

#             # Add noise after each operation
#             new_ops = []
#             for op in tape.operations:
#                 new_ops.append(op)

#                 if isinstance(op, (qml.RX, qml.RY, qml.RZ)):
#                     new_ops.append(qml.AmplitudeDamping(params['single_qubit'], wires=op.wires))
#                     new_ops.append(qml.PhaseDamping(params['phase'], wires=op.wires))
                
#                 elif isinstance(op, qml.CNOT):
#                     new_ops.append(qml.DepolarizingChannel(params['cnot'], wires=op.wires))
                
#                 elif isinstance(op, qml.Hadamard) and op.wires[0] == n_qubits - 1:
#                     new_ops.append(qml.BitFlip(params['readout'], wires=op.wires))

#             return qml.tape.QuantumScript(new_ops, tape.measurements, shots=tape.shots)

#         return noisy_qfunc

#     return dev, insert_noise  # Return the device and the noise function
import pennylane as qml

# def route_device(args, noise_level='medium'):
#     """Properly integrates noise into a QNode instead of the device."""
#     n_qubits = args.num_latent + 2 * args.num_trash + 1

#     # Noise parameters
#     NOISE_PARAMS = {
#         'high': {'single_qubit': 0.02, 'phase': 0.03, 'cnot': 0.06, 'readout': 0.03},
#         'medium': {'single_qubit': 0.01, 'phase': 0.015, 'cnot': 0.03, 'readout': 0.015},
#         'low': {'single_qubit': 0.005, 'phase': 0.008, 'cnot': 0.01, 'readout': 0.008}
#     }
#     params = NOISE_PARAMS[noise_level]

#     # Define device
#     dev = qml.device("default.qubit", wires=n_qubits)

#     # Define noise transformation
#     def insert_noise(fn):
#         def noisy_qfunc(*args, **kwargs):
#             with qml.tape.QuantumTape() as tape:
#                 fn(*args, **kwargs)

#             # Add noise after each operation
#             new_ops = []
#             for op in tape.operations:
#                 new_ops.append(op)

#                 if isinstance(op, (qml.RX, qml.RY, qml.RZ)):
#                     new_ops.append(qml.AmplitudeDamping(params['single_qubit'], wires=op.wires))
#                     new_ops.append(qml.PhaseDamping(params['phase'], wires=op.wires))
                
#                 elif isinstance(op, qml.CNOT):
#                     # Apply noise to both qubits involved in the CNOT separately
#                     new_ops.append(qml.DepolarizingChannel(params['cnot'], wires=op.wires[0]))
#                     new_ops.append(qml.DepolarizingChannel(params['cnot'], wires=op.wires[1]))
                
#                 elif isinstance(op, qml.Hadamard) and op.wires[0] == n_qubits - 1:
#                     new_ops.append(qml.BitFlip(params['readout'], wires=op.wires))

#             return qml.tape.QuantumScript(new_ops, tape.measurements, shots=tape.shots)

#         return noisy_qfunc

#     return dev, insert_noise  # Return the device and the noise function
# import pennylane as qml
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import NoiseModel, depolarizing_error

# #from qiskit_aer.noise import depolarizing_error

# def bit_flip_error(probability):
#     """ Custom bit-flip error modeled as a depolarizing error """
#     return depolarizing_error(probability, 1)



# def route_device(args, noise_level='medium'):
#     """Integrates noise using Qiskit Aer Simulator with PennyLane."""
#     n_qubits = args.num_latent + 2 * args.num_trash + 1

#     # Create Qiskit Aer simulator
#     simulator = AerSimulator()

#     # Define noise model based on selected noise level
#     noise_model = NoiseModel()

#     # Add appropriate noise to the model
#     if noise_level == 'high':
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ['rx', 'ry', 'rz'])
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(0.06, 2), ['cx'])
#         noise_model.add_all_qubit_quantum_error(bit_flip_error(0.03), ['measure'])
#     elif noise_level == 'medium':
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['rx', 'ry', 'rz'])
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), ['cx'])
#         noise_model.add_all_qubit_quantum_error(bit_flip_error(0.015), ['measure'])
#     else:  # low noise
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ['rx', 'ry', 'rz'])
#         noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ['cx'])
#         noise_model.add_all_qubit_quantum_error(bit_flip_error(0.008), ['measure'])

#     # Create a PennyLane device using Qiskit Aer simulator and noise model
#     dev = qml.device("qiskit.aer", wires=n_qubits, simulator=simulator, noise_model=noise_model)
    
#     return dev
def route_device(args, noise_level='medium'):
    n_qubits = args.num_latent + 2* args.num_trash + 1
    dev = qml.device("default.qubit", wires=n_qubits)

    return dev
