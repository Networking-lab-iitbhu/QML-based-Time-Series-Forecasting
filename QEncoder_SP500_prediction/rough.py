# from qiskit import IBMQ

# # Load IBMQ account
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')

# # Select backend (replace with your preferred IBM device)
# backend = provider.get_backend('ibmq_manila')

# # Get noise properties
# noise_props = backend.properties()

# # Get relaxation times (T1, T2)
# t1_values = [q.t1 for q in noise_props.qubits]  # T1 relaxation times
# t2_values = [q.t2 for q in noise_props.qubits]  # T2 dephasing times

# # Get gate error rates
# cx_errors = [noise_props.gate_error("cx", (i, i + 1)) for i in range(len(noise_props.qubits) - 1)]
# bitflip_probs = [noise_props.readout_error(i) for i in range(len(noise_props.qubits))]

# print("T1 (relaxation):", t1_values)
# print("T2 (dephasing):", t2_values)
# print("CNOT Errors:", cx_errors)
# print("Bit Flip Probabilities:", bitflip_probs)


from qiskit_ibm_runtime import QiskitRuntimeService

# Save IBM Quantum account (Run this only once)
QiskitRuntimeService.save_account(
    "a01a7b887b63ea23efdfce7696235527ba45018c7db1cb6ed46d0605641c5b8d5d871e87b7342c750ebdc21a7996f363dded6e3f0b228840f6296a5de8e8d9a0", 
    overwrite=True, 
    channel="ibm_quantum"
)

# Initialize service
service = QiskitRuntimeService()

# # List available backends
# backends = service.backends()
# for backend in backends:
#     print(f"{backend.name}: {backend.num_qubits} qubits, {backend.backend_version}")

backend = service.backend("ibm_brisbane")  # Change as needed

# Get backend properties
properties = backend.properties()

# Print qubit errors
for i in range(len(properties.qubits)):
    print(f"Qubit {i}:")
    print(f"  T1: {properties.t1(i)}")
    print(f"  T2: {properties.t2(i)}")
    print(f"  Readout error: {properties.readout_error(i)}")
    print(f"  Gate error (X): {properties.gate_error('x', i)}")

# Print two-qubit errors
for gate in properties.gates:
    if gate.gate == "cx":
        q0, q1 = gate.qubits
        print(f"CNOT Error ({q0}, {q1}): {properties.gate_error('cx', q0, q1)}")