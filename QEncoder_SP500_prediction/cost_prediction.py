import os
from qiskit import transpile, execute
from qiskit.providers.ibmq import IBMQ
from QEncoder_SP500_prediction.modules.encoder import train_encoder
from QEncoder_SP500_prediction.dataset import flattened

# Load IBMQ account
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

# Choose an IBM backend (e.g., 7-qubit or 27-qubit machine)
backend = provider.get_backend('ibmq_jakarta')

# Generate the trained quantum encoder circuit
trained_encoder = train_encoder(flattened, args=None)  # Set args=None if not needed

# Transpile the circuit for the IBM backend
transpiled_circuit = transpile(trained_encoder, backend)

# Get circuit properties
num_gates = transpiled_circuit.size()  # Total gates
circuit_depth = transpiled_circuit.depth()  # Circuit depth
gate_counts = transpiled_circuit.count_ops()

# Estimate execution time
single_qubit_gates = gate_counts.get('u1', 0) + gate_counts.get('u2', 0) + gate_counts.get('u3', 0)
two_qubit_gates = gate_counts.get('cx', 0)

execution_time = (single_qubit_gates * 0.035) + (two_qubit_gates * 0.250) + 5  # in microseconds

# Estimate cost
shots = 1024  # Default shots
total_time = execution_time * shots / 1e6  # Convert to seconds
cost_estimate = total_time * 1  # $1 per second (example pricing)

# Print results
print(f"✅ Circuit Gates: {num_gates}, Circuit Depth: {circuit_depth}")
print(f"⏳ Estimated Execution Time: {execution_time:.2f} μs")
print(f"💰 Estimated Cost: ${cost_estimate:.2f} for {shots} shots")

# (Optional) Run on IBM Quantum
run_on_ibm = input("Do you want to submit this job to IBM Quantum? (yes/no): ").strip().lower()
if run_on_ibm == "yes":
    job = execute(trained_encoder, backend, shots=shots)
    from qiskit.tools.monitor import job_monitor
    job_monitor(job)
    print(f"🆔 Job ID: {job.job_id()} - Check IBM Quantum Dashboard for cost details.")
