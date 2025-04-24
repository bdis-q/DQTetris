import os
import sys
from math import *
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import *

basis_gates = ["cu1", "u3"]

def select_circuit(name, num_qubits, num_qpus, qpus):
    assert num_qpus == len(qpus), "[ERROR] num_qpus != len(qpus)"
    
    circ = None
    if name == "QV":
        circ = QuantumVolume(num_qubits, seed=26).decompose()
    elif name == "DraperQFTAdder":
        circ = DraperQFTAdder(num_qubits).decompose()
    else:
        circ = load_circ_from_qasm(circ_name=name).decompose()

    assert circ != None, "[ERROR] Unknown circuit name."

    # 检查qpus是否能容纳下circ.num_qubits
    if sum(qpus) < circ.num_qubits:
        print(f"[WARNING] sum(qpus) ({sum(qpus)}) < circ.num_qubits ({circ.num_qubits})")
        print(f"[WARNING] Reallocate QPU capacities.")
        qpu_capacity = circ.num_qubits // num_qpus + 1
        if qpu_capacity % 2 == 1:
            qpu_capacity += 1
        qpus = [qpu_capacity] * num_qpus

    # 将线路转换到basis gates
    trans_circ = transpile(circ, basis_gates=basis_gates, optimization_level=0)
    # trans_circ = remove_single_qubit_gates(trans_circ)
    # print(trans_circ)
    # 输出线路和QPU信息
    gate_counts = trans_circ.count_ops()
    total_gates = sum(gate_counts.values())
    assert total_gates > 0, "[ERROR] An empty circuit."
    print(f"[INFO] {name} #Qubits: {trans_circ.num_qubits}")
    print(f"[INFO] {name} #Depths: {trans_circ.depth()}")
    print(f"[INFO] {name} #Gates: {total_gates}")
    print(f"[INFO] {num_qpus} QPUs: {qpus}\n\n")
    print(f"[INFO] {name} #Qubits: {trans_circ.num_qubits}", file=sys.stderr)
    print(f"[INFO] {name} #Depths: {trans_circ.depth()}", file=sys.stderr)
    print(f"[INFO] {name} #Gates: {total_gates}", file=sys.stderr)
    print(f"[INFO] {num_qpus} QPUs: {qpus}\n\n", file=sys.stderr)
    return circ, trans_circ, qpus

def load_circ_from_qasm(path="./benchmarks", circ_name=None):
    if circ_name == None:
        return None
    # load the .qasm file
    filename = os.path.join(path, circ_name + ".qasm")
    circ = QuantumCircuit.from_qasm_file(filename)
    return circ

def remove_single_qubit_gates(circuit):
    new_circuit = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit:
        # print(instruction.qubits)
        gate = instruction.operation
        # print(gate)
        qubits = [qubit._index for qubit in instruction.qubits]
        if qubits[0] == None:
            qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
        # print(qubits)
        if len(qubits) > 1:
            new_circuit.append(gate, qubits)
    return new_circuit
