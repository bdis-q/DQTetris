from pytket.extensions.qiskit import qiskit_to_tk
from pytket.circuit.display import render_circuit_as_html
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.utils import DQCPass
from pytket_dqc.distributors import *
from itertools import combinations
from qiskit import transpile
import time

def output_circuit_as_html(circ, filename):
    html = render_circuit_as_html(circ)
    with open(f"./outputs/html/{filename}.html", "w") as f:
        f.write(html)
    return

# 构建一个全连接网络
def build_fc_network(qpus):
    server_coupling = [
        list(combination)
        for combination in combinations([i for i in range(len(qpus))], 2)
    ]
    qubits = [i for i in range(sum(qpus))]
    server_qubits_list = [
        qubits[sum(qpus[:i]) : sum(qpus[:i+1])]
        for i in range(len(qpus))
    ]
    server_qubits = {
        i: qubits_list
        for i, qubits_list in enumerate(server_qubits_list)
    }
    # print(server_qubits)
    network = NISQNetwork(server_coupling, server_qubits)
    return network

class Pytket_DQC:
    def __init__(self, qiskit_circ, qpus, workflow):
        self.qiskit_circ = qiskit_circ
        self.qpus = qpus
        self.workflow = workflow
        self.build_network()

    @property
    def name(self):
        return f"Pytket-DQC ({self.workflow})"
    
    def build_network(self):
        start_time = time.time()
        self.qiskit_circ = transpile(self.qiskit_circ, basis_gates=["cu1", "rz", "h"], optimization_level=0)
        self.circ = qiskit_to_tk(self.qiskit_circ)
        DQCPass().apply(self.circ)
        # output_circuit_as_html(circ, "original_circ")

        self.network = build_fc_network(self.qpus)
        # f = network.draw_nisq_network()
        # f.savefig("fc_network.png")
        # print("network done")
        end_time = time.time()
        print(f"[Pytket-DQC] Preprocessing Time: {end_time - start_time} seconds")        
        return

    def distribute(self):
        start_time = time.time()
        if self.workflow == "CE":
            print("[Pytket-DQC: CoverEmbeddingSteinerDetached]")
            distribution = CoverEmbeddingSteinerDetached().distribute(self.circ, self.network, seed=26)
        else:
            print("[Pytket-DQC: PartitioningAnnealing]")
            distribution = PartitioningAnnealing().distribute(self.circ, self.network, seed=26)
        end_time = time.time()
        # distributed_circ = distribution.to_pytket_circuit()
        # output_circuit_as_html(distributed_circ, "distributed_circ")        
        self.num_comms = self.num_gates = distribution.cost()
        self.num_swaps = 0
        print(f"#comms: {self.num_comms}")
        print(f"#gate_comms: {self.num_gates}")
        print(f"#swap_comms: {self.num_swaps}")
        self.exec_time = end_time - start_time
        print(f"Time: {self.exec_time} seconds\n\n")

        # print("detached gate count", distribution.detached_gate_count())
        # print("non local gate count", distribution.non_local_gate_count())
        # print("hyperedge count", len(distribution.circuit.hyperedge_list))

        partition = [[] for _ in range(len(self.qpus))] # 每个qpu上一个划分
        for i in range(self.qiskit_circ.num_qubits): # q[i]->server[j]
            # print(i, distribution.placement.placement[i])
            partition[distribution.placement.placement[i]].append(i)
        filename = f"./outputs/paths/{self.qiskit_circ.name[:11]}_{self.qiskit_circ.num_qubits}_pytket.txt"
        with open(filename, 'w') as file:
            file.write(' '.join(map(str, partition)) + '\n')