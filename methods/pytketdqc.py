from pytket.extensions.qiskit import qiskit_to_tk
from pytket.circuit.display import render_circuit_as_html
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.utils import DQCPass
from pytket_dqc.distributors import *
from itertools import combinations
from qiskit import transpile
import time
import numpy as np
import networkx as nx

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
    # 计算每个节点到其他节点的qubit swap cost
    swap_cost_matrix = np.zeros((len(qpus), len(qpus)), dtype=int)
    for i in range(len(qpus)):
        for j in range(len(qpus)):
            swap_cost_matrix[i][j] = 1
        swap_cost_matrix[i][i] = 0
    return NISQNetwork(server_coupling, server_qubits), swap_cost_matrix

# 构建一个mesh-grid网络
def build_mesh_grid_network(qpus, n_rows=4, n_cols=2):
    assert len(qpus) == n_rows * n_cols
    server_coupling = []
    for row in range(n_rows): # 生成水平连接（左右）
        for col in range(n_cols - 1):
            server_coupling.append([row * n_cols + col, row * n_cols + col + 1])
    for row in range(n_rows - 1): # 生成垂直连接（上下）
        for col in range(n_cols):
            server_coupling.append([row * n_cols + col, (row + 1) * n_cols + col])
    qubits = [i for i in range(sum(qpus))]
    server_qubits_list = [
        qubits[sum(qpus[:i]) : sum(qpus[:i+1])]
        for i in range(len(qpus))
    ]
    server_qubits = {
        i: qubits_list
        for i, qubits_list in enumerate(server_qubits_list)
    }
    # 计算每个节点到其他节点的qubit swap cost
    G = nx.Graph()
    G.add_edges_from(server_coupling)
    swap_cost_matrix = np.zeros((len(G.nodes()), len(G.nodes())), dtype=int)
    for i in G.nodes():
        for j in G.nodes():
            swap_cost_matrix[i][j] = 2 * nx.shortest_path_length(G, source=i, target=j) - 1
        swap_cost_matrix[i][i] = 0
    return NISQNetwork(server_coupling, server_qubits), swap_cost_matrix

class Pytket_DQC:
    def __init__(self, qiskit_circ, qpus, workflow, network="fc"):
        self.qiskit_circ = qiskit_circ
        self.qpus = qpus
        self.workflow = workflow
        self.build_network(network)

    @property
    def name(self):
        return f"Pytket-DQC ({self.workflow})"

    def build_network(self, network):
        start_time = time.time()
        self.qiskit_circ = transpile(self.qiskit_circ, basis_gates=["cu1", "rz", "h"], optimization_level=0)
        self.circ = qiskit_to_tk(self.qiskit_circ)
        DQCPass().apply(self.circ)
        # output_circuit_as_html(circ, "original_circ")

        if network == "fc":
            self.network, _ = build_fc_network(self.qpus)
        elif network == "mesh":
            self.network, _ = build_mesh_grid_network(self.qpus, int(len(self.qpus)/2), 2)
        else:
            raise ValueError("Unsupported network type. Use 'fc' for full connection or 'mesh' for mesh-grid.")
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

        partition = [[] for _ in range(len(self.qpus))] # 每个qpu上一个划分
        for i in range(self.qiskit_circ.num_qubits): # q[i]->server[j]
            # print(i, distribution.placement.placement[i])
            partition[distribution.placement.placement[i]].append(i)
        filename = f"./outputs/paths/{self.qiskit_circ.name[:11]}_{self.qiskit_circ.num_qubits}_pytket.txt"
        with open(filename, 'w') as file:
            file.write(' '.join(map(str, partition)) + '\n')