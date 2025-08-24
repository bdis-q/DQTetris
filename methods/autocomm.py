from methods.AutoComm.gate_util import *
from methods.AutoComm.merge_func import *
from methods.AutoComm.commute_func import *
from methods.AutoComm.autocomm import *
import time
from qiskit import transpile

class QAutoComm:
    def __init__(self, circ, qpus):
        self.circ = circ
        self.qpus = qpus
        return

    @property
    def name(self):
        return "AutoComm"

    def transpile(self):
        self.gate_list = []
        # qubit_node_mapping
        global_phase = self.circ.global_phase
        # print(f"[DEBUG] global_phase: {global_phase}")

        # for instruction, qubits, _ in self.circ.data:
        for instruction in self.circ:
            gate = instruction.operation
            gate_name = gate.name
            qubits = [qubit._index for qubit in instruction.qubits]
            if qubits[0] == None:
                qubits = [self.circ.qubits.index(qubit) for qubit in instruction.qubits]
            params = list(gate.params) if gate.params else []
            # print(f"[DEBUG] gate_name: {gate_name}, qubits: {qubits}, params: {params}")

            # 处理特定门
            if gate_name == "u3":
                gate = build_gate("U3", qubits, params, global_phase=global_phase)
            elif gate_name == "rz":
                gate = build_RZ_gate(qubits[0], angle=params[0], global_phase=global_phase)
            elif gate_name == "cx":
                gate = build_CX_gate(qubits[0], qubits[1])
            elif gate_name == "cz":
                gate = build_CZ_gate(qubits[0], qubits[1])
            elif gate_name == "crz":
                gate = build_CRZ_gate(qubits[0], qubits[1], angle=params[0])
            elif gate_name == "cu1":
                gate = build_CU1_gate(qubits[0], qubits[1], angle=params[0])
            elif gate_name == "h":
                gate = build_H_gate(qubits[0])
            else:
                # 可以根据需要添加更多门的处理
                raise ValueError(f"Unsupported gate type: {gate_name}")

            self.gate_list.append(gate)
        return

    def generate_qubit_node_mapping(self):
        self.qubit_node_mapping = []
        node_index = 0
        remaining_capacity = self.qpus[node_index]

        for _ in range(self.circ.num_qubits):
            if remaining_capacity == 0:
                node_index += 1
                remaining_capacity = self.qpus[node_index]

            self.qubit_node_mapping.append(node_index)
            remaining_capacity -= 1

        return

    def distribute(self):
        print("[AutoComm]")
        start_time = time.time()
        # change a qiskit circuit to a gate list
        self.transpile()
        # 根据self.qpu构建初始的qubit_node_mapping
        self.generate_qubit_node_mapping()
        # print(f"[DEBUG] qubit_node_mapping: {self.qubit_node_mapping}")
        num_q = len(self.qubit_node_mapping)
        qb_per_node = max(self.qpus)
        epr_cnt, all_latency, assigned_gate_block_list1, comm_costs = autocomm_full(self.gate_list, 
                                                                        self.qubit_node_mapping, 
                                                                        aggregate_iter_cnt=num_q//qb_per_node, 
                                                                        schedule_iter_cnt=num_q//qb_per_node)
        end_time = time.time()
        self.num_gates = comm_costs[0]
        self.num_swaps = comm_costs[1] + comm_costs[2] + comm_costs[3]
        self.num_comms = self.num_gates + self.num_swaps
        self.exec_time = end_time - start_time
        print(f"#epr: {epr_cnt}, latency: {all_latency}, comm_costs: {comm_costs}")
        print(f"#comms: {self.num_comms}")
        print(f"#gate_comms: {self.num_gates}") # cat-comm
        print(f"#swap_comms: {self.num_swaps}") # tp-comm
        print(f"Time: {self.exec_time} seconds\n\n")
        return

def QFT(num_qubits, qb_per_node):
    gate_list = []
    for i in range(num_qubits-1):
        gate_list.append(build_H_gate(i))
        for j in range(i+1, num_qubits):
            gate_list.append(build_CX_gate(j,i))
            gate_list.append(build_RZ_gate(i,angle=-np.pi/4/2**(j-i)))
            gate_list.append(build_CX_gate(j,i))
            gate_list.append(build_RZ_gate(i,angle=np.pi/4/2**(j-i)))
    qubit_node_mapping = [i//qb_per_node for i in range(num_qubits)] # optimal mapping obtained
    return gate_list, qubit_node_mapping

if __name__ == "__main__":
    num_q, qb_per_node = 30, 10
    gate_list, qubit_node_mapping = QFT(num_q, qb_per_node)
    print("QFT gate list:", gate_list)
    epr_cnt, all_latency, assigned_gate_block_list1, comm_costs = autocomm_full(gate_list, qubit_node_mapping, allow_gate_pattern=True, aggregate_iter_cnt=1, schedule_iter_cnt=num_q//qb_per_node)
    print(epr_cnt, all_latency)