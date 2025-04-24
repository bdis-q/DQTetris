import math
from methods.oee import *

class WBCP(FGP_rOEE):
    @property
    def name(self):
        return "WBCP"
    
    def distribute(self):
        print("[WBCP]")
        start_time = time.time()
        self.k_way_WBCP()
        end_time = time.time()
        print(f"#depths: {self.num_depths}, win_len: {self.win_len}, #subcs: {self.num_subc}")
        print(f"#comms: {self.num_comms}")
        print(f"#gate_comms: {self.num_gates}")
        print(f"#swap_comms: {self.num_swaps}")
        self.exec_time = end_time - start_time
        print(f"Time: {self.exec_time} seconds\n\n")
        self.save_path_to(f"./outputs/paths/{self.circ.name[:11]}_{self.circ.num_qubits}_WBCP")
        return

    def k_way_WBCP(self):
        self.dag = circuit_to_dag(self.circ)
        self.num_depths = self.circ.depth()
        self.win_len = self.num_depths // 20
        if self.win_len == 0:
            self.win_len = self.num_depths
        self.num_subc = math.ceil(self.num_depths / self.win_len)

        # split qc into 'num_subc' sub-circuits
        # build the qubit interaction graph for each sub-circuit
        self.path = []
        self.num_comms = 0
        self.num_gates = 0
        self.num_swaps = 0

        # 第一个子线路直接用OEE算法得到初始划分
        right = min(self.win_len-1, self.num_depths-1)
        sub_qc = self.get_subcircuit_by_levels((0, right))
        self.build_qubit_interaction_graph(sub_qc) # -> self.qig
        self.k_way_OEE(self.qig)
        self.path.append(copy.deepcopy(self.partitions))
        self.num_gates += self.count_cut_edges(self.qig, self.partitions)

        for i in range(1, self.num_subc):
            # 获取子线路段
            right = min((i+1)*self.win_len-1, self.num_depths-1)
            sub_qc = self.get_subcircuit_by_levels((i*self.win_len, right))
            # 构造子线路的qubit interaction graph
            self.build_qubit_interaction_graph(sub_qc) # -> self.qig
            # 构造子线路带权重的qubit interaction graph，WBCP特有
            weighted_qig = self.build_weighted_qigraph(sub_qc, self.path[-1])
            self.k_way_OEE(weighted_qig)

            # calculate the entanglement costs
            # 1. 继续采用上一个partitions，即self.path[-1]
            ecosts = self.count_cut_edges(self.qig, self.path[-1])
            # 2. 采用新计算出来的self.partitions
            tmp_gates = self.count_cut_edges(self.qig, self.partitions)
            tmp_swaps = self.calculate_nonlocal_communications(self.path[-1], self.partitions)
            ecosts_new = tmp_gates + tmp_swaps
            if ecosts_new <= ecosts:
                self.path.append(copy.deepcopy(self.partitions))
                self.num_gates += tmp_gates
                self.num_swaps += tmp_swaps
            else:
                self.path.append(copy.deepcopy(self.path[-1]))
                self.num_gates += ecosts
        self.num_comms = self.num_gates + self.num_swaps
        return

    def get_subcircuit_by_levels(self, level_range):
        self.layers = list(self.dag.layers())
        sub_dag = self.dag.copy_empty_like() # 构造新的子线路的 DAG
        for level in range(level_range[0], level_range[1] + 1):
            for node in self.layers[level]["graph"].op_nodes():
                sub_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        # dag_drawer(sub_dag, scale=0.8, filename=f"dag_{level_range}.png")
        sub_qc = dag_to_circuit(sub_dag)
        return sub_qc

    def build_weighted_qigraph(self, circuit, prev_partitions):
        G = nx.Graph()
        for node in range(circuit.num_qubits):
            G.add_node(node) # 添加num_qubits个节点

        # 记录每个qubits所属的分区编号
        qubit_partition = {}
        for i, partition in enumerate(prev_partitions):
            for qubit in partition:
                qubit_partition[qubit] = i
        
        for instruction in circuit:
            qubits = [qubit._index for qubit in instruction.qubits]
            if qubits[0] == None:
                qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
            if len(qubits) > 1:
                if instruction.name == "barrier":
                    continue
                assert(len(qubits) == 2)
                edge_weight = 1
                # 检查qubits[0]和qubits[1]是否在同一个分区
                if qubit_partition[qubits[0]] == qubit_partition[qubits[1]]:
                    edge_weight = 2
                if G.has_edge(qubits[0], qubits[1]):
                    G[qubits[0]][qubits[1]]['weight'] += edge_weight
                else:
                    G.add_edge(qubits[0], qubits[1], weight = edge_weight)
        return G
