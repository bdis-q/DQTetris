'''g_list: a list of gates
'''
import numpy as np

def gate_list_to_layer(g_list):
    num_q_slot = max([max(g[1]) for g in g_list]) + 1
    q_slot = [0 for i in range(num_q_slot)]
    layer_list = []
    for g in g_list:
        g_layer_id = max([q_slot[qb] for qb in g[1]])
        if g_layer_id >= len(layer_list):
            layer_list.append([])
        layer_list[g_layer_id].append(g)
        for qb in g[1]:
            q_slot[qb] = g_layer_id + 1
    return layer_list

def gate_type(g): return g[0]
def gate_qubits(g): return [qb for qb in g[1]]
def gate_params(g): 
    if len(g) > 2:
        return [param for param in g[2]]
    else:
        return []

def is_equal_gate(g0, g1): return g0 == g1

def build_gate(name, qubits, params=[], global_phase=1):
    return [name, qubits, params, global_phase]

def build_T_gate(qb):
    return build_gate("RZ", [qb], [1j*np.pi/4], global_phase=np.exp(-1j*np.pi/8))

def build_S_gate(qb):
    return build_gate("RZ", [qb], [1j*np.pi/2], global_phase=np.exp(-1j*np.pi/4))

def build_Tdg_gate(qb):
    return build_gate("RZ", [qb], [-1j*np.pi/4], global_phase=np.exp(1j*np.pi/8))

def build_Sdg_gate(qb):
    return build_gate("RZ", [qb], [-1j*np.pi/2], global_phase=np.exp(1j*np.pi/4))

def build_RZ_gate(qb, angle, global_phase=1):
    return build_gate("RZ", [qb], [angle], global_phase=global_phase)

def build_RX_gate(qb, angle, global_phase=1):
    return build_gate("RX", [qb], [angle], global_phase=global_phase)

def build_H_gate(qb):
    return build_gate("H", [qb])

def build_CX_gate(ctrl, target):
    return build_gate("CX", [ctrl, target])

def build_CZ_gate(ctrl, target):
    return build_gate("CZ", [ctrl, target])

def build_CRZ_gate(ctrl, target, angle):
    return build_gate("CRZ", [ctrl, target], [angle])

def build_CU1_gate(ctrl, target, angle):
    return build_gate("CU1", [ctrl, target], [angle])

def build_toffoli_gate(qa, qb, qc):
    gate_list = []
    gate_list.append(build_H_gate(qc))
    gate_list.append(build_CX_gate(qb,qc))
    gate_list.append(build_Tdg_gate(qc))
    gate_list.append(build_CX_gate(qa,qc))
    gate_list.append(build_T_gate(qc))
    gate_list.append(build_CX_gate(qb,qc))
    gate_list.append(build_Tdg_gate(qc))
    gate_list.append(build_CX_gate(qa,qc))
    gate_list.append(build_T_gate(qb))
    gate_list.append(build_T_gate(qc))
    gate_list.append(build_H_gate(qc))
    gate_list.append(build_CX_gate(qa,qb))
    gate_list.append(build_Tdg_gate(qb))
    gate_list.append(build_CX_gate(qa,qb))
    gate_list.append(build_T_gate(qa))
    return gate_list

def crz_merge(g_list):
    layer_list = [[g] for g in g_list] # gate_list_to_layer(g_list)
    layer_qb_dict_list = []
    layer_qb_dict_list_control = []
    layer_qb_dict_list_target = []
    layer_gate_deleted = []
    for layer in layer_list:
        qb_dict = {}
        qb_dict_control = {}
        qb_dict_target = {}
        g_del_flag = []
        for gidx, g in enumerate(layer):
            qubits = gate_qubits(g)
            qb_dict[tuple(qubits)] = [gidx, g]
            if gate_type(g) == "CX":
                qb_dict_control[qubits[0]] = g
                qb_dict_target[qubits[1]] = g
            elif len(qubits) == 1:
                qb_dict_target[qubits[0]] = g
            g_del_flag.append(0)
        layer_qb_dict_list.append(qb_dict)
        layer_qb_dict_list_control.append(qb_dict_control)
        layer_qb_dict_list_target.append(qb_dict_target)
        layer_gate_deleted.append(g_del_flag)

    new_gate_list = []
    for lidx, layer in enumerate(layer_list):
        for gidx0, g0 in enumerate(layer):
            if layer_gate_deleted[lidx][gidx0] == 0:
                if gate_type(g0) == "CX" and (lidx+2) < len(layer_list):
                    qb = gate_qubits(g0)
                    next_lqd = layer_qb_dict_list[lidx+1]
                    next_next_lqd = layer_qb_dict_list[lidx+2]
                    _merge_flag = False
                    if tuple([qb[1]]) in next_lqd and tuple(qb) in next_next_lqd:
                        gidx1, g1 = next_lqd[tuple([qb[1]])]
                        gidx2, g2 = next_next_lqd[tuple(qb)]
                        if gate_type(g1) == "RZ" and gate_type(g2) == "CX":
                            if qb[0] in layer_qb_dict_list_control[lidx+1]:
                                _merge_flag = True
                            elif qb[0] not in layer_qb_dict_list_target[lidx+1]:
                                _merge_flag = True
                            else:
                                g3 = layer_qb_dict_list_target[lidx+1][qb[0]]
                                if gate_type(g3) == "RZ":
                                    _merge_flag = True
                    if _merge_flag:
                        layer_gate_deleted[lidx+1][gidx1] = 1
                        layer_gate_deleted[lidx+2][gidx2] = 1
                        new_gate_list.append(build_gate("CRZ", qb, [-2*param for param in gate_params(g1)]))
                        new_gate_list.append(build_gate("RZ", qb[1:], [param for param in gate_params(g1)]))
                    else:
                        new_gate_list.append(g0)
                else:
                    new_gate_list.append(g0)
            else:
                pass
    gate_del_flag = [0 for i in range(len(new_gate_list))]
    r_gate_list = []
    for gidx, g in enumerate(new_gate_list):
        if gate_del_flag[gidx] == 0:
            if gidx == len(new_gate_list) - 1:
                break
            g1 = new_gate_list[gidx+1]
            if gate_type(g) == "RZ":
                if gate_type(g1) == gate_type(g) and gate_qubits(g) == gate_qubits(g1) and np.isclose(gate_params(g)[0],-gate_params(g1)[0]):
                    gate_del_flag[gidx] = 1
                    gate_del_flag[gidx+1] = 1
                else:
                    r_gate_list.append(g)
            else:
                r_gate_list.append(g)
    return r_gate_list

def pattern_merged_circ(g_list, pattern_func_list=[crz_merge]):
    for pattern_func in pattern_func_list:
        g_list = pattern_func(g_list)
    return g_list

def remove_repeated_gates(gate_list):
    n_gate = len(gate_list)
    gate_del_flag = [0 for i in range(n_gate)]
    new_gate_list = []
    for gidx0, g0 in enumerate(gate_list):
        if gate_del_flag[gidx0] == 0:
            for gidx1 in range(gidx0+1, n_gate):
                g1 = gate_list[gidx1]
                if is_equal_gate(g0, g1):
                    gate_del_flag[gidx0] = 1
                    gate_del_flag[gidx1] = 1
                    break
            if gate_del_flag[gidx0] == 0:
                new_gate_list.append(g0)
    return new_gate_list

if __name__ == "__main__":
    g_list = [build_gate("RZ", [0], [0.1]),build_gate("CX", [1,2]),build_gate("RZ", [2], [0.1]),build_gate("CX", [1,2]), \
              build_gate("CX", [2,3]),build_gate("RZ", [3], [0.1]),build_gate("RZ", [2], [0.1]),build_gate("CX", [2,3]), build_gate("RZ", [0], [0.1]) \
            ]
    print(crz_merge(g_list))
                        
    
