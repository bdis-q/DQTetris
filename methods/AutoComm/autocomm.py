from methods.AutoComm.gate_util import *
from methods.AutoComm.commute_func import *
from methods.AutoComm.merge_func import *

# assume gates are formed of CX and single-qubit gates. It is okay to have other gates if related rules are defined
def comm_aggregate(gate_list, qubit_node_mapping, allow_gate_pattern=True, allow_test_merge=False, refine_iter_cnt=3, check_commute_func=commute_func_right):
    if allow_gate_pattern == True:
        gate_list = pattern_merged_circ(gate_list)
    new_gate_block_list = consecutive_merge(gate_list, qubit_node_mapping)
    new_gate_block_list = linear_merge_iter(new_gate_block_list, qubit_node_mapping, refine_iter_cnt, check_commute_func, allow_test_merge)                    
    return new_gate_block_list

def _comm_block_tag(source_qb, target_node, gate_block, qubit_node_mapping):
    _state = []
    _remote_gates = []
    for glidx, glocal in enumerate(gate_block):
        glqb = gate_qubits(glocal)
        if len(glqb) == 2:
            if glqb[0] == source_qb:
                _state.append(1)
            elif glqb[1] == source_qb:
                _state.append(-1)
            if qubit_node_mapping[glqb[0]] != qubit_node_mapping[glqb[1]]:
                _remote_gates.append(glidx)
    use_cat_comm = True
    if sum(_state) == len(_state):
        for ii in range(len(_remote_gates)-1):
            for jj in range(_remote_gates[ii]+1,_remote_gates[ii+1]):
                gjj = gate_block[jj]
                # if there is interleaving gates
                gjjqb = gate_qubits(gjj)
                if len(gjjqb) == 1 and gjjqb[0] == source_qb:
                    if gate_type(gjj) not in ["RZ", "Z", "X"]:
                        use_cat_comm = False
                        break
            if use_cat_comm == False:
                break
    elif sum(_state) == -len(_state):
        for jj in _remote_gates:
            gjj = gate_block[jj]
            if gate_type(gjj) != gate_type(gate_block[_remote_gates[0]]):
                use_cat_comm = False
                break
            if gate_type(gjj) not in ["CX", "CZ"]:
                use_cat_comm = False
                break
        if use_cat_comm == True:
            for ii in range(len(_remote_gates)-1):
                for jj in range(_remote_gates[ii]+1,_remote_gates[ii+1]):
                    gjj = gate_block[jj]
                    gjjqb = gate_qubits(gjj)
                    if len(gjjqb) == 1 and gjjqb[0] == source_qb:
                        if gate_type(gate_block[_remote_gates[0]]) == "CX":
                            if gate_type(gjj) not in ["RX", "Z", "X"]:
                                use_cat_comm = False
                                break
                        elif gate_type(gate_block[_remote_gates[0]]) == "CZ":
                            if gate_type(gjj) not in ["RZ", "Z", "X"]:
                                use_cat_comm = False
                                break
                if use_cat_comm == False:
                    break
    else:
        use_cat_comm = False
    if use_cat_comm:
        return [[[source_qb, target_node],0], gate_block] # use cat-comm
    else:
        return [[[source_qb, target_node],1], gate_block] # use tp-comm

def comm_assign(gate_block_list, qubit_node_mapping):
    assigned_gate_block_list = []
    for gbidx, gb in enumerate(gate_block_list):
        if is_comm_block(gb):
            if gb[0] == []:
                if gb[1] == []:
                    continue
                for _g in gb[1]:
                    _gqb = gate_qubits(_g)
                    if len(_gqb) == 2:
                        if qubit_node_mapping[_gqb[0]] != qubit_node_mapping[_gqb[1]]:
                            break
                source_qb, target_node = _gqb[0], qubit_node_mapping[_gqb[1]]
                new_comm_block = _comm_block_tag(source_qb, target_node, gb[1], qubit_node_mapping)
                if new_comm_block[0][1] == 0:
                    assigned_gate_block_list.append(new_comm_block)
                else:
                    source_qb, target_node = _gqb[1], qubit_node_mapping[_gqb[0]]
                    new_comm_block = _comm_block_tag(source_qb, target_node, gb[1], qubit_node_mapping)
                    assigned_gate_block_list.append(new_comm_block)
            else:
                source_qb, target_node = gb[0]
                new_comm_block = _comm_block_tag(source_qb, target_node, gb[1], qubit_node_mapping)
                assigned_gate_block_list.append(new_comm_block)
        else:
            assigned_gate_block_list.append(gb)
    return assigned_gate_block_list

def comm_schedule(assigned_gate_block_list, qubit_node_mapping, latency_metric=None, fidelity_metric=None, refine_iter_cnt=3, check_commute_func=commute_func_right):
    if latency_metric == None:
        latency_metric = {"1Q":0.1,"CX":1,"CZ":1,"CRZ":2.2,"CU1":2.2,"MS":5,"EP":12,"CB":1}
    assigned_gate_block_list = tp_comm_merge_iter(assigned_gate_block_list, qubit_node_mapping, refine_iter_cnt, check_commute_func)
    # start scheduling
    dqb_list = [f"dq{i}" for i in range(len(qubit_node_mapping))]
    node_count = max(qubit_node_mapping) + 1
    cqb_list = [f"cq{i}-{j}" for j in [0,1] for i in range(node_count)]
    qb_slot = {}
    for qb in dqb_list+cqb_list:
        qb_slot[qb] = 0
    epr_cnt = 0
    comm_costs = [0, 0, 0, 0]
    for gb in assigned_gate_block_list:
        if is_comm_block(gb):
            if gb[0][1] == 0: # cat-comm
                source, target_node = gb[0][0]
                source_node = qubit_node_mapping[source]
                scqb_idx = 0
                if qb_slot[f"cq{source_node}-{0}"] > qb_slot[f"cq{source_node}-{1}"]:
                    scqb_idx = 1
                tcqb_idx = 0
                if qb_slot[f"cq{target_node}-{0}"] > qb_slot[f"cq{target_node}-{1}"]:
                    tcqb_idx = 1
                # do cat-comm
                # EP
                scqb = f"cq{source_node}-{scqb_idx}"
                tcqb = f"cq{target_node}-{tcqb_idx}"
                source_qb = f"dq{source}"
                qb_slot[scqb] = max(qb_slot[scqb],qb_slot[tcqb]) + latency_metric["EP"]
                qb_slot[tcqb] = qb_slot[scqb]
                # CX
                qb_slot[source_qb] = max(qb_slot[source_qb],qb_slot[scqb]) + latency_metric["CX"]
                qb_slot[scqb] = qb_slot[source_qb]
                # Measure and correction
                qb_slot[scqb] += latency_metric["MS"]
                qb_slot[tcqb] = max(qb_slot[tcqb],qb_slot[scqb]+latency_metric["CB"])+latency_metric["1Q"]
                # main body
                source_qb = tcqb
                H_offset = 0
                for glocal in gb[1]:
                    glqb = gate_qubits(glocal)
                    if len(glqb) == 1:
                        if glqb[0] == source:
                            qb_slot[source_qb] += latency_metric["1Q"]
                        else:
                            qb_slot[f"dq{glqb[0]}"] += latency_metric["1Q"]
                    elif len(glqb) == 2:
                        ctrl, tgt = glqb
                        ctrl_qb = f"dq{ctrl}"
                        tgt_qb = f"dq{tgt}"
                        if gate_type(glocal) in ["CX", "CZ"]:
                            twoq_latency = latency_metric["CX"]
                        elif gate_type(glocal) in ["CRZ", "CU1"]:
                            twoq_latency = latency_metric["CRZ"]
                        else:
                            pass
                        if ctrl == source:
                            qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[tgt_qb]) + twoq_latency
                            qb_slot[tgt_qb] = qb_slot[source_qb]
                        elif tgt == source:
                            qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[ctrl_qb]) + twoq_latency
                            qb_slot[ctrl_qb] = qb_slot[source_qb]
                            H_offset = 1
                        else:
                            qb_slot[ctrl_qb] = max(qb_slot[ctrl_qb], qb_slot[tgt_qb]) + twoq_latency
                            qb_slot[tgt_qb] = qb_slot[ctrl_qb]
                # finish up
                source_qb = f"dq{source}"
                qb_slot[tcqb] += latency_metric["1Q"]+latency_metric["MS"]
                qb_slot[source_qb] = max(qb_slot[source_qb],qb_slot[tcqb]+latency_metric["CB"])+latency_metric["1Q"]
                epr_cnt += 1
                comm_costs[0] += 1
            else: # Tp-comm
                # serial
                # do parallel
                if len(gb[0][0]) == 2:
                    source, target_node = gb[0][0]
                    source_node = qubit_node_mapping[source]
                    scqb_idx = 0
                    if qb_slot[f"cq{source_node}-{0}"] > qb_slot[f"cq{source_node}-{1}"]:
                        scqb_idx = 1
                    tcqb_idx = 0
                    if qb_slot[f"cq{target_node}-{0}"] > qb_slot[f"cq{target_node}-{1}"]:
                        tcqb_idx = 1
                    source_qb = f"dq{source}"
                    scqb = f"cq{source_node}-{scqb_idx}"
                    tcqb = f"cq{target_node}-{tcqb_idx}"
                    # do tp-comm
                    # EP
                    qb_slot[scqb] = max(qb_slot[scqb],qb_slot[tcqb]) + latency_metric["EP"]
                    qb_slot[tcqb] = qb_slot[scqb]
                    # CX
                    qb_slot[source_qb] = max(qb_slot[source_qb],qb_slot[scqb]) + latency_metric["CX"]
                    qb_slot[scqb] = qb_slot[source_qb]
                    # H
                    qb_slot[source_qb] = qb_slot[source_qb] + latency_metric["1Q"]
                    # M
                    qb_slot[source_qb] = qb_slot[source_qb] + latency_metric["MS"]
                    qb_slot[scqb] = qb_slot[scqb] + latency_metric["MS"]
                    qb_slot[tcqb] = max(qb_slot[tcqb],qb_slot[scqb]+latency_metric["CB"]) + latency_metric["1Q"]
                    qb_slot[tcqb] = max(qb_slot[tcqb],qb_slot[source_qb]+latency_metric["CB"]) + latency_metric["1Q"]
                    qb_slot[source_qb] += latency_metric["1Q"] # reset
                    # main body
                    source_qb = tcqb
                    for glocal in gb[1]:
                        glqb = gate_qubits(glocal)
                        if len(glqb) == 1:
                            if glqb[0] == source:
                                qb_slot[source_qb] += latency_metric["1Q"]
                            else:
                                qb_slot[f"dq{glqb[0]}"] += latency_metric["1Q"]
                        elif len(glqb) == 2:
                            ctrl, tgt = glqb
                            ctrl_qb = f"dq{ctrl}"
                            tgt_qb = f"dq{tgt}"
                            if gate_type(glocal) in ["CX", "CZ"]:
                                twoq_latency = latency_metric["CX"]
                            elif gate_type(glocal) in ["CRZ", "CU1"]:
                                twoq_latency = latency_metric["CRZ"]
                            else:
                                pass
                            if ctrl == source:
                                qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[tgt_qb]) + twoq_latency
                                qb_slot[tgt_qb] = qb_slot[source_qb]
                            elif tgt == source:
                                qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[ctrl_qb]) + twoq_latency
                                qb_slot[ctrl_qb] = qb_slot[source_qb]
                            else:
                                qb_slot[ctrl_qb] = max(qb_slot[ctrl_qb], qb_slot[tgt_qb]) + twoq_latency
                                qb_slot[tgt_qb] = qb_slot[ctrl_qb]
                    # finish up
                    source_qb = f"dq{source}"
                    tcqb_new = f"cq{target_node}-{1-tcqb_idx}"
                    qb_slot[scqb] = max(qb_slot[scqb],qb_slot[tcqb_new]) + latency_metric["EP"]
                    qb_slot[tcqb_new] = qb_slot[scqb]
                    qb_slot[source_qb] = max(qb_slot[source_qb],qb_slot[scqb]) + 3*latency_metric["CX"]
                    qb_slot[scqb] = qb_slot[source_qb]
                    qb_slot[tcqb_new] = max(qb_slot[tcqb_new],qb_slot[tcqb]) + latency_metric["CX"]
                    qb_slot[tcqb] = qb_slot[tcqb_new]
                    qb_slot[tcqb_new] += latency_metric["MS"]
                    qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[tcqb_new]+latency_metric["CB"]) + latency_metric["1Q"]
                    qb_slot[tcqb] += latency_metric["1Q"] + latency_metric["MS"]
                    qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[tcqb]+latency_metric["CB"]) + latency_metric["1Q"]
                    epr_cnt += 2
                    comm_costs[1] += 1
                # do serial
                if len(gb[0][0]) > 2:
                    source = gb[0][0][0]
                    target_nodes = gb[0][0][1:]
                    source_node = qubit_node_mapping[source]
                    prev_target = qubit_node_mapping[source]
                    scqb_idx_prev = 0
                    if qb_slot[f"cq{source_node}-{0}"] > qb_slot[f"cq{source_node}-{1}"]:
                        scqb_idx_prev = 1
                    source_qb_prev = f"dq{source}"
                    
                    for tnidx, target_node in enumerate(target_nodes):
                        source_node = prev_target
                        prev_target = target_node

                        scqb_idx = scqb_idx_prev
                        tcqb_idx = 0
                        if qb_slot[f"cq{target_node}-{0}"] > qb_slot[f"cq{target_node}-{1}"]:
                            tcqb_idx = 1
                        scqb_idx_prev = 1 - tcqb_idx
                        
                        source_qb = source_qb_prev
                        scqb = f"cq{source_node}-{scqb_idx}"
                        tcqb = f"cq{target_node}-{tcqb_idx}"
                        source_qb_prev = tcqb

                        # do tp-comm
                        # EP
                        qb_slot[scqb] = max(qb_slot[scqb],qb_slot[tcqb]) + latency_metric["EP"]
                        qb_slot[tcqb] = qb_slot[scqb]
                        # CX
                        qb_slot[source_qb] = max(qb_slot[source_qb],qb_slot[scqb]) + latency_metric["CX"]
                        qb_slot[scqb] = qb_slot[source_qb]
                        # H
                        qb_slot[source_qb] = qb_slot[source_qb] + latency_metric["1Q"]
                        # M
                        qb_slot[source_qb] = qb_slot[source_qb] + latency_metric["MS"]
                        qb_slot[scqb] = qb_slot[scqb] + latency_metric["MS"]
                        qb_slot[tcqb] = max(qb_slot[tcqb],qb_slot[scqb]+latency_metric["CB"]) + latency_metric["1Q"]
                        qb_slot[tcqb] = max(qb_slot[tcqb],qb_slot[source_qb]+latency_metric["CB"]) + latency_metric["1Q"]
                        qb_slot[source_qb] += latency_metric["1Q"] # reset
                        epr_cnt += 1
                        comm_costs[2] += 1

                        # main body
                        source_qb = tcqb
                        for glocal in gb[1+tnidx]:
                            glqb = gate_qubits(glocal)
                            if len(glqb) == 1:
                                if glqb[0] == source:
                                    qb_slot[source_qb] += latency_metric["1Q"]
                                else:
                                    qb_slot[f"dq{glqb[0]}"] += latency_metric["1Q"]
                            elif len(glqb) == 2:
                                ctrl, tgt = glqb
                                ctrl_qb = f"dq{ctrl}"
                                tgt_qb = f"dq{tgt}"
                                if gate_type(glocal) in ["CX", "CZ"]:
                                    twoq_latency = latency_metric["CX"]
                                elif gate_type(glocal) in ["CRZ", "CU1"]:
                                    twoq_latency = latency_metric["CRZ"]
                                else:
                                    pass
                                if ctrl == source:
                                    qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[tgt_qb]) + twoq_latency
                                    qb_slot[tgt_qb] = qb_slot[source_qb]
                                elif tgt == source:
                                    qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[ctrl_qb]) + twoq_latency
                                    qb_slot[ctrl_qb] = qb_slot[source_qb]
                                else:
                                    qb_slot[ctrl_qb] = max(qb_slot[ctrl_qb], qb_slot[tgt_qb]) + twoq_latency
                                    qb_slot[tgt_qb] = qb_slot[ctrl_qb]
                    # finish up
                    source_qb = f"dq{source}"
                    source_node = qubit_node_mapping[source]
                    scqb_idx = 0
                    if qb_slot[f"cq{source_node}-{0}"] > qb_slot[f"cq{source_node}-{1}"]:
                        scqb_idx = 1
                    scqb = f"cq{source_node}-{scqb_idx}"
                    tcqb_new = f"cq{target_node}-{1-tcqb_idx}"

                    qb_slot[scqb] = max(qb_slot[scqb],qb_slot[tcqb_new]) + latency_metric["EP"]
                    qb_slot[tcqb_new] = qb_slot[scqb]
                    qb_slot[source_qb] = max(qb_slot[source_qb],qb_slot[scqb]) + 3*latency_metric["CX"]
                    qb_slot[scqb] = qb_slot[source_qb]
                    qb_slot[tcqb_new] = max(qb_slot[tcqb_new],qb_slot[tcqb]) + latency_metric["CX"]
                    qb_slot[tcqb] = qb_slot[tcqb_new]
                    qb_slot[tcqb_new] += latency_metric["MS"]
                    qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[tcqb_new]+latency_metric["CB"]) + latency_metric["1Q"]
                    qb_slot[tcqb] += latency_metric["1Q"] + latency_metric["MS"]
                    qb_slot[source_qb] = max(qb_slot[source_qb], qb_slot[tcqb]+latency_metric["CB"]) + latency_metric["1Q"]
                    epr_cnt += 1
                    comm_costs[3] += 1
        else:
            gqb = gate_qubits(gb)
            if len(gqb) == 1:
                qb_slot[f"dq{gqb[0]}"] += latency_metric["1Q"]
            elif len(gqb) == 2: # must be local
                qb0, qb1 = gqb
                if gate_type(gb) in ["CX", "CZ"]:
                    qb_slot[f"dq{qb0}"] = max(qb_slot[f"dq{qb0}"], qb_slot[f"dq{qb1}"]) + latency_metric["CX"]
                    qb_slot[f"dq{qb1}"] = qb_slot[f"dq{qb0}"]
                elif gate_type(gb) in ["CRZ", "CU1"]:
                    qb_slot[f"dq{qb0}"] = max(qb_slot[f"dq{qb0}"], qb_slot[f"dq{qb1}"]) + latency_metric["CRZ"]
                    qb_slot[f"dq{qb1}"] = qb_slot[f"dq{qb0}"]
    all_latency =  max(qb_slot.values())
    return epr_cnt, all_latency, assigned_gate_block_list, comm_costs


def autocomm_full(gate_list, qubit_node_mapping, allow_gate_pattern=False, allow_test_merge=False, allow_local_view=False, aggregate_iter_cnt=1, schedule_iter_cnt=1):
    if allow_local_view == False:
        g_list = comm_aggregate(gate_list, qubit_node_mapping, allow_gate_pattern=allow_gate_pattern, allow_test_merge=allow_test_merge, refine_iter_cnt=aggregate_iter_cnt)
        assigned_gate_block_list = comm_assign(g_list, qubit_node_mapping)
        epr_cnt, all_latency, assigned_gate_block_list_scheduled, comm_costs = comm_schedule(assigned_gate_block_list, qubit_node_mapping, refine_iter_cnt=schedule_iter_cnt)
        return epr_cnt, all_latency, assigned_gate_block_list_scheduled, comm_costs
    else:
        g_list = []
        for sub_gate_list in gate_list:
            g_list += comm_aggregate(sub_gate_list, qubit_node_mapping, allow_gate_pattern=allow_gate_pattern, allow_test_merge=allow_test_merge, refine_iter_cnt=aggregate_iter_cnt)
        assigned_gate_block_list = comm_assign(g_list, qubit_node_mapping)
        epr_cnt, all_latency, assigned_gate_block_list_scheduled, comm_costs = comm_schedule(assigned_gate_block_list, qubit_node_mapping, refine_iter_cnt=schedule_iter_cnt)
        return epr_cnt, all_latency, assigned_gate_block_list_scheduled, comm_costs

if __name__ == "__main__":
    print(comm_assign([[[],[["CX",[0,1]],["RX",[0]],["RX",[1]],["CX",[0,1]]]]],[0,2]))
