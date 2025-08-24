from methods.AutoComm.gate_util import *

# you can write your own
def commute_func_right(lblk, rblk): # right to left
    is_commute = False
    if lblk == [] or rblk == []:
        return True, -1, -1, lblk, rblk
    
    new_lblk = [] # the lblk after moving after rblk
    for lgidx, lg in enumerate(reversed(lblk)):
        cur_check_point = [lg]
        new_rblk = []
        for rgidx, rg in enumerate(rblk):
            rgtype = gate_type(rg)
            new_check_point = []
            new_rg = rg
            for cur_lg in reversed(cur_check_point):
                lgtype = gate_type(cur_lg)
                lgqb = gate_qubits(cur_lg)
                rgqb = gate_qubits(rg)
                if lgtype in ["RZ"]:
                    if rgtype in ["RZ", "Z"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["RX"]:
                        if lgqb[0] != rgqb[0]:
                            is_commute = True
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                    elif rgtype in ["X"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            new_check_point.append(build_gate(gate_type(cur_lg), gate_qubits(cur_lg), [-gate_params(cur_lg)[0]]))
                        else:
                            new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["H"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            new_check_point.append(build_gate("RX", gate_qubits(cur_lg), gate_params(cur_lg)))
                        else:
                            new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["CX", "CRX"]:
                        is_commute = True
                        if lgqb[0] not in rgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[0] == rgqb[0]: # on control line
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                is_commute = False
                    elif rgtype in ["CZ","CRZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                elif lgtype in ["RX"]:
                    if rgtype in ["RX", "X"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["RZ"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            is_commute = False
                        else:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                    elif rgtype in ["Z"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            new_check_point.append(build_gate(gate_type(cur_lg), gate_qubits(cur_lg), [-gate_params(cur_lg)[0]]))
                        else:
                            new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["H"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            new_check_point.append(build_gate("RZ", gate_qubits(cur_lg), gate_params(cur_lg)))
                        else:
                            new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["CX", "CRX"]:
                        is_commute = True
                        if lgqb[0] not in rgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[0] == rgqb[1]: # on target line
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                is_commute = False
                    elif rgtype in ["CZ","CRZ"]:
                        is_commute = True
                        if lgqb[0] not in rgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            is_commute = False
                elif lgtype in ["H"]:
                    if rgtype in ["RX"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        if lgqb[0] == rgqb[0]:
                            new_rg = (build_gate("RZ", gate_qubits(rg), gate_params(rg)))
                        else:
                            new_rg = (rg)
                    elif rgtype in ["RZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        if lgqb[0] == rgqb[0]:
                            new_rg = (build_gate("RX", gate_qubits(rg), gate_params(rg)))
                        else:
                            new_rg = (rg)
                    elif rgtype in ["Z"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        if lgqb[0] == rgqb[0]:
                            new_rg = (build_gate("X", gate_qubits(rg), gate_params(rg)))
                        else:
                            new_rg = (rg)
                    elif rgtype in ["X"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        if lgqb[0] == rgqb[0]:
                            new_rg = (build_gate("Z", gate_qubits(rg), gate_params(rg)))
                        else:
                            new_rg = (rg)
                    elif rgtype in ["H"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["CX", "CRX"]:
                        is_commute = True
                        if lgqb[0] not in rgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[0] == rgqb[1]: # on target line
                                new_check_point.append(cur_lg)
                                new_rg = (build_gate(rgtype[:-1]+"Z", gate_qubits(rg), gate_params(rg)))
                            else:
                                is_commute = False # could commute, current not implemented
                    elif rgtype in ["CZ","CRZ"]:
                        is_commute = True
                        if lgqb[0] not in rgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[0] == rgqb[1]: # on target line
                                new_check_point.append(cur_lg)
                                new_rg = (build_gate(rgtype[:-1]+"X", gate_qubits(rg), gate_params(rg)))
                            else:
                                is_commute = False # could commute, current not implemented
                elif lgtype == "X":
                    if rgtype in ["RX"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["RZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        if lgqb[0] == rgqb[0]:
                            new_rg = (build_gate("RZ", gate_qubits(rg), [-param for param in gate_params(rg)]))
                        else:
                            new_rg = (rg)
                    elif rgtype in ["Z"]: # a global phase
                        is_commute = True
                        new_check_point.append(cur_lg)
                        if lgqb[0] == rgqb[0]:
                            new_rg = (build_gate("Z", gate_qubits(rg), gate_params(rg)))
                        else:
                            new_rg = (rg)
                    elif rgtype in ["X"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["H"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            new_check_point.append(build_gate("Z", gate_qubits(cur_lg), gate_params(cur_lg)))
                        else:
                            new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["CX", "CRX"]:
                        is_commute = True
                        if lgqb[0] not in rgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[0] == rgqb[1]: # on target line
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else: # on control line
                                new_check_point.append(cur_lg)
                                new_check_point.append(build_gate(rgtype[1:],gate_qubits(rg)[1:],[-param for param in gate_params(rg)]))
                                new_rg = (rg)
                    elif rgtype in ["CZ","CRZ"]:
                        is_commute = True
                        if lgqb[0] not in rgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[0] == rgqb[1]: # on target line
                                new_check_point.append(cur_lg)
                                if rgtype != "CRZ":
                                    new_check_point.append(build_gate(rgtype[1:],gate_qubits(rg)[:1],[-param for param in gate_params(rg)]))
                                else:
                                    is_commute = False
                                new_rg = (rg)
                            else: # on control line
                                new_check_point.append(cur_lg)
                                new_check_point.append(build_gate(rgtype[1:],gate_qubits(rg)[1:],[-param for param in gate_params(rg)]))
                                new_rg = (rg)                          
                elif lgtype == "Z":
                    if rgtype in ["RZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["RX"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        if lgqb[0] == rgqb[0]:
                            new_rg = (build_gate("RX", gate_qubits(rg), [-param for param in gate_params(rg)]))
                        else:
                            new_rg = (rg)
                    elif rgtype in ["X"]: # a global phase
                        is_commute = True
                        new_check_point.append(cur_lg)
                        if lgqb[0] == rgqb[0]:
                            new_rg = (build_gate("X", gate_qubits(rg), gate_params(rg)))
                        else:
                            new_rg = (rg)
                    elif rgtype in ["Z"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["H"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            new_check_point.append(build_gate("X", gate_qubits(cur_lg), gate_params(cur_lg)))
                        else:
                            new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["CZ", "CRZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["CX","CRX"]:
                        is_commute = True
                        if lgqb[0] not in rgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[0] == rgqb[1]: # on target line
                                new_check_point.append(cur_lg)
                                if rgtype == "CX":
                                    new_check_point.append(build_gate("Z",gate_qubits(rg)[:1],[-param for param in gate_params(rg)]))
                                new_rg = (rg)
                            else: # on control line
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                elif lgtype in ["CX"]:
                    if rgtype in ["RX"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if rgqb[0] == lgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                is_commute = False
                    elif rgtype in ["RZ"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if rgqb[0] == lgqb[0]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                is_commute = False
                    elif rgtype in ["Z"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if rgqb[0] == lgqb[0]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                new_check_point.append(build_gate('Z', lgqb[:1], gate_params(cur_lg)))
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                    elif rgtype in ["X"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if rgqb[0] == lgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                new_check_point.append(build_gate('X', lgqb[1:], gate_params(cur_lg)))
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                    elif rgtype in ["H"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[1] == rgqb[0]:
                                new_check_point.append(build_gate("CZ", gate_qubits(cur_lg), gate_params(cur_lg)))
                                new_rg = (rg)
                            else:
                                is_commute = False # not implemented
                    elif rgtype in ["CX", "CRX"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        elif lgqb[0] == rgqb[1]:
                            is_commute = False
                        elif lgqb[1] == rgqb[0]:
                            is_commute = False
                        else:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                    elif rgtype in ["CZ","CRZ"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            if lgqb[1] != rgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                new_check_point.append(cur_lg)
                                new_rg = (build_gate(rgtype, rgqb, [-param for param in gate_params(rg)]))
                        elif lgqb[0] == rgqb[1]:
                            if lgqb[1] != rgqb[0]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                new_check_point.append(cur_lg)
                                new_rg = (build_gate(rgtype, rgqb, [-param for param in gate_params(rg)]))
                        elif lgqb[1] == rgqb[0]:
                            is_commute = False
                        else:
                            is_commute = False
                elif lgtype in ["CZ"]:
                    if rgtype in ["RZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["RX"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            is_commute = False
                    elif rgtype in ["X"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if rgqb[0] == lgqb[0]:
                                new_check_point.append(build_gate('X', lgqb[1:], gate_params(cur_lg)))
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                new_check_point.append(build_gate('X', lgqb[:1], gate_params(cur_lg)))
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                    elif rgtype in ["Z"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["H"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[1] == rgqb[0]:
                                new_check_point.append(build_gate("CX", gate_qubits(cur_lg), gate_params(cur_lg)))
                            else:
                                new_check_point.append(build_gate("CX", gate_qubits(cur_lg)[::-1], gate_params(cur_lg)))
                            new_rg = (rg)
                    elif rgtype in ["CZ", "CRZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["CX","CRX"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            if lgqb[1] != rgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                new_check_point.append(cur_lg)
                                new_rg = (build_gate(rgtype, rgqb, [-param for param in gate_params(rg)]))
                        elif lgqb[0] == rgqb[1]:
                            is_commute = False
                        elif lgqb[1] == rgqb[0]:
                            if lgqb[0] != rgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                new_check_point.append(cur_lg)
                                new_rg = (build_gate(rgtype, rgqb, [-param for param in gate_params(rg)]))
                        else:
                            is_commute = False
                elif lgtype in ["CRZ"]:
                    if rgtype in ["RZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["RX"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            is_commute = False
                    elif rgtype in ["X"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if rgqb[0] == lgqb[0]:
                                new_check_point.append(build_gate('RZ', lgqb[1:], [-param for param in gate_params(cur_lg)]))
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                is_commute = False
                    elif rgtype in ["Z"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["H"]:
                        is_commute = True
                        if rgqb[0] not in lgqb:
                            new_check_point.append(cur_lg)
                            new_rg = (rg)
                        else:
                            if lgqb[1] == rgqb[0]:
                                new_check_point.append(build_gate("CRX", gate_qubits(cur_lg), gate_params(cur_lg)))
                                new_rg = (rg)
                            else:
                                is_commute = True  
                    elif rgtype in ["CZ", "CRZ"]:
                        is_commute = True
                        new_check_point.append(cur_lg)
                        new_rg = (rg)
                    elif rgtype in ["CX"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            if lgqb[1] != rgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                new_check_point.append(build_gate(lgtype, lgqb, [-param for param in gate_params(cur_lg)]))
                                new_rg = (rg)
                        elif lgqb[0] == rgqb[1]:
                            is_commute = False
                        elif lgqb[1] == rgqb[0]:
                            if lgqb[0] != rgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                is_commute = False
                        else:
                            is_commute = False
                    elif rgtype in ["CRX"]:
                        is_commute = True
                        if lgqb[0] == rgqb[0]:
                            if lgqb[1] != rgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                is_commute = False
                        elif lgqb[0] == rgqb[1]:
                            is_commute = False
                        elif lgqb[1] == rgqb[0]:
                            if lgqb[0] != rgqb[1]:
                                new_check_point.append(cur_lg)
                                new_rg = (rg)
                            else:
                                is_commute = False
                        else:
                            is_commute = False
                    else: # not implemented
                        pass
                if is_commute == False:
                    return False, lgidx, rgidx, [], []
                else:
                    rg = new_rg
            new_check_point = remove_repeated_gates(new_check_point)
            cur_check_point = new_check_point[::-1]
            new_rblk.append(new_rg)
        rblk = new_rblk
        new_lblk.extend(remove_repeated_gates(new_check_point))
    new_lblk = new_lblk[::-1]
    return True, -1, -1, new_lblk, new_rblk