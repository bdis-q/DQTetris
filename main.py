import warnings
import time
import sys
from utils.qcircuits import *
from utils.inout import *
from methods.oee import OEE, FGP_rOEE
from methods.pytketdqc import Pytket_DQC
from methods.wbcp import WBCP
from methods.autocomm import QAutoComm
from methods.dqtetris import DQTetris

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    circ, trans_circ, qpus = select_circuit(args.circuit_name,
                                            args.qubit_count,
                                            args.core_count,
                                            args.core_capacities,
                                            args.gate_set)
    distributors = []

    # Static OEE
    print("[Static OEE]", file=sys.stderr)
    oee = OEE(trans_circ, qpus, network=args.network)
    distributors.append(oee)
    oee.distribute()

    # FGP_rOEE
    print("[FGP_rOEE]", file=sys.stderr)
    fgp = FGP_rOEE(trans_circ, qpus, 10, network=args.network)
    distributors.append(fgp)
    fgp.distribute()

    # Pytket DQC
    if args.gate_set == ["cu1", "u3"]:
        print("[Pytket_DQC (CE)]", file=sys.stderr)
        ce = Pytket_DQC(circ, qpus, "CE", network=args.network)
        distributors.append(ce)
        ce.distribute()

    # Pytket DQC
    if args.gate_set == ["cu1", "u3"]:
        print("[Pytket_DQC (PA)]", file=sys.stderr)
        pa = Pytket_DQC(circ, qpus, "PA", network=args.network)
        distributors.append(pa)
        pa.distribute()

    # Window-based circuit partitioning
    print("[WBCP]", file=sys.stderr)
    wbcp = WBCP(trans_circ, qpus, network=args.network)
    distributors.append(wbcp)
    wbcp.distribute()

    # AutoComm
    if args.network == "fc":
        print("[AutoComm]", file=sys.stderr)
        qautocomm = QAutoComm(trans_circ, qpus)
        distributors.append(qautocomm)
        qautocomm.distribute()

    # Ours
    print("[DQTetris]", file=sys.stderr)
    dqtetris = DQTetris(trans_circ, qpus, network=args.network)
    distributors.append(dqtetris)
    dqtetris.distribute()

    output_results(args.circuit_name, trans_circ, qpus, distributors)
    return

if __name__ == "__main__":
    args = get_args()
    filename = f"{args.circuit_name}_{args.qubit_count}_{args.core_count}"
    original_stdout = sys.stdout
    with open(f'outputs/{filename}.txt', 'a') as f:
        sys.stdout = f
        start_time = time.time()
        main(args)
        end_time = time.time()
        print(f"[Total Runtime] {end_time - start_time} seconds\n\n")
        sys.stdout = original_stdout