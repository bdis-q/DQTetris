import warnings
import time
import sys
from utils.inout import *
from utils.qcircuits import select_circuit
from methods.oee import OEE, FGP_rOEE
from methods.pytketdqc import Pytket_DQC
from methods.wbcp import WBCP
from methods.dqtetris import DQTetris

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    circ, trans_circ, qpus = select_circuit(args.circuit_name,
                                            args.qubit_count,
                                            args.core_count,
                                            args.core_capacities)
    distributors = []

    # Static OEE
    print("[Static OEE]", file=sys.stderr)
    oee = OEE(trans_circ, qpus)
    distributors.append(oee)
    oee.distribute()

    # FGP_rOEE
    print("[FGP_rOEE]", file=sys.stderr)
    fgp = FGP_rOEE(trans_circ, qpus, 10)
    distributors.append(fgp)
    fgp.distribute()

    # Pytket DQC
    print("[Pytket_DQC (CE)]", file=sys.stderr)
    ce = Pytket_DQC(circ, qpus, "CE")
    distributors.append(ce)
    ce.distribute()

    # Pytket DQC
    print("[Pytket_DQC (PA)]", file=sys.stderr)
    pa = Pytket_DQC(circ, qpus, "PA")
    distributors.append(pa)
    pa.distribute()

    # Window-based circuit partitioning
    print("[WBCP]", file=sys.stderr)
    wbcp = WBCP(trans_circ, qpus)
    distributors.append(wbcp)
    wbcp.distribute()

    # Ours
    print("[DQTetris]", file=sys.stderr)
    dqtetris = DQTetris(trans_circ, qpus)
    distributors.append(dqtetris)
    dqtetris.distribute()

    output_results(args.circuit_name, trans_circ, qpus, distributors)
    return

if __name__ == "__main__":
    args = get_args()
    filename = f"{args.circuit_name}_{args.qubit_count}_{args.core_count}"
    # filename = "output"
    original_stdout = sys.stdout
    with open(f'outputs/{filename}.txt', 'a') as f:
        sys.stdout = f
        start_time = time.time()
        main(args)
        end_time = time.time()
        print(f"[Total Runtime] {end_time - start_time} seconds\n\n")
        sys.stdout = original_stdout