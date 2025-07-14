import argparse
from typing import List
import pandas as pd

def parse_int_list(input_str: str) -> List[int]:
    """Convert a comma-separated string to a list of integers (e.g., '4,6,8' -> [4,6,8])"""
    try:
        return [int(item.strip()) for item in input_str.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer list format: '{input_str}'. Please use comma-separated integers, e.g., '4,6,8'")

def get_args():
    parser = argparse.ArgumentParser(description='Distributed quantum circuit mapping parameter configuration')

    # Required arguments
    parser.add_argument('--core_count', '-core', type=int, required=True,
                        help='Number of QPUs (integer)')
    parser.add_argument('--core_capacity', '-cap', type=parse_int_list, required=True,
                        help='Capacity of each QPU (comma-separated integers, e.g., "4" or "4,6,8")')
    parser.add_argument('--circuit_name', '-cname', type=str, required=True,
                        help='Name of the quantum circuit (string)')
    parser.add_argument('--qubit_count', '-nq', type=int, required=True,
                        help='Number of qubits in the quantum circuit (integer)')

    args = parser.parse_args()

    # Unify the capacity list for each QPU
    if len(args.core_capacity) == 1:
        core_capacities = args.core_capacity * args.core_count
    else:
        core_capacities = args.core_capacity

    # Validate the array length
    if len(core_capacities) != 1 and len(core_capacities) != args.core_count:
        raise ValueError(
            f"The QPU capacity must be a single integer (for all QPUs) or {args.core_count} comma-separated values (specified individually for each QPU). "
            f"Current input: {args.core_capacity}"
        )
    args.core_capacities = core_capacities

    print("[INFO] Configuration parameters:")
    print(f"[INFO] Number of QPUs: {args.core_count}")
    print(f"[INFO] Capacity of each QPU: {args.core_capacities}")
    print(f"[INFO] Name of the quantum circuit: {args.circuit_name}")
    print(f"[INFO] Number of qubits in the quantum circuit: {args.qubit_count}")
    return args

def output_results(cname, circ, qpus, distributors):
    """
    将数据写入.csv文件
    """
    headers = ["Circuit", "#Qubits", "#Depths", "#Gates", "#Modules", "Metrics"]
    metrics = ["Comm Costs", "#RGate", "#RSWAP", "Exec Time"]
    for dis in distributors:
        headers.append(dis.name)
    data = {}
    for head in headers:
        data[head] = []
    gate_counts = circ.count_ops()
    total_gates = sum(gate_counts.values())

    for m in metrics:
        data["Circuit"].append(cname)
        data["#Qubits"].append(circ.num_qubits)
        data["#Depths"].append(circ.depth())
        data["#Gates"].append(total_gates)
        data["#Modules"].append(len(qpus))
        data["Metrics"].append(m)

    # 对每个distributor，写入四行
    for distributor in distributors:
        data[distributor.name].append(distributor.num_comms)
        data[distributor.name].append(distributor.num_gates)
        data[distributor.name].append(distributor.num_swaps)
        data[distributor.name].append(distributor.exec_time)

    print(data)

    filename = f"./outputs/data.csv"
    df = pd.DataFrame(data)
    df.to_csv(filename, mode="a", header=not pd.io.common.file_exists(filename), index=False)
    return

if __name__ == '__main__':
    get_args()
