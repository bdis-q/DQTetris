# DQTetris
A quantum circuit mapping technique designed for distributed architectures to reduce inter-module communications.

## Requirements

- Qiskit 1.4.1
- Pytket-DQC: https://github.com/CQCL/pytket-dqc

## Execution

```shell
./run.sh
```

## Directory Description

```
DQTetris
├── benchmarks: the .qasm files of the tested quantum circuits
├── figs
│   ├── data.xlsx: the experimental results
│   └── fig.ipynb: the visualization of the experimental results
├── main.py: the main file
├── methods: the implementation of different mapping techniques
├── outputs: the directory for saving the program outputs
├── run.sh: the execution script
└── utils: utility functions for input, output, and loading quantum circuits
```