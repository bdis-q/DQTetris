# Evaluation on fixed module capacity with 10 qubits
# S-OEE FGP-rOEE P-DQC (CE) P-DQC (PA) WBCP DQTetris (PA)
python main.py -core 2 -cap 10 -cname QV -nq 20
python main.py -core 3 -cap 10 -cname QV -nq 30
python main.py -core 4 -cap 10 -cname QV -nq 40
python main.py -core 5 -cap 10 -cname QV -nq 50
python main.py -core 6 -cap 10 -cname QV -nq 60

# S-OEE FGP-rOEE P-DQC (PA) WBCP DQTetris (PA)
python main.py -core 7 -cap 10 -cname QV -nq 70
python main.py -core 8 -cap 10 -cname QV -nq 80
python main.py -core 9 -cap 10 -cname QV -nq 90
python main.py -core 10 -cap 10 -cname QV -nq 100
python main.py -core 11 -cap 10 -cname QV -nq 110
python main.py -core 12 -cap 10 -cname QV -nq 120
python main.py -core 13 -cap 10 -cname QV -nq 130

# S-OEE P-DQC (PA) WBCP DQTetris (PA)
python main.py -core 14 -cap 10 -cname QV -nq 140
python main.py -core 15 -cap 10 -cname QV -nq 150
python main.py -core 16 -cap 10 -cname QV -nq 160

# Evaluation on fixed module capacity with 50 qubits
# S-OEE P-DQC (PA) WBCP DQTetris (PA)
python main.py -core 2 -cap 50 -cname QV -nq 100
python main.py -core 3 -cap 50 -cname QV -nq 150
python main.py -core 4 -cap 50 -cname QV -nq 200
python main.py -core 5 -cap 50 -cname QV -nq 250
python main.py -core 6 -cap 50 -cname QV -nq 300
python main.py -core 7 -cap 50 -cname QV -nq 350

# Evaluation on a 200-qubit circuit, varing module capacities
# S-OEE, P-DQC (PA), WBCP, DQTetris (PA)
python main.py -core 2 -cap 100 -cname QV -nq 200
python main.py -core 3 -cap 68 -cname QV -nq 200
python main.py -core 4 -cap 50 -cname QV -nq 200
python main.py -core 5 -cap 40 -cname QV -nq 200
python main.py -core 6 -cap 34 -cname QV -nq 200
python main.py -core 7 -cap 30 -cname QV -nq 200
python main.py -core 8 -cap 26 -cname QV -nq 200
python main.py -core 9 -cap 24 -cname QV -nq 200
python main.py -core 10 -cap 20 -cname QV -nq 200

# Evaluation on different benchmarks
# S-OEE FGP-rOEE P-DQC (PA) WBCP DQTetris (PA)
python main.py -core 4 -cap 20 -cname random_n70 -nq 70
python main.py -core 7 -cap 20 -cname random_n130 -nq 130
python main.py -core 7 -cap 20 -cname qnn_n130 -nq 130
python main.py -core 4 -cap 20 -cname multiplier_n75 -nq 75
# S-OEE FGP-rOEE P-DQC (CE) WBCP DQTetris (CE)
python main.py -core 22 -cap 20 -cname ising_n420 -nq 420
python main.py -core 5 -cap 10 -cname DraperQFTAdder -nq 50
