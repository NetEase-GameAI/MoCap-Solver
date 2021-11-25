import sys
import os
sys.path.append(os.path.join('external'))

SEED = 100

print('############## generate test synthetic data ##################################')
from SyntheticDataGeneration.generate_test_data import generate_test_data
generate_test_data(SEED)

print('############## generate test windows data ##################################')
from MoCap_Solver.extract.generate_test_windows_data import generate_test_windows_data
generate_test_windows_data()

print('############## generate mocap solver dataset ##################################')
from MoCap_Solver.extract.generate_moc_sol_dataset import generate_moc_sol_dataset
generate_moc_sol_dataset()

print('############## evaluate mocap solver ##################################')
from MoCap_Solver.evaluate.evaluate_sequence import evaluate_sequence
evaluate_sequence()
