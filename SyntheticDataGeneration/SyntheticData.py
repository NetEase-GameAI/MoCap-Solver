import sys
import os
sys.path.append(os.path.join('external'))
class SyntheticData():
    def __init__(self):
        return

    def synthetic_data(self, SEED):
        from SyntheticDataGeneration.extract_shape_data import extract_shape_data
        extract_shape_data()
        from SyntheticDataGeneration.generate_train_data import generate_train_data
        generate_train_data(SEED)
        from SyntheticDataGeneration.generate_test_data import generate_test_data
        generate_test_data(SEED)
        return