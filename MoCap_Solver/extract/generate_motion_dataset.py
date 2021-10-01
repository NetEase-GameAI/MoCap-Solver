import numpy as np
import os
import glob
from tqdm import tqdm

def generate_motion_dataset():
    train_input_folder = os.path.join('data', 'training_windows_data')
    test_input_folder = os.path.join('data', 'testing_windows_data')

    train_output_folder = os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'motion_dataset')
    test_output_folder = os.path.join('MoCap_Solver', 'data', 'Testing_dataset', 'motion_dataset')

    if not os.path.exists(os.path.join('MoCap_Solver', 'data')):
        os.mkdir(os.path.join('MoCap_Solver', 'data'))

    if not os.path.exists(os.path.join('MoCap_Solver', 'data', 'Training_dataset')):
        os.mkdir(os.path.join('MoCap_Solver', 'data', 'Training_dataset'))

    if not os.path.exists(os.path.join('MoCap_Solver', 'data', 'Testing_dataset')):
        os.mkdir(os.path.join('MoCap_Solver', 'data', 'Testing_dataset'))

    if not os.path.exists(train_input_folder):
        os.mkdir(train_input_folder)

    if not os.path.exists(test_input_folder):
        os.mkdir(test_input_folder)

    if not os.path.exists(train_output_folder):
        os.mkdir(train_output_folder)

    if not os.path.exists(test_output_folder):
        os.mkdir(test_output_folder)

    train_files = glob.glob(os.path.join(train_input_folder, '*.npz'))
    test_files = glob.glob(os.path.join(test_input_folder, '*.npz'))

    for npz_file in tqdm(train_files):
        file_name = os.path.basename(npz_file).split('.')[0]
        h = np.load(npz_file)
        J = h['J']
        motion = h['motion']
        output_file = os.path.join(train_output_folder, file_name + '.npz')
        np.savez(output_file, J=J, motion=motion)

    for npz_file in tqdm(test_files):
        file_name = os.path.basename(npz_file).split('.')[0]
        h = np.load(npz_file)
        J = h['J']
        motion = h['motion']
        output_file = os.path.join(test_output_folder, file_name + '.npz')
        np.savez(output_file, J=J, motion=motion)