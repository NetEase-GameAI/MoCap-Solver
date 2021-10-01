import glob
import os
import numpy as np
from tqdm import tqdm
from MoCap_Solver.io.parse_data import corrupt
from utils.utils import MARKERNUM, ref_idx, FRAMENUM

def processing_noise_add(input_folder, input_windows_folder,  output_windows_folder, output_folder):
    npz_frames = glob.glob(os.path.join(input_windows_folder, '*.npz'))
    windows_size = FRAMENUM
    for npz_frame in tqdm(npz_frames):
        filename1 = os.path.basename(npz_frame).split('.')[0]
        filename = filename1[:-2]
        if not os.path.exists(os.path.join(input_folder, filename + '.npz')):
            continue
        h = np.load(os.path.join(input_folder, filename + '.npz'))
        # filename = os.path.basename(npz_frame).split('.')[0]
        M = h['M']
        N = M.shape[0]
        M1 = np.zeros((N, MARKERNUM, 3))

        h1 = np.load(os.path.join(input_windows_folder, filename1 + '.npz'))
        RigidR = h1['RigidR']
        RigidT = h1['RigidT']
        for i in range(N):
            M1[i, :, :] = corrupt(M[i, :, :], 0.1, 0.1, 0.3)
            M1[i, ref_idx, :] = M[i, ref_idx, :]
        windows_count = np.int(np.ceil(N / np.float(windows_size / 2)))
        M1_windows = np.zeros((windows_count, windows_size, MARKERNUM, 3))
        step_size = np.int(windows_size / 2)
        if windows_count <= 1:
            continue
        for i in range(windows_count):
            R = RigidR[i]
            T = RigidT[i]
            for j in range(windows_size):
                M1_windows[i, j, :, :] = (R.dot(M1[min(i * step_size + j, N - 1), :, :].T) + T).T

        outfile = os.path.join(output_folder, filename1 +'.npy')
        np.save(outfile, M1)
        outfile1 = os.path.join(output_windows_folder, filename1 + '.npy')
        np.save(outfile1, M1_windows)

def add_noise():
    ############### Add noise in the training dataset ##########################################################
    train_input_folder = os.path.join('data', 'train_sample_data')
    train_input_windows_folder = os.path.join('data', 'training_windows_data')
    train_output_folder = os.path.join('MoCap_Solver', 'data', 'training_noise_data')
    train_output_windows_folder = os.path.join('MoCap_Solver', 'data', 'training_noise_windows_data')
    if not os.path.exists(train_output_folder):
        os.mkdir(train_output_folder)
    if not os.path.exists(train_output_windows_folder):
        os.mkdir(train_output_windows_folder)
    processing_noise_add(train_input_folder, train_input_windows_folder, train_output_windows_folder, train_output_folder)

