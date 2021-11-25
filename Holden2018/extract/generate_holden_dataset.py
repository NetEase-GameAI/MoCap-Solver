import numpy as np
import os
import glob
from utils.utils import ref_idx, MARKERNUM, JOINTNUM
from utils.parse_data import get_transfer_matrix
from tqdm import tqdm
from utils.parse_data import get_marker_config1 as get_marker_config

def simulate_outlier_remove(M, M1):
    remove_threshold = 0.2
    detM = np.sqrt( np.sum( np.square(M1-M), axis = -1) )
    M1[detM > remove_threshold] = [0.0,0.0,0.0]
    return M1

def generate_holden_dataset():
    train_input_folder = os.path.join('data', 'train_sample_data')
    test_input_folder = os.path.join('data', 'test_sample_data')

    train_output_folder = os.path.join('Holden2018', 'data', 'Training_dataset')
    test_output_folder = os.path.join('Holden2018', 'data', 'Testing_dataset')

    if not os.path.exists(os.path.join('Holden2018', 'data')):
        os.mkdir(os.path.join('Holden2018', 'data'))

    if not os.path.exists(train_output_folder):
        os.mkdir(train_output_folder)

    if not os.path.exists(test_output_folder):
        os.mkdir(test_output_folder)

    train_files = glob.glob(os.path.join(train_input_folder, '*.npz'))
    test_files = glob.glob(os.path.join(test_input_folder, '*.npz'))
    weights = np.load(os.path.join('models', 'weights.npy')).reshape(MARKERNUM, JOINTNUM, 1)
    t_pos_marker = np.load(os.path.join('models', 't_pos_marker.npy'))
    for npz_file in tqdm(train_files):
        file_name = os.path.basename(npz_file).split('.')[0]
        h = np.load(npz_file)
        M1 = h['M1']
        N = M1.shape[0]
        M = h['M']
        J = h['J']
        J_R = h['J_R']
        J_t = h['J_t']
        M1 = simulate_outlier_remove(M, M1)
        for i in range(N):
            M_i = M[i, ref_idx, :]
            R, T = get_transfer_matrix(M_i, t_pos_marker)
            M[i, :, :] = (R.dot(M[i, :, :].T) + T).T
            J_t[i, :, :] = (R.dot(J_t[i, :, :].T) + T).T
            for j in range(JOINTNUM):
                J_R[i, j, :, :] = R.dot(J_R[i, j, :, :])
        J_t = J_t.reshape(N, JOINTNUM, 3, 1)
        J_all = np.concatenate([J_R, J_t], axis=3)
        Marker = h['Marker']
        mrk_config = get_marker_config(Marker, J, weights.reshape(MARKERNUM, JOINTNUM))
        weighted_mrk_config = np.sum(weights * mrk_config, axis=1)
        weighted_mrk_config = weighted_mrk_config.reshape(1, MARKERNUM, 3)
        weighted_mrk_config = np.tile(weighted_mrk_config, [N, 1, 1])


        output_file = os.path.join(train_output_folder, file_name + '.npz')
        np.savez(output_file, M1=M1, M=M, mrk_config = mrk_config, weighted_mrk_config=weighted_mrk_config, J_all=J_all)

    for npz_file in tqdm(test_files):
        file_name = os.path.basename(npz_file).split('.')[0]
        h = np.load(npz_file)
        M1 = h['M1']
        N = M1.shape[0]
        M = h['M']
        J = h['J']
        J_R = h['J_R']
        J_t = h['J_t']
        M1 = simulate_outlier_remove(M, M1)
        for i in range(N):
            M_i = M[i, ref_idx, :]
            R, T = get_transfer_matrix(M_i, t_pos_marker)
            M[i, :, :] = (R.dot(M[i, :, :].T) + T).T
            J_t[i, :, :] = (R.dot(J_t[i, :, :].T) + T).T
            for j in range(JOINTNUM):
                J_R[i, j, :, :] = R.dot(J_R[i, j, :, :])
        J_t = J_t.reshape(N, JOINTNUM, 3, 1)
        J_all = np.concatenate([J_R, J_t], axis=3)
        Marker = h['Marker']
        mrk_config = get_marker_config(Marker, J, weights.reshape(MARKERNUM, JOINTNUM))
        weighted_mrk_config = np.sum(weights * mrk_config, axis=1)
        weighted_mrk_config = weighted_mrk_config.reshape(1, MARKERNUM, 3)
        weighted_mrk_config = np.tile(weighted_mrk_config, [N, 1, 1])

        output_file = os.path.join(test_output_folder, file_name + '.npz')
        np.savez(output_file, M1=M1, M=M, mrk_config = mrk_config, weighted_mrk_config=weighted_mrk_config, J_all=J_all)