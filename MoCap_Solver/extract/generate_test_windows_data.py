import glob
import os
import numpy as np
from tqdm import tqdm
from utils.utils import JOINTNUM, MARKERNUM, topology, FRAMENUM, ref_idx
from utils.parse_data import get_transfer_matrix, rot_error, get_Quaternion_fromRotationMatrix
from utils.parse_data import get_marker_config1 as get_marker_config
from pyquaternion import Quaternion
def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result
def test_processing(input_folder, output_folder):
    npz_frames = glob.glob(os.path.join(input_folder, '*.npz'))
    weights = np.load(os.path.join('models', 'weights.npy'))
    windows_size = FRAMENUM
    scale = 1
    for npz_frame in tqdm(npz_frames[:]):
        file_name = os.path.basename(npz_frame).split('.')[0]
        npz_frame = os.path.join(input_folder, file_name + '.npz')
        h = np.load(npz_frame)
        M = h['M']
        M1=h['M1']
        J_t = h['J_t']
        J_R = h['J_R']
        J = h['J']
        M = M[::scale]
        J_t = J_t[::scale]
        J_R = J_R[::scale]
        N = M.shape[0]
        Marker = h['Marker']
        ################ Get motions ###########################################################
        motions = np.zeros((N, 4 * (JOINTNUM) + 3))
        prevlist = topology
        for i in range(N):
            for j in range(JOINTNUM):
                rot = np.linalg.inv(J_R[i, prevlist[j], :, :]).dot(J_R[i, j, :, :])
                q = Quaternion(matrix=rot, atol=1., rtol=1.)
                motions[i, (4 * j):(4 * j + 4)] = np.array([q[0], q[1], q[2], q[3]])
            motions[i, (4 * JOINTNUM):] = J_t[i, 0, :]

        N = M.shape[0]
        #######Find the Rigid Transform########
        windows_count = np.int(np.ceil(N / np.float(windows_size / 2)))
        if windows_count <= 1:
            continue
        step_size = np.int(windows_size / 2)
        newM = np.zeros((windows_count, windows_size, MARKERNUM, 3))
        newM1 = np.zeros((windows_count, windows_size, MARKERNUM, 3))
        new_J_t = np.zeros((windows_count, windows_size, JOINTNUM, 3))
        new_J_R = np.zeros((windows_count, windows_size, JOINTNUM, 3, 3))
        new_motion = np.zeros((windows_count, 4 * (JOINTNUM - 1) + 3, windows_size))
        new_first_rot = np.zeros((windows_count, windows_size, 4))

        RigidR = list()
        RigidT = list()
        t_pos_marker = np.load(os.path.join('models', 't_pos_marker.npy'))
        for idx_select in range(1):
            for i in range(windows_count):
                rot_error_list = []
                identity_rot = np.array([[1.,0.,0.], [0.,1.,0.],[0.,0.,1.]])
                for ind in range(windows_size):
                    M_i = M[min(i * step_size + ind, N-1), ref_idx, :]
                    R, T = get_transfer_matrix(M_i, t_pos_marker)
                    re = rot_error(identity_rot, R)
                    rot_error_list.append(re)
                rot_error_list = np.array(rot_error_list)
                min_index_list  = np.argsort(rot_error_list)
                min_index = min_index_list[idx_select]
                M_i = M[min(i * step_size + min_index, N-1), ref_idx, :]
                R, T = get_transfer_matrix(M_i, t_pos_marker)
                T = T.reshape(3, 1)
                RigidR.append(R)
                RigidT.append(T)
                for j in range(windows_size):
                    newM[i, j, :, :] = (R.dot(M[min(i * step_size + j, N - 1), :, :].T) + T).T
                    newM1[i, j, :, :] = (R.dot(M1[min(i * step_size + j, N - 1), :, :].T) + T).T
                    new_J_t[i, j, :, :] = (R.dot(J_t[min(i * step_size + j, N - 1), :, :].T) + T).T
                    for l in range(JOINTNUM):
                        new_J_R[i, j, l, :, :] = R.dot(J_R[min(i * step_size + j, N - 1), l, :, :])
                    new_motion[i, :, j] = motions[min(i * step_size + j, N - 1), 4:]
                    new_motion[i, -3:, j] = new_J_t[i, j, 0, :]
                    new_first_rot[i, j, :] = get_Quaternion_fromRotationMatrix(new_J_R[i, j, 0, :, :])
                q_orig = new_first_rot[i, :, :].copy()
                if (min_index < windows_size-1):
                    q_orig1 = q_orig[min_index:].copy()
                    q_orig1 = q_orig1.reshape(-1, 1, 4)
                    q_orig1 = qfix(q_orig1)
                    q_orig1 = q_orig1.reshape(-1, 4)
                    new_first_rot[i, min_index:] = q_orig1
                if (min_index > 0):
                    q_orig1 = q_orig[0:min_index+1].copy()
                    q_orig1 = q_orig1[::-1]
                    q_orig1 = q_orig1.reshape(-1, 1, 4)
                    q_orig1 = qfix(q_orig1)
                    q_orig1 = q_orig1[::-1]
                    q_orig1 = q_orig1.reshape(-1, 4)
                    new_first_rot[i, 0:min_index+1] = q_orig1
            #########compute the marker config and weighted marker config############
            mrk_config = get_marker_config(Marker, J, weights)
            mrk_config = np.tile(mrk_config.reshape(1, MARKERNUM, JOINTNUM, 3), (windows_count, 1, 1, 1))
            offsets = J - J[prevlist]
            new_offsets = np.tile(offsets.reshape(1, JOINTNUM, 3), (windows_count, 1, 1))
            outfile = os.path.join(output_folder, os.path.basename(file_name + '_' + str(idx_select) + '.npz'))
            np.savez(outfile, offsets=new_offsets, M=newM, M1=newM1, J_t=new_J_t, J_R=new_J_R, RigidR=RigidR, RigidT=RigidT,
                     shape=h['shape'], Marker=Marker, J=J, mrk_config=mrk_config, motion=new_motion, first_rot=new_first_rot)

def generate_test_windows_data():
    train_input_folder = os.path.join('data', 'test_sample_data')
    train_output_folder = os.path.join('data', 'testing_windows_data')
    if not os.path.exists(train_input_folder):
        os.mkdir(train_input_folder)
    if not os.path.exists(train_output_folder):
        os.mkdir(train_output_folder)
    test_processing(train_input_folder, train_output_folder)