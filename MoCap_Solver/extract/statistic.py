import numpy as np


def get_batch_mu(batch_data):
    return np.mean(batch_data, axis=0)


def get_batch_sigma(batch_data):
    return np.std(batch_data, ddof=1,
                  axis=0)  # calculate the sample std instead of the global std,therefore ddof should be set 1
import os
import glob
from tqdm import tqdm
from utils.utils import topology, FRAMENUM, JOINTNUM, MARKERNUM, get_RotationMatrix_fromQuaternion

def statistic():
    npzs = glob.glob(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', '*.npz'))
    train_list = [npz for npz in npzs[:]]
    train_list = np.array(train_list)
    np.random.seed(10000)
    np.random.shuffle(train_list)
    train_list = train_list[:]
    totalframe = 0
    npzfiles = train_list
    for npzfile in tqdm(npzfiles):
        h = np.load(npzfile)
        M = h['M']
        totalframe += M.shape[0]
    clean_markers = np.zeros((totalframe, FRAMENUM, MARKERNUM, 3))
    currentframe = 0
    currentidx = 0
    for npzfile in tqdm(npzfiles):
        filename = os.path.basename(npzfile).split('.')[0]
        h1 = np.load(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', filename + '.npz'))
        M = h1['M']
        N = M.shape[0]
        clean_markers[currentframe: (currentframe + N)] = M
        currentidx += 1
        currentframe += N
    train_marker_data_mean = get_batch_mu(clean_markers.reshape(-1, FRAMENUM, MARKERNUM, 3))
    train_marker_data_var = get_batch_sigma(clean_markers.reshape(-1, FRAMENUM, MARKERNUM, 3))
    train_marker_data_var[train_marker_data_var < 1e-4] = 1.
    train_marker_data = np.array([train_marker_data_mean, train_marker_data_var])
    np.save(os.path.join('models', 'train_marker_data.npy'), train_marker_data)
    del clean_markers, train_marker_data


    motion_latent = np.zeros((totalframe, 1792))
    currentframe = 0
    currentidx = 0
    for npzfile in tqdm(npzfiles):
        filename = os.path.basename(npzfile).split('.')[0]
        h1 = np.load(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', filename + '.npz'))
        motion_latents = h1['motion_latent']
        N = motion_latents.shape[0]
        motion_latent[currentframe: (currentframe + N)] = motion_latents
        currentidx += 1
        currentframe += N
    train_motion_code_data_mean = get_batch_mu(motion_latent)
    train_motion_code_data_var = get_batch_sigma(motion_latent)
    train_motion_code_data_var[train_motion_code_data_var < 1e-4] = 1.
    train_motion_code_data = np.array([train_motion_code_data_mean, train_motion_code_data_var])
    np.save(os.path.join('models', 'train_motion_code_data.npy'), train_motion_code_data)
    del train_motion_code_data, motion_latent


    offsets_latent = np.zeros((totalframe, 168))
    currentframe = 0
    currentidx = 0
    for npzfile in tqdm(npzfiles):
        filename = os.path.basename(npzfile).split('.')[0]
        h1 = np.load(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', filename + '.npz'))
        offsets_latents = h1['offsets_latent']
        N = offsets_latents.shape[0]
        offsets_latent[currentframe: (currentframe + N)] = offsets_latents
        currentidx += 1
        currentframe += N
    train_offset_code_data_mean = get_batch_mu(offsets_latent)
    train_offset_code_data_var = get_batch_sigma(offsets_latent)
    train_offset_code_data_var[train_offset_code_data_var < 1e-4] = 1.
    train_offset_code_data = np.array([train_offset_code_data_mean, train_offset_code_data_var])
    np.save(os.path.join('models', 'train_offset_code_data.npy'), train_offset_code_data)
    del train_offset_code_data, offsets_latent

    mc_latent_code = np.zeros((totalframe, 1024))
    currentframe = 0
    currentidx = 0
    for npzfile in tqdm(npzfiles):
        filename = os.path.basename(npzfile).split('.')[0]
        h1 = np.load(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', filename + '.npz'))
        mc_latent_codes = h1['mc_latent_code']
        N = mc_latent_codes.shape[0]
        mc_latent_code[currentframe: (currentframe + N)] = mc_latent_codes
        currentidx += 1
        currentframe += N
    train_mc_code_data_mean = get_batch_mu(mc_latent_code)
    train_mc_code_data_var = get_batch_sigma(mc_latent_code)
    train_mc_code_data_var[train_mc_code_data_var < 1e-4] = 1.
    train_mc_code_data = np.array([train_mc_code_data_mean, train_mc_code_data_var])
    np.save(os.path.join('models', 'train_mc_code_data.npy'), train_mc_code_data)
    del mc_latent_code, train_mc_code_data


    first_rot = np.zeros((totalframe, FRAMENUM, 4))
    currentframe = 0
    currentidx = 0
    for npzfile in tqdm(npzfiles):
        filename = os.path.basename(npzfile).split('.')[0]
        h1 = np.load(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', filename + '.npz'))
        first_rots = h1['first_rot']
        N = first_rots.shape[0]
        first_rot[currentframe: (currentframe + N)] = first_rots
        currentidx += 1
        currentframe += N
    train_first_rot_data_mean = get_batch_mu(first_rot)
    train_first_rot_data_var = get_batch_sigma(first_rot)
    train_first_rot_data_var[train_first_rot_data_var < 1e-4] = 1.
    train_first_rot_data = np.array([train_first_rot_data_mean, train_first_rot_data_var])
    np.save(os.path.join('models', 'train_first_rot_data.npy'), train_first_rot_data)
    del first_rot, train_first_rot_data

