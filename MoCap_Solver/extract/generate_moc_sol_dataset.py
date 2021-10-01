import glob
import os
import numpy as np
from tqdm import tqdm
import torch
from MoCap_Solver.model.skeleton import build_edge_topology
from MoCap_Solver.model.enc_and_dec import AE
from MoCap_Solver.model.MC_Encoder import MC_AE, TS_AE
from utils.utils import JOINTNUM, topology

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

def mocap_processing(input_folder, output_folder):
    npz_frames = glob.glob(os.path.join(input_folder, '*.npz'))
    device = torch.device('cuda:0')
    joint_topology = topology
    edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
    motion_encoder = AE(edges).to(device)
    static_encoder = TS_AE(edges).to(device)
    mc_encoder = MC_AE(edges).to(device)
    motion_encoder.load_state_dict(
        torch.load(os.path.join('models', 'motion_encoder.pt'),
                   map_location=device))
    mc_encoder.load_state_dict(
        torch.load(os.path.join('models', 'mc_encoder.pt'),
                   map_location=device))
    static_encoder.load_state_dict(
        torch.load(os.path.join('models', 'ts_encoder.pt'),
                   map_location=device))

    motion_encoder.eval()
    mc_encoder.eval()
    static_encoder.eval()

    train_offset_data = np.load(os.path.join('models', 'train_data_ts.npy'))
    train_offset_data = torch.tensor(train_offset_data, dtype=torch.float32).to(device)

    train_motion_data = np.load(os.path.join('models', 'train_motion_data.npy'))
    train_motion_data = torch.tensor(train_motion_data, dtype=torch.float32).to(device)
    train_mc_data = np.load(os.path.join('models', 'train_data_marker_config.npy'))
    train_mc_data = torch.tensor(train_mc_data, dtype=torch.float32).to(device)

    for npz_frame in tqdm(npz_frames):
        h = np.load(npz_frame)
        M = h['M']
        J_t = h['J_t']
        J_R = h['J_R']
        J = h['J']
        offsets = h['offsets']
        RigidR = h['RigidR']
        RigidT = h['RigidT']
        Marker = h['Marker']
        mrk_config = h['mrk_config']
        weighted_mrk_config = h['mrk_config']
        motion = h['motion']
        first_rot = h['first_rot']
        M1 = h['M1']

        ####### Find the Rigid Transform ########
        windows_count = M.shape[0]
        offsets_input = torch.tensor(offsets, dtype=torch.float32).to(device)
        offsets_input = (offsets_input - train_offset_data[0]) / train_offset_data[1]
        offset_latent = []
        res_offset_list = []
        with torch.no_grad():
            offset_latent = static_encoder.encode(offsets_input)
            if(len(offset_latent.shape)==1):
                offset_latent = offset_latent.view(1, -1)
            res_offset_list = static_encoder.decode1(offset_latent)
        res_offset_list[0] = res_offset_list[0].view(-1, JOINTNUM, 3)
        res_offset_list[0] = res_offset_list[0] * train_offset_data[1] + train_offset_data[0]
        res_offset_input = (res_offset_list[0] - train_offset_data[0])/train_offset_data[1]
        new_first_rot = first_rot
        offsets_latent = offset_latent.cpu().numpy()
        offset_latent1 = torch.tensor(offset_latent, dtype=torch.float32).to(device)
        motion_input = torch.tensor(motion[:, :, :], dtype=torch.float32).to(device)
        motion_input = (motion_input - train_motion_data[0]) / train_motion_data[1]
        motion_latent = []
        with torch.no_grad():
            motion_latent, _ = motion_encoder.forward(motion_input, res_offset_list)
        motion_latent = motion_latent.view(windows_count, -1)
        motion_latent = motion_latent.cpu().numpy()
        ######### compute the marker config and weighted marker config ############
        mrk_config = mrk_config
        mrk_config_input = torch.tensor(mrk_config,dtype=torch.float32).to(device)
        mrk_config_input = (mrk_config_input - train_mc_data[0])/train_mc_data[1]
        with torch.no_grad():
            mc_latent_code = mc_encoder.encode(mrk_config_input, res_offset_input, offset_latent1)
        mc_latent_code = mc_latent_code.cpu().numpy()
        outfile = os.path.join(output_folder, os.path.basename(npz_frame))
        np.savez(outfile, offsets=offsets, M=M,M1=M1, J_t=J_t, J_R=J_R, RigidR=RigidR, RigidT=RigidT,
                 shape=h['shape'], Marker=Marker, J=J, mrk_config=mrk_config, weighted_mrk_config=weighted_mrk_config, mc_latent_code=mc_latent_code, motion_latent=motion_latent,
                 motion=motion, first_rot=new_first_rot, offsets_latent=offsets_latent)

def generate_moc_sol_dataset():
    mean_motion = torch.load(os.path.join('models', 'motion_data_mean_group.pt')).cpu().numpy()[0]
    var_motion = torch.load(os.path.join('models', 'motion_data_var_group.pt')).cpu().numpy()[0]
    np.save(os.path.join('models', 'train_motion_data'), np.array([mean_motion, var_motion]))
    train_input_folder = os.path.join('data', 'training_windows_data')
    train_output_folder = os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset')
    if not os.path.exists(train_output_folder):
        os.mkdir(train_output_folder)

    test_input_folder = os.path.join('data', 'testing_windows_data')
    test_output_folder = os.path.join('MoCap_Solver', 'data', 'Testing_dataset', 'mocap_dataset')
    if not os.path.exists(test_output_folder):
        os.mkdir(test_output_folder)

    mocap_processing(train_input_folder, train_output_folder)
    mocap_processing(test_input_folder, test_output_folder)
