import torch
import os
from SyntheticDataGeneration.smpl_layer_new import SMPL_layer_new as SMPL_Layer
from tqdm import tqdm
import numpy as np
import json

def skinning_from_mrk_config( mrk_config, joints_pos, joints_transform, weights):
    '''

    :param mrk_config: marker configurarion: (56, 24, 3)
    :param joints_pos: The position of joints: (24, 3)
    :param joints_transform: The rotation matrix of joints: (24, 3, 3)
    :param weights: The skinning weights: (56, 24)
    :return:
        mrk_pos: The position of markers:(56, 3)
    '''
    _offset_global = list()
    mrk_num = mrk_config.shape[0]
    joint_num = mrk_config.shape[1]
    for mrk_index in range(mrk_num):
        offset = mrk_config[mrk_index]
        _offset_ = [joints_transform[i].dot(offset[i]) + joints_pos[i] for i in range(joint_num)]
        _offset_ = np.array(_offset_)
        _offset_global.append(_offset_)
    _offset_global = np.array(_offset_global)
    mrk_pos = (_offset_global * np.tile(weights.reshape(-1, 24, 1), (1, 1, 3))).sum(axis=1)
    return mrk_pos

def get_marker_config(markers_pos, joints_pos, weights):
    '''

    :param markers_pos: The position of markers: (56, 3)
    :param joints_pos: The position of joints: (24, 3)
    :param joints_transform: The roration matrix of joints: (24, 3, 3)
    :param weights: The skinning weights: (56, 24)
    :return:
        marker_configuration: (56, 24, 3)
    '''
    _offset_list = list()
    mrk_pos_matrix = np.array(markers_pos)
    jts_pos_matrix = np.array(joints_pos)
    weights_matrix = np.array(weights)
    tile_num = joints_pos.shape[0]
    for mrk_index in range(mrk_pos_matrix.shape[0]):
        mark_pos = mrk_pos_matrix[mrk_index]
        jts_offset = mark_pos - jts_pos_matrix
        jts_offset_local = [
            np.int64(weights_matrix[mrk_index, i] > 1e-5) * (jts_offset[i]).reshape(3) for i in
            range(tile_num)]
        jts_offset_local = np.array(jts_offset_local)
        _offset_list.append(jts_offset_local)
    return np.array(_offset_list)

def corrupt( x, sigma_occlude, sigma_shift, beta):
    '''
    corrupt the input markers
    x: input markers
    sigma_occlude:value adjusts the probability of a marker being occluded
    sigma_shift:value adjusts the probability of a marker being shifted
    beta:controls the scale of the random translations applied to shifted markers.
    '''
    _u_Normal = 0
    _sigma_occlude_Normal = sigma_occlude
    _sigma_shift_Normal = sigma_shift
    _beta = beta
    _size_Normal = 1
    alpha_occ = np.random.normal(loc=_u_Normal, scale=_sigma_occlude_Normal, size=1)
    alpha_shift = np.random.normal(loc=_u_Normal, scale=_sigma_shift_Normal, size=1)
    _min_alpha_occ = min(abs(alpha_occ), 2 * _sigma_occlude_Normal)
    _min_alpha_shift = min(abs(alpha_shift), 2 * _sigma_shift_Normal)
    _size_Bernoulli = x.shape[0]
    X_occu = np.random.binomial(n=1, p=_min_alpha_occ, size=_size_Bernoulli)[:, np.newaxis]  # from (len,) to (len,1)
    X_shift = np.random.binomial(n=1, p=_min_alpha_shift, size=_size_Bernoulli)[:, np.newaxis]
    X_v = np.random.uniform(-_beta, _beta, size=(_size_Bernoulli, 3))
    X_corrupted = (x + X_shift * X_v) * (1 - X_occu)  # element-wise product
    assert np.shape(X_corrupted) == np.shape(x)
    return X_corrupted


def generate_train_data(SEED):
    marker_idx_set = [[414, 1219, 3495, 2837, 3207, 447, 709, 2911, 1910, 104, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1112, 3257, 1238, 1442, 1686, 6604, 3941, 4195, 5244, 5090, 3517, 4778, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 5322, 4915, 5157, 1330, 751], [414, 1301, 3497, 2837, 3207, 447, 2935, 2911, 1910, 104, 650, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1112, 3257, 1861, 1442, 1686, 6604, 3941, 6396, 5244, 5090, 3517, 4139, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 4124, 4915, 5157, 1330, 751], [414, 453, 3073, 2837, 3207, 447, 1812, 2911, 1910, 104, 1535, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1112, 3257, 783, 1442, 1686, 6604, 3941, 5273, 5244, 5090, 3517, 4077, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 4721, 4915, 5157, 1330, 751], [414, 1219, 3495, 2837, 3207, 517, 709, 2911, 1910, 2786, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1135, 3257, 1238, 1442, 1980, 6604, 3973, 4195, 5244, 5090, 3635, 4778, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4621, 6657, 5322, 4915, 5441, 1330, 751], [414, 1219, 3495, 2837, 3207, 450, 709, 2911, 1910, 5, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1115, 3257, 1238, 1442, 1685, 6604, 3939, 4195, 5244, 5090, 3515, 4778, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4599, 6657, 5322, 4915, 5154, 1330, 751], [414, 1301, 3497, 2837, 3207, 447, 2935, 2911, 1910, 104, 650, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1135, 3257, 1861, 1442, 1980, 6604, 3941, 6396, 5244, 5090, 3517, 4139, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4621, 6657, 4124, 4915, 5441, 1330, 751], [414, 453, 3073, 2837, 3207, 447, 1812, 2911, 1910, 104, 1535, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1135, 3257, 783, 1442, 1980, 6604, 3941, 5273, 5244, 5090, 3517, 4077, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4621, 6657, 4721, 4915, 5441, 1330, 751], [414, 1219, 3495, 2837, 3207, 447, 709, 2911, 1910, 104, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1135, 3257, 1238, 1442, 1980, 6604, 3941, 4195, 5244, 5090, 3517, 4778, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4621, 6657, 5322, 4915, 5441, 1330, 751], [414, 1219, 3495, 2837, 3207, 447, 709, 2911, 1910, 104, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1115, 3257, 1238, 1442, 1685, 6604, 3941, 4195, 5244, 5090, 3517, 4778, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4599, 6657, 5322, 4915, 5154, 1330, 751], [414, 1219, 3495, 2837, 3207, 517, 709, 2911, 1910, 2786, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3311, 846, 2173, 2032, 1112, 3257, 1238, 1442, 1686, 6604, 3973, 4195, 5244, 5090, 3635, 4778, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 5322, 4915, 5157, 1330, 751]]
    np.random.seed(SEED)
    h = np.load(os.path.join('data','file_list.npz'))
    file_list = h['file_list']
    file_folders = h['file_folders']
    motion_num = file_list.shape[0]
    motion_set = np.random.permutation(motion_num)
    train_motion_set = motion_set[:1860]

    marker_idx4 = [414, 1219, 3495, 2837, 3207, 447, 709, 2911, 1910, 104, 1294, 2920, 3331, 1023, 1718, 1995, 2110, \
                  1068, 1043, 3232, 3311, 846, 2173, 2032, 1112, 3257, 1238, 1442, 1686, 6604, 3941, 4195, 5244,
                  5090, 3517,
                  4778, 6380, \
                  6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6682, 4332, 5751, 5609, 4598, 6657, 5322, 4915,
                  5157,
                  1330, 751]
    marker_idx1 = [414, 1219, 3495, 2837, 3207, 447, 709, 2911, 1910, 104, 1294, 2920, 3331, 1023, 1718, 1995, 2110, 1068, 1043, 3232, 3285, 846, 2173, 2032, 1112, 3213, 1238, 1442, 1686, 6604, 6882, 4195, 5244, 5090, 3517, 4778, 6380, 6730, 4520, 4861, 5594, 5480, 4555, 4529, 6634, 6683, 4332, 5655, 5609, 4598, 6612, 5322, 4915, 5157, 1330, 751]
    # marker_idx_set = [[], []]
    marker_idx_set_idx = [0, 1, 2, 3,4 ,5,6,7,8,9]
    person_idx_set_idx = np.random.permutation(1700)
    train_person_set_idx = person_idx_set_idx[:1530]
    trainfile = os.path.join('external', 'CMU')
    validfile = os.path.join('external', 'CMU')
    out_train_file = os.path.join('data', 'train_sample_data')
    out_valid_file = os.path.join('data', 'test_sample_data')
    shape_file_dir = os.path.join('external', 'male_shape_data')
    if not os.path.exists(out_train_file):
        os.mkdir(out_train_file)
    if not os.path.exists(out_valid_file):
        os.mkdir(out_valid_file)
    Marker_idx = [i for i in range(56)]
    ref_idx = [54, 3, 46, 32, 36, 21, 7, 11]
    non_ref_idx = list(set(Marker_idx).difference(set(ref_idx)))
    cuda = True
    batch_size = 100

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='male',
        model_root=os.path.join('external', 'smplpytorch', 'native', 'models'))

    for npz_frame_idx in tqdm(train_motion_set[:20]):
        try:
            folder_name = file_folders[npz_frame_idx]
            filename = file_list[npz_frame_idx]
            npz_frame = os.path.join(trainfile, folder_name, filename+'.npz')
            cdata = np.load(npz_frame)
        except:
            print('Could not read %s ! skipping..' % npz_frame)
            continue
        for mc_select_idx in range(3):
            rand_number = np.random.choice(marker_idx_set_idx)
            marker_idx = marker_idx_set[rand_number]
            person_rand_number = np.random.choice(train_person_set_idx)
            N = len(cdata['poses'])
            poses = cdata['poses'][:, :]
            trans = cdata['trans']
            pose_params = np.zeros((N, 72))
            # pose_params[:, 69:] = trans
            pose_params[:, :66] = poses[:, :66]
            pose_params[:, 66:69] = poses[:, 75:78]
            pose_params[:, 69:72] = poses[:, 120:123]
            pose_params = pose_params.astype(np.float32)
            trans = trans.astype(np.float32)
            M = np.zeros((N, 56, 3), dtype=np.float32)
            M1 = np.zeros((N, 56, 3), dtype=np.float32)
            J_R = np.zeros((N, 24, 3, 3), dtype=np.float32)
            J_t = np.zeros((N, 24, 3), dtype=np.float32)
            shape_file = os.path.join(shape_file_dir, '%04d.json' % (person_rand_number))
            load_dict = json.load(open(shape_file, 'r'))
            shape = np.array(load_dict['betas']).reshape(1, 10).astype(np.float32)
            n_batch = np.int(np.ceil(N / 400))
            for i in range(n_batch):

                pose_param1 = pose_params[400 * i: min(400 * (i + 1), N)]
                pose_param1 = torch.tensor(pose_param1)
                shape_params = np.tile(shape, (pose_param1.shape[0], 1))
                shape_params = torch.tensor(shape_params)
                trans_param1 = trans[400 * i: min(400 * (i + 1), N)]
                trans_param1 = torch.tensor(trans_param1)
                # GPU mode
                if cuda:
                    pose_param1 = pose_param1.cuda()
                    shape_params = shape_params.cuda()
                    trans_param1 = trans_param1.cuda()
                    smpl_layer.cuda()

                # Forward from the SMPL layer
                _, Jtr, JtrR = smpl_layer(pose_param1, th_betas=shape_params, th_trans=trans_param1)
                # M[400 * i: min(400 * (i + 1), N), :, :] = verts.cpu().detach()[:, marker_idx, :]
                J_t[400 * i: min(400 * (i + 1), N), :, :] = Jtr.cpu().detach()
                J_R[400 * i: min(400 * (i + 1), N), :, :, :] = np.transpose(JtrR.cpu().detach()[:, :3, :3, :],
                                                                            (0, 3, 1, 2))
            filename = os.path.basename(npz_frame).split('.')[0]

            output_file = os.path.join(out_train_file, folder_name + '_' + filename + '_' + str(mc_select_idx) + '.npz')
            # output_file = os.path.join(out_train_file, filename  + '.npz')
            pose_param1 = np.zeros((1, 72), dtype=np.float32)
            shape_params = shape.astype(np.float32)
            pose_param1 = torch.tensor(pose_param1)
            shape_params = torch.tensor(shape_params)
            if cuda:
                pose_param1 = pose_param1.cuda()
                shape_params = shape_params.cuda()
                smpl_layer.cuda()
            verts, Jtr, JtrR = smpl_layer(pose_param1, th_betas=shape_params)
            Joint = np.array(Jtr.cpu().detach())[0, :, :]
            t_pose_marker = np.array(verts.cpu().detach())[0, marker_idx, :]
            weights = smpl_layer.th_weights[marker_idx, :].cpu().numpy()
            weights1 = smpl_layer.th_weights[marker_idx4, :].cpu().numpy()
            np.save('weights.npy', weights1)
            ############ skinning ###########################################
            # 1. compute marker configuration
            marker_config = get_marker_config(t_pose_marker, Joint, weights1)
            for i in range(N):
                M[i] = skinning_from_mrk_config(marker_config, J_t[i], J_R[i], weights1)
                M1[i, ref_idx] = M[i, ref_idx]
                M1[i, non_ref_idx] = corrupt(M[i, non_ref_idx], 0.1, 0.1, 0.3)
            np.savez(output_file, person_idx=person_rand_number, mrkconfig_idx=rand_number, M=M, M1=M1, J_t=J_t, J_R=J_R, shape=shape, J=Joint, Marker=t_pose_marker,
                     weights=weights1)



