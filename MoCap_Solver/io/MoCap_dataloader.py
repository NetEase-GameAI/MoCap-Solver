import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from MoCap_Solver.model.Kinematics import ForwardKinematics
from utils.utils import topology, FRAMENUM, JOINTNUM, MARKERNUM, get_RotationMatrix_fromQuaternion
npzs = glob.glob(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', '*.npz'))
test_npzs = glob.glob(os.path.join('MoCap_Solver', 'data', 'Testing_dataset', 'mocap_dataset', '*.npz'))
train_list = [npz for npz in npzs[:]]
test_list = [npz for npz in test_npzs[:]]
class TrainData(Dataset):
    def __init__(self, args, statistic_on=False):
        '''
                file_dirs: the all dirs of the data
                weights_path: the path of weights matrix
                epoch: the epoch number
                batch_size: train batch
                label_size: the lable size
                file_type: the file type of data
        '''
        # self.fk = ForwardKinematics(topology)
        self.train_data_jts_label_list = list()
        self.train_data_mean = 0
        self.train_data_sigma = 0
        self.train_label_mean = 0
        self.train_label_sigma = 0
        self.train_marker_config_mean = 0
        self.train_marker_config_sigma = 0
        self.weighted_marker_config_mean = 0
        self.weighted_marker_config_sigma = 0
        self.joint_num = JOINTNUM
        self.mrk_num = MARKERNUM
        self.f_col = 3
        self.f_ch = 1
        npzs = glob.glob(os.path.join('MoCap_Solver','data', 'Training_dataset', 'mocap_dataset', '*.npz'))
        train_list = [npz for npz in npzs[:]]
        train_list = np.array(train_list)
        self.npzfiles = train_list[:]
        totalframe = 0
        for npzfile in self.npzfiles:
            filename = os.path.basename(npzfile).split('.')[0]
            if os.path.exists(os.path.join('MoCap_Solver','data', 'Training_dataset', 'mocap_dataset', filename + '.npz')):
                npzfile = os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', filename + '.npz')
                h = np.load(npzfile)
                M = h['M']
                totalframe += M.shape[0]
        self.totalframe = totalframe
        self.raw_markers = np.zeros((totalframe, FRAMENUM, MARKERNUM, 3))
        self.clean_markers = np.zeros((totalframe, FRAMENUM, MARKERNUM, 3))
        self.skeleton_pos = np.zeros((totalframe, FRAMENUM, JOINTNUM, 3))
        self.motion = np.zeros((totalframe, (JOINTNUM-1)* 4 + 3, FRAMENUM))
        self.offsets = np.zeros((totalframe, JOINTNUM, 3))
        self.marker_config = np.zeros((totalframe, MARKERNUM, JOINTNUM, 3))
        self.motion_latent = np.zeros((totalframe, 1792))
        self.offsets_latent = np.zeros((totalframe, 168))
        self.mc_latent_code = np.zeros((totalframe, 1024))
        self.first_rot = np.zeros((totalframe, FRAMENUM, 4))
        self.transform = np.zeros((totalframe, FRAMENUM, JOINTNUM, 3, 3))
        currentframe = 0
        currentidx = 0
        for npzfile in tqdm(self.npzfiles):
            filename = os.path.basename(npzfile).split('.')[0]
            if os.path.exists(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', filename + '.npz')):
                h = np.load(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'mocap_dataset', filename + '.npz'))
                M1 = h['M1']
                # h = np.load(npzfile)
                M = h['M']
                # M1 = h['M1']
                offsets = h['offsets']
                mrk_config = h['mrk_config']
                mc_latent_code = h['mc_latent_code']
                motion_latent = h['motion_latent']
                motion = h['motion']
                N = M.shape[0]
                J_R = h['J_R']
                J_t = h['J_t']
                first_rot = h['first_rot']
                offsets_latent = h['offsets_latent']
                self.raw_markers[currentframe: (currentframe + N)] = M1
                self.clean_markers[currentframe: (currentframe + N)] = M
                self.skeleton_pos[currentframe: (currentframe + N)] = J_t
                self.motion[currentframe: (currentframe + N)] = motion
                self.offsets[currentframe: (currentframe + N)] = offsets
                self.marker_config[currentframe: (currentframe + N)] = mrk_config
                self.motion_latent[currentframe: (currentframe + N)] = motion_latent
                self.offsets_latent[currentframe: (currentframe + N)] = offsets_latent
                self.mc_latent_code[currentframe: (currentframe + N)] = mc_latent_code
                self.first_rot[currentframe: (currentframe + N)] = first_rot
                self.transform[currentframe: (currentframe + N)] = J_R
                currentidx += 1
                currentframe += N

        # for i in tqdm(range(totalframe)):
        #     rot = get_RotationMatrix_fromQuaternion(self.first_rot[i])
        #     self.first_rot_r6d[i] = np.array([rot[0, 0], rot[1,0], rot[2,0], rot[0,1], rot[1,1], rot[2,1]])

        self.length = self.raw_markers.shape[0]
        if statistic_on:
            train_marker_data_mean = self.get_batch_mu(self.clean_markers.reshape(-1, FRAMENUM, MARKERNUM, 3))
            train_marker_data_var = self.get_batch_sigma(self.clean_markers.reshape(-1, FRAMENUM, MARKERNUM, 3))
            train_marker_data_var[train_marker_data_var<1e-4] = 1.
            self.train_marker_data = np.array([train_marker_data_mean, train_marker_data_var])
            np.save(os.path.join('models', 'train_marker_data.npy'), self.train_marker_data)
            train_motion_code_data_mean = self.get_batch_mu(self.motion_latent)
            train_motion_code_data_var = self.get_batch_sigma(self.motion_latent)
            train_motion_code_data_var[train_motion_code_data_var<1e-4] = 1.
            self.train_motion_code_data = np.array([train_motion_code_data_mean, train_motion_code_data_var])
            np.save(os.path.join('models', 'train_motion_code_data.npy'), self.train_motion_code_data)
            train_offset_code_data_mean = self.get_batch_mu(self.offsets_latent)
            train_offset_code_data_var = self.get_batch_sigma(self.offsets_latent)
            train_offset_code_data_var[train_offset_code_data_var<1e-4] = 1.
            self.train_offset_code_data = np.array([train_offset_code_data_mean, train_offset_code_data_var])
            np.save(os.path.join('models', 'train_offset_code_data.npy'), self.train_offset_code_data)
            train_mc_code_data_mean = self.get_batch_mu(self.mc_latent_code)
            train_mc_code_data_var = self.get_batch_sigma(self.mc_latent_code)
            train_mc_code_data_var[train_mc_code_data_var<1e-4] = 1.
            self.train_mc_code_data = np.array([train_mc_code_data_mean, train_mc_code_data_var])
            np.save(os.path.join('models', 'train_mc_code_data.npy'), self.train_mc_code_data)
            train_first_rot_data_mean = self.get_batch_mu(self.first_rot)
            train_first_rot_data_var = self.get_batch_sigma(self.first_rot)
            train_first_rot_data_var[train_first_rot_data_var<1e-4] = 1.
            self.train_first_rot_data = np.array([train_first_rot_data_mean, train_first_rot_data_var])
            np.save(os.path.join('models', 'train_first_rot_data.npy'), self.train_first_rot_data)
        self.train_marker_data = np.load(os.path.join('models', 'train_marker_data.npy'))
        self.raw_markers = (self.raw_markers - self.train_marker_data[0])/self.train_marker_data[1]
        self.clean_markers = torch.tensor(self.clean_markers, dtype=torch.float32)
        self.raw_markers = torch.tensor(self.raw_markers, dtype=torch.float32)
        self.skeleton_pos = torch.tensor(self.skeleton_pos, dtype=torch.float32)
        self.motion = torch.tensor(self.motion, dtype=torch.float32)
        self.offsets = torch.tensor(self.offsets, dtype=torch.float32)
        self.marker_config = torch.tensor(self.marker_config, dtype=torch.float32)
        self.first_rot = torch.tensor(self.first_rot, dtype=torch.float32)
        # self.first_rot_r6d = torch.tensor(self.first_rot_r6d, dtype=torch.float32)
        self.offsets_latent = torch.tensor(self.offsets_latent, dtype=torch.float32)
        self.mc_latent_code = torch.tensor(self.mc_latent_code, dtype=torch.float32)
        self.transform = torch.tensor(self.transform, dtype=torch.float32)
        print(self.raw_markers.shape, self.clean_markers.shape, self.skeleton_pos.shape, self.motion.shape, self.offsets.shape,
                self.marker_config.shape, self.first_rot.shape)
        return

    def __len__(self):
        return self.length

    def update_noise(self):
        del self.raw_markers
        self.raw_markers = np.zeros((self.totalframe, FRAMENUM, MARKERNUM, 3))
        currentframe = 0
        for npzfile in self.npzfiles:
            filename = os.path.basename(npzfile).split('.')[0]
            M1 = np.load(os.path.join('MoCap_Solver', 'data', 'training_noise_windows_data', filename+'.npy'))
            N = M1.shape[0]
            self.raw_markers[currentframe: (currentframe + N)] = M1
            currentframe += N
        self.raw_markers = (self.raw_markers - self.train_marker_data[0]) / self.train_marker_data[1]
        self.raw_markers = torch.tensor(self.raw_markers, dtype=torch.float32)

    def __getitem__(self, item):
        # raw_marker, clean_marker, skeleton_pos, motion, offsets, marker_config, first_rot
        return [self.raw_markers[item], self.clean_markers[item], self.skeleton_pos[item], self.motion[item], self.offsets[item],
                self.marker_config[item], self.first_rot[item], self.offsets_latent[item], self.mc_latent_code[item], self.transform[item]]

    def get_batch_mu(self,batch_data):
        return np.mean(batch_data, axis=0)

    def get_batch_sigma(self,batch_data):
        return np.std(batch_data, ddof=1, axis=0) #calculate the sample std instead of the global std,therefore ddof should be set 1


class ValData(Dataset):
    def __init__(self, args):
        '''
                file_dirs: the all dirs of the data
                weights_path: the path of weights matrix
                epoch: the epoch number
                batch_size: train batch
                label_size: the lable size
                file_type: the file type of data
        '''
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        # self.weights = np.load(self.weights_file_path, allow_pickle=True)
        self.train_data_jts_label_list = list()
        self.train_data_mean = 0
        self.train_data_sigma = 0
        self.train_label_mean = 0
        self.train_label_sigma = 0
        self.train_marker_config_mean = 0
        self.train_marker_config_sigma = 0
        self.weighted_marker_config_mean = 0
        self.weighted_marker_config_sigma = 0
        self.joint_num = JOINTNUM
        self.mrk_num = MARKERNUM
        self.f_col = 3
        self.f_ch = 1
        npzfiles = test_list
        totalframe = 0
        for npzfile in tqdm(npzfiles):
            h = np.load(npzfile)
            M = h['M']
            totalframe += M.shape[0]
        self.raw_markers = np.zeros((totalframe, FRAMENUM, MARKERNUM, 3))
        self.clean_markers = np.zeros((totalframe, FRAMENUM, MARKERNUM, 3))
        self.skeleton_pos = np.zeros((totalframe, FRAMENUM, JOINTNUM, 3))
        self.motion = np.zeros((totalframe, 4 * (JOINTNUM - 1) + 3, FRAMENUM))
        self.offsets = np.zeros((totalframe, JOINTNUM, 3))
        self.marker_config = np.zeros((totalframe, MARKERNUM, JOINTNUM, 3))
        self.motion_latent = np.zeros((totalframe, 1792))
        self.offsets_latent = np.zeros((totalframe, 168))
        self.mc_latent_code = np.zeros((totalframe, 1024))
        self.first_rot = np.zeros((totalframe, FRAMENUM, 4))
        # self.first_rot_r6d = np.zeros((totalframe, FRAMENUM, 6))
        self.transform = np.zeros((totalframe, FRAMENUM, JOINTNUM, 3, 3))
        currentframe = 0
        currentidx = 0
        for npzfile in tqdm(npzfiles):
            filename = os.path.basename(npzfile).split('.')[0]
            h1 = np.load(os.path.join('MoCap_Solver', 'data', 'Testing_dataset', 'mocap_dataset', filename + '.npz'))
            h = np.load(npzfile)
            M = h['M']
            M1 = h1['M1']
            offsets = h1['offsets']
            mrk_config = h1['mrk_config']
            mc_latent_code = h1['mc_latent_code']
            motion_latent = h1['motion_latent']
            motion = h1['motion']
            N = M.shape[0]
            J_R = h1['J_R']
            J_t = h1['J_t']
            first_rot = h1['first_rot']
            offsets_latent = h1['offsets_latent']
            self.raw_markers[currentframe: (currentframe + N)] = M1
            self.clean_markers[currentframe: (currentframe + N)] = M
            self.skeleton_pos[currentframe: (currentframe + N)] = J_t
            self.motion[currentframe: (currentframe + N)] = motion
            self.offsets[currentframe: (currentframe + N)] = offsets
            self.marker_config[currentframe: (currentframe + N)] = mrk_config
            self.motion_latent[currentframe: (currentframe + N)] = motion_latent
            self.offsets_latent[currentframe: (currentframe + N)] = offsets_latent
            self.mc_latent_code[currentframe: (currentframe + N)] = mc_latent_code
            self.first_rot[currentframe: (currentframe + N)]  = first_rot
            self.transform[currentframe: (currentframe + N)] = J_R
            currentidx += 1
            currentframe += N
        self.length = self.raw_markers.shape[0]
        self.train_marker_data = np.load(os.path.join('models', 'train_marker_data.npy'))
        self.raw_markers = (self.raw_markers - self.train_marker_data[0])/self.train_marker_data[1]
        self.clean_markers = torch.tensor(self.clean_markers, dtype=torch.float32).to(device)
        self.raw_markers = torch.tensor(self.raw_markers, dtype=torch.float32).to(device)
        self.skeleton_pos = torch.tensor(self.skeleton_pos, dtype=torch.float32).to(device)
        self.motion = torch.tensor(self.motion, dtype=torch.float32).to(device)
        self.offsets = torch.tensor(self.offsets, dtype=torch.float32).to(device)
        self.marker_config = torch.tensor(self.marker_config, dtype=torch.float32).to(device)
        self.first_rot = torch.tensor(self.first_rot, dtype=torch.float32).to(device)
        # self.first_rot_r6d = torch.tensor(self.first_rot_r6d, dtype=torch.float32).to(device)
        self.offsets_latent = torch.tensor(self.offsets_latent, dtype=torch.float32)
        self.mc_latent_code = torch.tensor(self.mc_latent_code, dtype=torch.float32)
        self.transform = torch.tensor(self.transform, dtype=torch.float32)
        return

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return [self.raw_markers[item], self.clean_markers[item], self.skeleton_pos[item], self.motion[item], self.offsets[item],
                self.marker_config[item], self.first_rot[item], self.offsets_latent[item], self.mc_latent_code[item], self.transform[item]]

    def get_batch_mu(self,batch_data):
        return np.mean(batch_data, axis=0)

    def get_batch_sigma(self,batch_data):
        return np.std(batch_data, ddof=1, axis=0) #calculate the sample std instead of the global std,therefore ddof should be set 1

