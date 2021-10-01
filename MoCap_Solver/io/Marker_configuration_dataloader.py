from torch.utils.data import Dataset
import os
import numpy as np
import glob
from utils.utils import prev_list, MARKERNUM, JOINTNUM


class Marker_configuration_dataloader(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """

    def __init__(self, args):
        super(Marker_configuration_dataloader, self).__init__()
        file_path_dir = os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'marker_configuration_dataset')
        test_file_path_dir = os.path.join('MoCap_Solver', 'data', 'Testing_dataset', 'marker_configuration_dataset')
        npz_files = glob.glob(os.path.join(file_path_dir, '*.npz'))
        test_npz_files = glob.glob(os.path.join(test_file_path_dir, '*.npz'))
        self.data = np.zeros((len(npz_files), MARKERNUM, JOINTNUM, 3), dtype=np.float32)
        self.test_data = np.zeros((len(test_npz_files), MARKERNUM, JOINTNUM, 3), dtype=np.float32)
        self.ts_data = np.zeros((len(npz_files), JOINTNUM, 3), dtype=np.float32)
        self.test_ts_data = np.zeros((len(test_npz_files), JOINTNUM, 3), dtype=np.float32)
        self.weights = np.load(os.path.join('models', 'weights.npy'))
        for npz_file_idx in range(len(npz_files)):
            npz_file = npz_files[npz_file_idx]
            h = np.load(npz_file)
            Marker = h['Marker']
            J = h['J']
            prev_J = J[prev_list, :]
            self.ts_data[npz_file_idx, :, :] = J - prev_J
            mc = self.get_marker_config(Marker, J, self.weights)
            self.data[npz_file_idx, :, :, :] = mc

        for npz_file_idx in range(len(test_npz_files)):
            npz_file = test_npz_files[npz_file_idx]
            h = np.load(npz_file)
            Marker = h['Marker']
            J = h['J']
            prev_J = J[prev_list, :]
            self.test_ts_data[npz_file_idx, :, :] = J - prev_J
            mc = self.get_marker_config(Marker, J, self.weights)
            self.test_data[npz_file_idx, :, :, :] = mc

        self.total_frame = len(npz_files)
        self.args = args
        self.test_set = self.test_data
        self.data = self.data
        self.ts_data = self.ts_data
        self.test_ts_data = self.test_ts_data
        self.mean_data = self.get_batch_mu(self.data)
        self.sigma_data = self.get_batch_sigma(self.data)
        self.sigma_data[self.sigma_data < 1e-5] = 1.
        np.save(os.path.join('models', 'train_data_marker_config.npy'), np.array([self.mean_data, self.sigma_data]))

    def get_marker_config(self, markers_pos, joints_pos, weights):
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

    def get_batch_mu(self, batch_data):
        return np.mean(batch_data, axis=0)

    def get_batch_sigma(self, batch_data):
        return np.std(batch_data, ddof=1,
                      axis=0)  # calculate the sample std instead of the global std,therefore ddof should be set 1

    def reset_length(self, length):
        # self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]

        return [self.data[item], self.ts_data[item]]

    def norm(self, data):
        return (data - self.mean_data) / self.sigma_data

    def denorm(self, data):
        return data * self.sigma_data + self.mean_data
