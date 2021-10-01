from torch.utils.data import Dataset
import os
import numpy as np
import glob
from utils.utils import topology

class Template_skeleton_dataloader(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args):
        super(Template_skeleton_dataloader, self).__init__()
        training_file_path_dir = os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'template_skeleton_dataset')
        testing_file_path_dir = os.path.join('MoCap_Solver', 'data', 'Testing_dataset', 'template_skeleton_dataset')
        npz_files = glob.glob(os.path.join(training_file_path_dir, '*.npz'))
        testing_npz_files = glob.glob(os.path.join(testing_file_path_dir, '*.npz'))
        self.data = np.zeros((len(npz_files), 24, 3), dtype=np.float32)
        prev_list = topology
        for npz_file_idx in range(len(npz_files)):
            npz_file = npz_files[npz_file_idx]
            h = np.load(npz_file)
            Marker = h['J']
            prev_Marker = Marker[prev_list, :]
            self.data[npz_file_idx, :, :] = Marker - prev_Marker
        self.total_frame = len(npz_files)
        self.args = args
        self.test_set = np.zeros((len(testing_npz_files), 24, 3), dtype=np.float32)
        for npz_file_idx in range(len(testing_npz_files)):
            npz_file = testing_npz_files[npz_file_idx]
            h = np.load(npz_file)
            Marker = h['J']
            prev_Marker = Marker[prev_list, :]
            self.test_set[npz_file_idx, :, :] = Marker - prev_Marker
        self.mean_data = self.get_batch_mu(self.data)
        self.sigma_data = self.get_batch_sigma(self.data)
        self.sigma_data[self.sigma_data < 1e-5] = 1.
        np.save(os.path.join('models', 'train_data_ts.npy'), np.array([self.mean_data, self.sigma_data]))

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

        return self.data[item]
