from torch.utils.data import Dataset
import os
import sys
import numpy as np
import torch

class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args):
        super(MotionData, self).__init__()
        name = args.dataset
        file_path = os.path.join(name)
        h = np.load(file_path)
        self.total_frame = 0
        self.args = args
        self.data = []
        motions = h['motion']
        self.motion_length = []
        new_windows = torch.tensor(motions, dtype=torch.float32)
        self.data.append(new_windows)
        self.data = torch.cat(self.data)
        train_len = self.data.shape[0]
        self.test_set = self.data[train_len:, ...]
        self.data = self.data[:train_len, ...]
        self.data_reverse = torch.tensor(self.data.numpy()[..., ::1].copy(), dtype = torch.float)
        self.reset_length_flag = 0
        self.virtual_length = 0

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]
        if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
            return self.data[item]
        else:
            return self.data_reverse[item]

    def subsample(self, motion):
        return motion[::1, :]

    def denormalize(self, motion):
        if self.args.normalization:
            if self.var.device != motion.device:
                self.var = self.var.to(motion.device)
                self.mean = self.mean.to(motion.device)
            ans = motion * self.var + self.mean
        else: ans = motion
        return ans
