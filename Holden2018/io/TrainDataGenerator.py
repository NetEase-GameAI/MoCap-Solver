import os
import glob
import numpy as np
from keras.utils import Sequence
from tqdm import tqdm
from utils.utils import MARKERNUM, JOINTNUM


class TrainGenerator(Sequence):
    def __init__(self, weights_path, statistic_on=False):
        '''
                weights_path: the path of weights matrix
                statistic_on: whether to statistic mean and vars of training data.
        '''
        self.epoch = 0
        self.sample_id = 0
        self.weights_file_path = weights_path
        self.weights = np.load(self.weights_file_path, allow_pickle=True)
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
        self.f_ch = 2
        npzfiles = glob.glob(os.path.join('Holden2018', 'data', 'Training_dataset', '*.npz'))
        totalframe = 0
        for npzfile in tqdm(npzfiles):
            h = np.load(npzfile)
            M = h['M'][:]
            totalframe += M.shape[0]
        totalframe = totalframe
        self.train_data_vec = np.zeros((totalframe, MARKERNUM, 3, 2))
        self.label_data_vec = np.zeros((totalframe, MARKERNUM, 3))
        self.train_data_jts = np.zeros((totalframe, JOINTNUM, 3, 4))
        self.train_data_weighed = np.zeros((totalframe,MARKERNUM,  3))
        currentframe = 0
        currentidx = 0
        for npzfile in tqdm(npzfiles):
            h = np.load(npzfile)
            M = h['M'][:]
            N = M.shape[0]
            J_all = h['J_all']
            M1 = h['M1']
            weighted_mrk_config = h['weighted_mrk_config']
            self.train_data_vec[currentframe:(currentframe + N), :,:, 0] = M1
            self.train_data_vec[currentframe:(currentframe + N), :, :, 1] = weighted_mrk_config
            self.label_data_vec[currentframe:(currentframe + N)] = M
            self.train_data_jts[currentframe:(currentframe + N)] = J_all
            self.train_data_weighed[currentframe:(currentframe + N)] = weighted_mrk_config
            currentidx += 1
            currentframe += N

        self.data_num = self.train_data_vec.shape[0]
        self.shuffle_list = np.random.permutation(self.data_num)

        if statistic_on:
            self.weighted_marker_config_mean = self.get_batch_mu(self.train_data_weighed)
            self.weighted_marker_config_sigma = self.get_batch_sigma(self.train_data_weighed)
            np.save(os.path.join('models', 'train_data_marker_config_holden.npy'), np.array([self.weighted_marker_config_mean, self.weighted_marker_config_sigma]))
            self.train_label_mean = self.get_batch_mu(self.label_data_vec)
            self.train_label_sigma = self.get_batch_sigma(self.label_data_vec)

            np.save(os.path.join('models', 'train_data_marker_holden.npy'), np.array([self.train_label_mean, self.train_label_sigma]))
            self.train_joint_mean = self.get_batch_mu(self.train_data_jts)
            self.train_joint_sigma = self.get_batch_sigma(self.train_data_jts)
            np.save(os.path.join('models', 'train_data_joint_holden.npy'),
                    np.array([self.train_joint_mean, self.train_joint_sigma]))
        else:
            train_mrk_config = np.load(os.path.join('models', 'train_data_marker_config_holden.npy'))
            self.weighted_marker_config_mean = train_mrk_config[0]
            self.weighted_marker_config_sigma = train_mrk_config[1]
            self.train_label_data = np.load(os.path.join('models', 'train_data_marker_holden.npy'))
            self.train_label_mean = self.train_label_data[0]
            self.train_label_sigma = self.train_label_data[1]
            self.train_joint_data = np.load(os.path.join('models', 'train_data_joint_holden.npy'))
            self.train_joint_mean = self.train_joint_data[0]
            self.train_joint_sigma = self.train_joint_data[1]
        return

    def __len__(self):
        return self.data_num

    def get_all_vectors(self):
        x_vec = self.train_data_vec
        y_vec0 = self.train_data_jts[:]
        x_vec[:, :, :, 0] = (x_vec[:, :, :, 0] - self.train_label_mean) / self.train_label_sigma
        x_vec[:, :, :,  1] = (x_vec[:, :, :,  1] - self.weighted_marker_config_mean) / self.weighted_marker_config_sigma
        return x_vec, y_vec0

    def __getitem__(self, idx):

        return

    def corrupt(self,x,sigma_occlude,sigma_shift,beta):
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
        _min_alpha_occ = min(abs(alpha_occ),2*_sigma_occlude_Normal)
        _min_alpha_shift = min(abs(alpha_shift),2*_sigma_shift_Normal)
        _size_Bernoulli = x.shape[0]
        X_occu = np.random.binomial(n=1,p=_min_alpha_occ,size=_size_Bernoulli)[:,np.newaxis] # from (len,) to (len,1)
        X_shift = np.random.binomial(n=1,p=_min_alpha_shift,size=_size_Bernoulli)[:,np.newaxis]
        X_v = np.random.uniform(-_beta,_beta,size=(_size_Bernoulli,3))
        X_corrupted = (x + X_shift * X_v) * (1 - X_occu) #element-wise product
        assert np.shape(X_corrupted) == np.shape(x)
        return X_corrupted

    def get_batch_mu(self,batch_data):
        return np.mean(batch_data, axis=0)

    def get_batch_sigma(self,batch_data):
        return np.std(batch_data, ddof=1, axis=0) #calculate the sample std instead of the global std,therefore ddof should be set 1


class ValGenerator(Sequence):
    def __init__(self, weights_path):
        '''
                weights_path: the path of weights matrix
        '''
        self.epoch = 0
        self.sample_id = 0
        self.weights_file_path = weights_path
        self.weights = np.load(self.weights_file_path, allow_pickle=True)
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
        self.f_ch = 2
        npzfiles = glob.glob(os.path.join('Holden2018', 'data', 'Testing_dataset', '*.npz'))
        totalframe = 0
        for npzfile in tqdm(npzfiles):
            h = np.load(npzfile)
            M = h['M'][:]
            totalframe += M.shape[0]
        totalframe = totalframe
        self.train_data_vec = np.zeros((totalframe, MARKERNUM, 3, 2))
        self.label_data_vec = np.zeros((totalframe, MARKERNUM, 3))
        self.train_data_jts = np.zeros((totalframe, JOINTNUM, 3, 4))
        self.train_data_weighed = np.zeros((totalframe,MARKERNUM,  3))
        # self.statistic_data_weighed = np.zeros((len(npzfiles), MARKER_NUM, JOINT_NUM, 3))
        # self.train_data_bone_length = np.zeros((totalframe, WINDOW_SIZE, len(prevlist)))
        currentframe = 0
        currentidx = 0
        for npzfile in tqdm(npzfiles):
            h = np.load(npzfile)
            M = h['M'][:]
            N = M.shape[0]
            J_all = h['J_all']
            M1 = h['M1']
            weighted_mrk_config = h['weighted_mrk_config']
            self.train_data_vec[currentframe:(currentframe + N), :, :, 0] = M1
            self.train_data_vec[currentframe:(currentframe + N), :, :, 1] = weighted_mrk_config
            self.label_data_vec[currentframe:(currentframe + N)] = M
            self.train_data_jts[currentframe:(currentframe + N)] = J_all
            self.train_data_weighed[currentframe:(currentframe + N)] = weighted_mrk_config
            currentidx += 1
            currentframe += N

        self.data_num = self.train_data_vec.shape[0]
        self.shuffle_list = np.random.permutation(self.data_num)

        train_mrk_config = np.load(os.path.join('models', 'train_data_marker_config_holden.npy'))
        self.weighted_marker_config_mean = train_mrk_config[0]
        self.weighted_marker_config_sigma = train_mrk_config[1]
        self.train_label_data = np.load(os.path.join('models', 'train_data_marker_holden.npy'))
        self.train_label_mean = self.train_label_data[0]
        self.train_label_sigma = self.train_label_data[1]
        self.train_joint_data = np.load(os.path.join('models', 'train_data_joint_holden.npy'))
        self.train_joint_mean = self.train_joint_data[0]
        self.train_joint_sigma = self.train_joint_data[1]
        return

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return

    def get_all_vectors(self):
        x_vec = self.train_data_vec
        y_vec0 = self.train_data_jts[:]
        x_vec[:, :,:, 0] = (x_vec[:, :, :,0] - self.train_label_mean) / self.train_label_sigma
        x_vec[:, :, :, 1] = (x_vec[:, :, :, 1] - self.weighted_marker_config_mean) / self.weighted_marker_config_sigma
        return x_vec, y_vec0

