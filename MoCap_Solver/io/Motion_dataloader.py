from torch.utils.data import Dataset
import copy
from MoCap_Solver.io.motion_dataset import MotionData
import os
import numpy as np
import torch
from tqdm import tqdm
from utils.utils import topology, ee_ids, prev_list

class MixedData0(Dataset):
    """
    Mixed data for many skeletons but one topologies
    """
    def __init__(self, args, motions, skeleton_idx):
        super(MixedData0, self).__init__()

        self.motions = motions
        self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.args.data_augment == 0 or torch.rand(1) < 0.5:
            return [self.motions[item], self.skeleton_idx[item]]
        else:
            return [self.motions_reverse[item], self.skeleton_idx[item]]


class MixedData(Dataset):
    """
    data_gruop_num * 2 * samples
    """
    def __init__(self, args, datasets_groups, istrain=True):
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.length = 0
        self.offsets = []
        self.joint_topologies = []
        self.ee_ids = []
        self.means = []
        self.vars = []
        dataset_num = 0
        seed = 19260817
        total_length = 10000000
        all_datas = []
        for datasets in datasets_groups:
            offsets_group = []
            dataset_num += len(datasets)
            tmp = []
            for i, dataset in tqdm(enumerate(datasets)):
                new_args = copy.copy(args)
                new_args.data_augment = 0
                new_args.dataset = dataset
                tmp.append(MotionData(new_args))
                h = np.load(os.path.join(dataset))
                if i == 0:
                    self.joint_topologies.append(topology)
                    self.ee_ids.append(ee_ids)
                J = h['J'][:, :]
                new_offset = J - J[prev_list]
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)

                total_length = min(total_length, len(tmp[-1]))
            all_datas.append(tmp)
            datas = [mdd.data for mdd in tmp]
            offsets_group = torch.cat(offsets_group, dim=0)
            offsets_group = offsets_group.to(device)
            #### statistic the mean and var ###############
            if istrain:
                print('statistic!')
                L = torch.cat(datas, dim=0)
                var = torch.var(L, (0, 2), keepdim=True)
                var = var ** (1 / 2)
                mean = torch.mean(L, (0, 2), keepdim = True)
                means_group = [mean for dataset in datasets]
                vars_group = [var for dataset in datasets]

                means_group = torch.cat(means_group, dim=0).to(device)
                vars_group = torch.cat(vars_group, dim=0).to(device)
                vars_group = vars_group.cpu().numpy()
                idx = vars_group < 1e-5
                vars_group[idx] = 1.
                vars_group = torch.tensor(vars_group).to(device)
                torch.save(means_group, os.path.join('models', 'motion_data_mean_group.pt'))
                torch.save(vars_group, os.path.join('models', 'motion_data_var_group.pt'))
            means_group = torch.load( os.path.join('models', 'motion_data_mean_group.pt'))
            vars_group = torch.load(os.path.join('models', 'motion_data_var_group.pt'))
            means_group = means_group.to(device)
            vars_group = vars_group.to(device)
            idx = vars_group < 1e-5
            vars_group[idx] = 1
            means_group0 = means_group[0][:]
            vars_group0 = vars_group[0][:]
            means_group0 = means_group0.to(torch.device('cpu'))
            vars_group0 = vars_group0.to(torch.device('cpu'))
            for mdd in tmp:
                mdd.data = (mdd.data - means_group0) / vars_group0
                mdd.mean = means_group0
                mdd.var = vars_group0
            self.offsets.append(offsets_group)
            self.means.append(means_group)
            self.vars.append(vars_group)

        offset_train_data = np.load(os.path.join('models', 'train_data_ts.npy'))
        offset_train_data = np.array(offset_train_data, dtype=np.float32)
        offset_train_data = torch.tensor(offset_train_data).to(device)
        self.offset_train_data = offset_train_data
        for datasets in all_datas:
            pt = 0
            motions = []
            skeleton_idx = []
            for dataset in datasets:
                motions.append(dataset[:])
                skeleton_idx += [pt] * len(dataset)
                pt += 1
            motions = torch.cat(motions, dim=0)
            if self.length != 0 and self.length != len(skeleton_idx):
                self.length = min(self.length, len(skeleton_idx))
            else:
                self.length = len(skeleton_idx)
            self.final_data.append(MixedData0(args, motions, skeleton_idx))

    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        return data * var + means

    def offset_denorm(self,  data):
        return data * self.offset_train_data[1, ...] + self.offset_train_data[0, ...]

    def offset_norm(self,  data):
        return (data - self.offset_train_data[0, ...]) / self.offset_train_data[1, ...]


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        res = []
        for data in self.final_data:
            res.append(data[item])
        return res

