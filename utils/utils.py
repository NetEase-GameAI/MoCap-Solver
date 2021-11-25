# Classes in this file are mainly borrowed from Jun-Yan Zhu's cycleGAN repository
from torch import nn
import torch
import random
from torch.optim import lr_scheduler
import math
from pyquaternion import Quaternion
import numpy as np

topology = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
ee_ids = [10, 11, 15, 22, 23]
prev_list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

_dict_marker = {
    # this formate is subject to the name of the markers in file which stores the data of original markers
    "ARIEL": 0, "C7": 1, "CLAV": 2, "L4": 3, "LANK": 4, "LBHD": 5, "LBSH": 6, "LBWT": 7, "LELB": 8, "LFHD": 9,
    "LFSH": 10, "LFWT": 11, "LHEL": 12, "LHIP": 13,
    "LIEL": 14, "LIHAND": 15, "LIWR": 16, "LKNE": 17, "LKNI": 18, "LMT1": 19, "LMT5": 20, "LMWT": 21, "LOHAND": 22,
    "LOWR": 23, "LSHN": 24, "LTOE": 25, "LTSH": 26,
    "LUPA": 27, "LWRE": 28, "RANK": 29, "RBHD": 30, "RBSH": 31, "RBWT": 32, "RELB": 33, "RFHD": 34, "RFSH": 35,
    "RFWT": 36, "RHEL": 37, "RHIP": 38, "RIEL": 39, "RIHAND": 40,
    "RIWR": 41, "RKNE": 42, "RKNI": 43, "RMT1": 44, "RMT5": 45, "RMWT": 46, "ROHAND": 47, "ROWR": 48, "RSHN": 49,
    "RTOE": 50, "RTSH": 51, "RUPA": 52, "RWRE": 53, "STRN": 54,
    "T10": 55
}
_dict_marker.update(dict([reversed(_item) for _item in _dict_marker.items()]))
FRAMENUM = 64
JOINTNUM = 24
MARKERNUM = 56
BATCHSIZE = 256
STEPSIZE = 32
LAYERSNUM = 2
KERNELSIZE = 15
SKELETONDIST = 2
EXTRACONV = 0
CUDADIVICE = 'cuda:0'
LEARNINGRATE = 2e-4
fps = 120.
ref_idx = [54, 3, 46, 32, 36, 21, 7, 11]
_dict_joints = {
    "Pelvis": 0,
    "L_Hip": 1,
    "L_Knee": 2,
    "L_Ankle": 3,
    "L_Foot": 4,
    "R_Hip": 5,
    "R_Knee": 6,
    "R_Ankle": 7,
    "R_Foot": 8,
    "Spine1": 9,
    "Spine2": 10,
    "Spine3": 11,
    "L_Collar": 12,
    "L_Shoulder": 13,
    "L_Elbow": 14,
    "L_Wrist": 15,
    "L_Hand": 16,
    "Neck": 17,
    "Head": 18,
    "R_Collar": 19,

    "R_Shoulder": 20,
    "R_Elbow": 21,
    "R_Wrist": 22,
    "R_Hand": 23
}
_dict_joints.update(dict([reversed(_item) for _item in _dict_joints.items()]))


def get_Quaternion_fromRotationMatrix(mat):
    q = Quaternion(matrix=mat, atol=1., rtol=1.)
    res = np.array([q.w, q.x, q.y, q.z])
    return res

def get_RotationMatrix_fromQuaternion(t_quaternion):
    q = Quaternion(t_quaternion[0], t_quaternion[1], t_quaternion[2], t_quaternion[3])
    return q.rotation_matrix

def get_transfer_matrix(vecs_1, vecs_2):
    '''
    Attention: the three points should be non-collinear
    input: list 1 and list 2 ([[x1,y1,z1],[x2,y2,z2],[....]])
    return: R and T
    '''
    _vecs_1 = np.array(vecs_1)
    _vecs_2 = np.array(vecs_2)
    N = _vecs_1.shape[0]
    _mean_vec1 = np.mean(_vecs_1, axis=0)
    _mean_vec2 = np.mean(_vecs_2, axis=0)

    # print ('_mean_vec1',_mean_vec1,'_mean_vec2',_mean_vec2)
    _shift1 = _vecs_1 - _mean_vec1.reshape(1, 3)  # np.tile(_mean_vec1, (N, 1))
    _shift2 = _vecs_2 - _mean_vec2.reshape(1, 3)  # np.tile(_mean_vec2, (N, 1))
    # print ('_shift1 shape',_shift1.shape)
    X = _shift1.T
    Y = _shift2.T
    # print ("X shape {} Y shape{}".format(np.shape(X),np.shape(Y)))
    _t = X.dot(Y.T)
    # print (np.shape(_t))
    U, Sigma, Vt = np.linalg.svd(_t)

    Reflection = np.identity(3)
    Reflection[2, 2] = np.linalg.det(Vt.T.dot(U.T))
    # print ('Reflection',Reflection)
    R = Vt.T.dot(Reflection)
    R = R.dot(U.T)
    # print (np.shape(R),np.shape(_mean_vec1))
    T = - R.dot(_mean_vec1) + _mean_vec2
    T = T.reshape(-1, 1)
    return R, T


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


def get_weighed_marker_config(markers_pos, joints_pos, weights):
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
    mc = np.array(_offset_list)
    wmc = (mc * np.tile(weights.reshape(mrk_pos_matrix.shape[0], tile_num, 1), (1, 1, 3))).sum(axis=1)
    return wmc

class HuberLoss(nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.HUBER_DELTA = delta

    def forward(self, input, target):
        error_mat = input - target
        _error_ = torch.sqrt(torch.sum(error_mat **2))
        HUBER_DELTA = self.HUBER_DELTA
        switch_l = _error_<HUBER_DELTA
        switch_2 = _error_>=HUBER_DELTA
        x = switch_l * (0.5* _error_**2 ) + switch_2 * (0.5* HUBER_DELTA**2 + HUBER_DELTA*(_error_-HUBER_DELTA))
        return x

class angle_loss(nn.Module):
    def __init__(self):
        super(angle_loss, self).__init__()

    def forward(self, input, target):
        y_pred1 = input[..., 0, :]
        y_pred2 = input[..., 1, :]
        y_pred3 = input[..., 2, :]
        y_pred1 = y_pred1.view(-1,  1, 3)
        y_pred2 = y_pred2.view(-1, 1, 3)
        y_pred3 = y_pred3.view(-1, 1, 3)
        y_pred1 = y_pred1.repeat([1, 3, 1])
        y_pred2 = y_pred2.repeat([1,  3, 1])
        y_pred3 = y_pred3.repeat([1, 3, 1])
        target = target.view(-1, 3, 3)
        z_pred1 = target * y_pred1
        z_pred2 = target * y_pred2
        z_pred3 = target * y_pred3
        z_pred1 = torch.sum(z_pred1, axis=-1)
        z_pred2 = torch.sum(z_pred2, axis=-1)
        z_pred3 = torch.sum(z_pred3, axis=-1)
        z_pred1 = z_pred1.view(-1,  3, 1)
        z_pred2 = z_pred2.view(-1,  3, 1)
        z_pred3 = z_pred3.view(-1,  3, 1)
        z_pred = torch.cat([z_pred1, z_pred2, z_pred3], axis=2)
        # z_pred_trace = torch.trace(z_pred)
        z_pred_trace = z_pred[:,  0, 0] + z_pred[:,  1, 1] + z_pred[:,  2, 2]
        z_pred_trace = (z_pred_trace - 1.)/2.0000000000
        z_pred_trace = torch.clamp(z_pred_trace, -1.0, 1.0)
        z_pred_trace = torch.acos(z_pred_trace)
        z_pred_trace = z_pred_trace * 180./3.141592653
        error = torch.mean(z_pred_trace)
        return error

class vertex_loss(nn.Module):
    def __init__(self):
        super(vertex_loss, self).__init__()

    def forward(self, input, target):

        return torch.mean(torch.sqrt(torch.sum(torch.pow(input - target, 2), dim=-1)))


class GAN_loss(nn.Module):
    def __init__(self, gan_mode, real_lable=1.0, fake_lable=0.0):
        super(GAN_loss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_lable))
        self.register_buffer('fake_label', torch.tensor(fake_lable))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'none':
            self.loss = None
        else:
            raise Exception('Unknown GAN mode')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss



class Criterion_EE:
    def __init__(self, args, base_criterion, norm_eps=0.008):
        self.args = args
        self.base_criterion = base_criterion
        self.norm_eps = norm_eps

    def __call__(self, pred, gt):
        reg_ee_loss = self.base_criterion(pred, gt)
        if self.args.ee_velo:
            gt_norm = torch.norm(gt, dim=-1)
            contact_idx = gt_norm < self.norm_eps
            extra_ee_loss = self.base_criterion(pred[contact_idx], gt[contact_idx])
        else:
            extra_ee_loss = 0
        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return []

class Criterion_EE_2:
    def __init__(self, args, base_criterion, norm_eps=0.008):
        print('Using adaptive EE')
        self.args = args
        self.base_criterion = base_criterion
        self.norm_eps = norm_eps
        self.ada_para = nn.Linear(15, 15).to(torch.device(args.cuda_device))

    def __call__(self, pred, gt):
        pred = pred.reshape(pred.shape[:-2] + (-1,))
        gt = gt.reshape(gt.shape[:-2] + (-1,))
        pred = self.ada_para(pred)
        reg_ee_loss = self.base_criterion(pred, gt)
        extra_ee_loss = 0
        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return list(self.ada_para.parameters())

class Eval_Criterion:
    def __init__(self, parent):
        self.pa = parent
        self.base_criterion = nn.MSELoss()
        pass

    def __call__(self, pred, gt):
        for i in range(1, len(self.pa)):
            pred[..., i, :] += pred[..., self.pa[i], :]
            gt[..., i, :] += pred[..., self.pa[i], :]
        return self.base_criterion(pred, gt)


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_ee(pos, pa, ees, velo=False, from_root=False):
    pos = pos.clone()
    for i, fa in enumerate(pa):
        if i == 0: continue
        if not from_root and fa == 0: continue
        pos[:, :, i, :] += pos[:, :, fa, :]

    pos = pos[:, :, ees, :]
    if velo:
        pos = pos[:, 1:, ...] - pos[:, :-1, ...]
        pos = pos * 10
    return pos


def rot_error(r_gt, r_est):
    RR= np.dot(np.linalg.inv(r_gt),r_est)
    R_trace = np.trace(RR)
    r = (R_trace -1) /2.
    r = np.clip(r, -1, 1)
    dis = abs(math.acos(r))
    return dis

def simulate_outlier_remove(M, M1):
    remove_threshold = 0.2
    detM = np.sqrt( np.sum( np.square(M1-M), axis = -1) )
    M1[detM > remove_threshold] = [0.0,0.0,0.0]
    return M1