import numpy as np
from pyquaternion import Quaternion
import math

def get_transfer_matrix(vecs_1, vecs_2):
    '''
    Attention: the three points should be non-collinear
    input: list 1 and list 2 ([[x1,y1,z1],[x2,y2,z2],[....]])
    return: R and T
    '''
    _vecs_1 = np.array(vecs_1)
    _vecs_2 = np.array(vecs_2)
    _mean_vec1 = np.mean(_vecs_1, axis=0)
    _mean_vec2 = np.mean(_vecs_2, axis=0)
    _shift1 = _vecs_1 - _mean_vec1.reshape(1, 3)  # np.tile(_mean_vec1, (N, 1))
    _shift2 = _vecs_2 - _mean_vec2.reshape(1, 3)  # np.tile(_mean_vec2, (N, 1))
    X = _shift1.T
    Y = _shift2.T
    _t = X.dot(Y.T)
    U, Sigma, Vt = np.linalg.svd(_t)
    _reflection = np.identity(3)
    _reflection[2, 2] = np.linalg.det(Vt.T.dot(U.T))
    R = Vt.T.dot(_reflection)
    R = R.dot(U.T)
    # print (np.shape(R),np.shape(_mean_vec1))
    T = - R.dot(_mean_vec1) + _mean_vec2
    T = T.reshape(-1, 1)
    return R, T

def get_marker_config1(markers_pos, joints_pos, weights):
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

def get_marker_config(markers_pos, joints_pos, joints_transform, weights):
    """
    Compute the marker configuration from the bind pose.
    Args:
    markers_pos: The position of markers: (56, 3)
    joints_pos: The position of joints: (31, 3)
    joints_transform: The rotation matrix of joints: (31, 3, 3)
    weights: The skinning weights: (56, 31)
    return:
        marker_configuration: (56, 31, 3)
    """
    _offset_list = list()
    mrk_pos_matrix = np.array(markers_pos)
    jts_pos_matrix = np.array(joints_pos)
    jts_rot_matrix = np.array(joints_transform)
    weights_matrix = np.array(weights)
    tile_num = joints_pos.shape[0]
    for mrk_index in range(mrk_pos_matrix.shape[0]):
        mark_pos = mrk_pos_matrix[mrk_index]
        jts_offset = mark_pos - jts_pos_matrix
        for jt_index in range(jts_pos_matrix.shape[0]):
            jts_offset = np.linalg.inv(jts_rot_matrix).dot(jts_offset[jt_index])
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

def rot_error(r_gt, r_est):
    RR= np.dot(np.linalg.inv(r_gt),r_est)
    R_trace = np.trace(RR)
    r = (R_trace -1) /2.
    r = np.clip(r, -1, 1)
    dis = abs(math.acos(r))
    return dis

def get_Quaternion_fromRotationMatrix(mat):
    q = Quaternion(matrix=mat, atol=1., rtol=1.)
    res = np.array([q.w, q.x, q.y, q.z])
    return res