import numpy as np
import os
from utils.utils import rot_error
import math

class Normalization(object):
    """
    This class is the module of data Normalization. It converts each windows of raw markers into the local coordinate \
    system of the first frame, and converts the output markers, joint rotations and position of our local space into \
    the global coordinate system.
    """
    def __init__(self):
        self.t_pose_marker = np.load(os.path.join('models', 't_pos_marker.npy'))
        self.ref_marker_idx = [54, 3, 46, 32, 36, 21, 7, 11]
        return

    def norm(self, sequence):
        """
        converts each windows of raw markers into the local coordinate system of the first frame.
        Args:
            sequence: The sequence data structure which contains the input raw markers: windows_raw_markers.

        Returns:
            sequence: The sequence data structure which contains the local coordinates of raw markers: windows_raw_markers
        """
        sequence.Rigid_R = list()
        sequence.Rigid_T = list()
        sequence.Rigid_R_inv = list()
        sequence.Rigid_T_inv = list()

        for _window_idx in range(sequence.window_num):
            ###### compute each rigid transformations(_Rigid_R, _Rigid_T) of each window ##############################
            _Rigid_R = []
            _Rigid_T = []
            _current_rot_angle = 1000.
            _current_frame = -1
            identity_rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            rot_error_list = []
            for _frame_idx in range(sequence.window_size):

                _ref_marker_pos = sequence.windows_raw_markers[_window_idx, _frame_idx, self.ref_marker_idx, :].copy()
                _Rigid_R1, _Rigid_T1 = self.get_transfer_matrix(_ref_marker_pos, self.t_pose_marker)
                de = rot_error(identity_rot, _Rigid_R1)
                rot_error_list.append(de)
            min_index = rot_error_list.index(np.min(rot_error_list))
            _ref_marker_pos = sequence.windows_raw_markers[_window_idx, min_index, self.ref_marker_idx, :].copy()
            _Rigid_R, _Rigid_T = self.get_transfer_matrix(_ref_marker_pos, self.t_pose_marker)

            _Rigid_T = np.mean(self.t_pose_marker, axis=0) - _Rigid_R.dot(np.mean(_ref_marker_pos, axis=0))
            _Rigid_T = _Rigid_T.reshape(3, 1)

            _Rigid_R_inv = np.linalg.inv(_Rigid_R)
            _Rigid_T_inv = (_Rigid_R_inv.dot(-_Rigid_T))
            sequence.Rigid_R.append(_Rigid_R)
            sequence.Rigid_T.append(_Rigid_T)
            sequence.Rigid_R_inv.append(_Rigid_R_inv)
            sequence.Rigid_T_inv.append(_Rigid_T_inv)
            ##### convert raw markers of each frame into the local coordinate system of the first frame ###############
            for _frame_idx in range(sequence.window_size):
                sequence.windows_raw_markers[_window_idx, _frame_idx, :, :] = (_Rigid_R.dot(
                    sequence.windows_raw_markers[_window_idx, _frame_idx, :, :].reshape(
                        sequence.marker_num, 3).T) + _Rigid_T).T
        #print("inv R error: " + str(np.mean(np.array(R_error))) + " T error: " + str(np.mean(np.array(T_error))))
        return sequence

    def denorm(self, sequence):
        """
        converts the output markers, joint rotations and position of our local space into the global coordinate system.
        Args:
            sequence: The sequence data structure which contains the local coordinates of predicted markers, joints:
            markers_output_windows_data, jts_pos_windows_data, jts_rot_windows_data

        Returns:
            sequence: The sequence data structure which contains the global coordinates of the predicted markers , joints:
            markers_output_windows_data, jts_pos_windows_data, jts_rot_windows_data
        """
        for _windows_idx in range(sequence.window_num):
            for _frame_idx in range(sequence.window_size):
                ##### convert predicted markers of local space into global coordinate system ###########################
                sequence.markers_output_windows_data[_windows_idx, _frame_idx, :, :] = (sequence.Rigid_R_inv[_windows_idx].dot(
                    sequence.markers_output_windows_data[_windows_idx, _frame_idx, :, :].T) + sequence.Rigid_T_inv[_windows_idx]).T
                ##### convert predicted joint rotations of local space into global coordinate system ###################
                sequence.jts_pos_windows_data[_windows_idx, _frame_idx, :, :] = (sequence.Rigid_R_inv[_windows_idx].dot(
                    sequence.jts_pos_windows_data[_windows_idx, _frame_idx, :, :].T) + sequence.Rigid_T_inv[_windows_idx]).T
                _joint_num = sequence.jts_pos_windows_data.shape[2]
                ##### convert predicted joint positions of local space into global coordinate system ###################
                for joint_idx in range(_joint_num):
                    sequence.jts_rot_windows_data[_windows_idx, _frame_idx, joint_idx, :, :] = sequence.Rigid_R_inv[_windows_idx].dot(sequence.jts_rot_windows_data[_windows_idx, _frame_idx, joint_idx, :, :])
        return sequence

    def get_transfer_matrix(self, vectors_1, vectors_2):
        """
        compute the rigid transformation from vectors_1 to vectors_2
        Attention: the three points should be non-collinear
        input: list 1 and list 2 ([[x1,y1,z1],[x2,y2,z2],[....]])
        return: R and T
        """
        _vectors_1 = np.array(vectors_1)
        _vectors_2 = np.array(vectors_2)
        _mean_vec1 = np.mean(_vectors_1, axis=0)
        _mean_vec2 = np.mean(_vectors_2, axis=0)
        _shift1 = _vectors_1 - _mean_vec1.reshape(1, 3)  # np.tile(_mean_vec1, (N, 1))
        _shift2 = _vectors_2 - _mean_vec2.reshape(1, 3)  # np.tile(_mean_vec2, (N, 1))
        _X = _shift1.T
        _Y = _shift2.T
        _t = _X.dot(_Y.T)
        _U, _Sigma, _Vt = np.linalg.svd(_t)
        _Reflection = np.identity(3)
        _Reflection[2, 2] = np.linalg.det(_Vt.T.dot(_U.T))
        _R = _Vt.T.dot(_Reflection)
        _R = _R.dot(_U.T)
        # print (np.shape(R),np.shape(_mean_vec1))
        _T = - _R.dot(_mean_vec1) + _mean_vec2
        _T = _T.reshape(-1, 1)
        return _R, _T
