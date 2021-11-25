from pyquaternion import Quaternion
from scipy.signal import savgol_filter
import numpy as np


class PostProcessing(object):
    """
    Post-Processing: smooth the trajectories of marker position
    """
    def __init__(self, marker_num=56, joint_num=24):
        self.joint_num = joint_num
        self.marker_num = marker_num

    def get_Euler_fromRotationMatrix(self, mat):
        q = Quaternion(matrix=mat, atol=1., rtol=1.)
        return q.yaw_pitch_roll

    def get_Quaternion_fromRotationMatrix(self, mat):
        q = Quaternion(matrix=mat, atol=1., rtol=1.)
        res = np.array([q.w, q.x, q.y, q.z])
        return res

    def get_RotationMatrix_fromQuaternion(self, t_quaternion):
        q = Quaternion(t_quaternion[0], t_quaternion[1], t_quaternion[2], t_quaternion[3])
        return q.rotation_matrix

    def post_process_mrks(self, pred_list, polynomial_degree, window_size):
        """
        Using savgol filter to smooth the marker positions
        Args:
            pred_list: The marker positions we have predicted.
            polynomial_degree: The polynomial degree of savgol filter
            window_size: The window size of savgol filter

        Returns:
            pred_res: The smoothed marker positions.
        """
        _pred_arr = np.array(pred_list)
        _pred_post_list = []
        for _marker_idx in range(_pred_arr.shape[1]):
            post_tmp_list = []
            for frame_idx in range(_pred_arr.shape[0]):
                post_tmp_list.append(_pred_arr[frame_idx, _marker_idx, :])
            _pos_arr = np.array(post_tmp_list)
            for ch in range(_pos_arr.shape[1]):
                _pos_arr[:, ch] = savgol_filter(_pos_arr[:, ch], window_size, polynomial_degree)
            _pred_post_list.append(_pos_arr)
        pred_res = list(np.transpose(_pred_post_list, (1, 0, 2)))
        return pred_res


    def post_processing(self, sequence):
        """
        Post-Processing:  smooth the trajectories of marker position
        """
        sequence.pred_marker_data = np.zeros((sequence.frame_num, self.marker_num, 3))
        sequence.pred_trans_data = np.zeros((sequence.frame_num, self.joint_num, 3))

        for _frame_idx in range(sequence.frame_num):
            _windows_idx = int(_frame_idx/sequence.step_size)
            _step_idx = _frame_idx%(sequence.step_size)

            if _windows_idx == 0:
                sequence.pred_marker_data[_frame_idx] =  sequence.markers_output_windows_data[_windows_idx, _step_idx]
                sequence.pred_trans_data[_frame_idx] =  sequence.jts_pos_windows_data[_windows_idx, _step_idx]
            if _windows_idx >= 1:
                sequence.pred_marker_data[_frame_idx] = 0.5 * (sequence.markers_output_windows_data[_windows_idx, _step_idx] + sequence.markers_output_windows_data[_windows_idx - 1, _step_idx + sequence.step_size])
                sequence.pred_trans_data[_frame_idx] = 0.5 * (sequence.jts_pos_windows_data[_windows_idx, _step_idx] + sequence.jts_pos_windows_data[_windows_idx - 1, _step_idx + sequence.step_size])

        sequence.pred_marker_data = self.post_process_mrks(sequence.pred_marker_data, polynomial_degree=7, window_size=51)
        sequence.pred_trans_data = self.post_process_mrks(sequence.pred_trans_data, polynomial_degree=7, window_size=51)
        sequence.jts_rot_windows_data = sequence.jts_rot_windows_data[:, :sequence.step_size, :, :, :]
        return sequence
