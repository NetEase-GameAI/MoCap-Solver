import numpy as np
import scipy.spatial.distance as scipy_dis


class Sequence(object):
    def __init__(self, seq_name, Markers, Joints=None, batch_size=32, window_size=64, fps=60, step_size=32):
        """

        Args:
            seq_name: The name of the sequence.
            Markers: The marker positions of the sequence.
            Joints: The joint information of the sequence
            batch_size: The size of the batch feeding to the network.
            window_size: The size of the window.
            fps: the fps of the sequence.
            step_size: The step size to split the sequence to windows.
        """
        self.raw_markers = np.array(Markers)
        self.raw_markers_first = self.raw_markers.copy()
        self.Joints = Joints
        self.frame_num = self.raw_markers.shape[0]
        self.marker_num = self.raw_markers.shape[1]
        self.batch_size = batch_size
        self.window_size = window_size
        self.step_size = step_size
        self.fps = fps
        self.window_num = np.int(np.ceil(np.float(self.frame_num) / self.step_size))
        self.batch_num = np.int(np.ceil(np.float(self.window_num) / self.batch_size))
        self.Rigid_R = list()
        self.Rigid_T = list()
        self.Rigid_R_inv = list()
        self.Rigid_T_inv = list()
        self.windows_raw_markers = np.zeros((self.window_num, self.window_size, self.marker_num, 3))
        self.input_batch_data = list()
        self.distance_image_list = list()
        self.distance_image_list = self.genrate_distance_image(self.raw_markers)
        self.reference_distance_matrix = self.distance_image_list[0]
        self.jts_rot_windows_data = list()
        self.jts_pos_windows_data = list()
        self.markers_output_windows_data = list()
        self.pred_marker_data = list()
        self.ref_unreliable_frames = list()

    def genrate_distance_image(self, Markers):
        """
        Generate distance images of each frame of the markers sequence.
        Args:
            Markers: Marker pos (frame_count * marker_num * 3)

        Returns:
            _markers_distance_matrix: Distance images (frame_count * marker_num * marker_num)

        """

        _markers_distance_matrix = list()
        frame_count = Markers.shape[0]
        for frame_idx in range(frame_count):
            _markers_arr = Markers[frame_idx]
            dis_arr = scipy_dis.pdist(_markers_arr, metric='euclidean')
            dis_mat = np.array(scipy_dis.squareform(dis_arr))
            _markers_distance_matrix.append(dis_mat)
        return _markers_distance_matrix

    def get_window_data(self):
        for _window_idx in range(self.window_num):
            for _frame_idx in range(self.window_size):
                self.windows_raw_markers[_window_idx, _frame_idx, :, :] = self.raw_markers[min(_window_idx * self.step_size + _frame_idx, self.frame_num - 1), :, :]
        return

    def get_batch_data(self):
        """
        Transfer the window_data into batches feeding to the neural network.
        """
        self.input_batch_data = list()
        for _batch_idx in range(self.batch_num):
            if _batch_idx < self.batch_num - 1:
                self.input_batch_data.append(self.windows_raw_markers[_batch_idx * self.batch_size: (
                            _batch_idx * self.batch_size + self.batch_size)])
            else:
                self.input_batch_data.append(self.windows_raw_markers[_batch_idx * self.batch_size:])
        return
