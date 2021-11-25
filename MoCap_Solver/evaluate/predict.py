from MoCap_Solver.model.MoCapSolverClass import MoCapSolverModel
import numpy as np
import os
import utils.option_parser as option_parser
from utils.utils import topology
import torch

class PredictClass(object):
    """
    This class is to predict the marker pos, joint rotations and positions from neural network.
    """

    def __init__(self):
        """
        Initialization: Load the neural network model.
        """
        args = option_parser.get_args()
        self.model = MoCapSolverModel(args, topology)

        self.model_path = os.path.join('models', 'mocap_solver.pth')
        self.weights_path = os.path.join('models', 'weights.npy')
        self.model.model.load1(path=self.model_path)
        #self.model.load2()
        self.model.setup()
        self.model.model.Marker_encoder.eval() ## add 
        print('model set up over!')
        self.marker_num = 56
        self.joints_nums = 24
        # self.model = _model_class.decoder
        # self.model.load_weights(self.model_path)
        self.jts_statistic_path = os.path.join('models', 'train_data_joint.npy')
        self.joint_data_statistic_data = np.load(self.jts_statistic_path)
        self.marker_statistic_path = os.path.join('models', 'train_marker_data.npy')
        self.marker_statistic_data = np.load(os.path.join( "models", "train_marker_data.npy"), allow_pickle=True)

    def predict(self, sequence):
        """
        Predict the marker positions, joint rotations and positions from neural network.
        Args:
            sequence: The Sequence structure that contains windows_raw_markers.

        Returns:
            sequence: The sequence structure that contains the predicted  marker positions, joint rotations and positions.
                sequence.jts_rot_windows_data: Predicted joint rotations.
                sequence.jts_pos_windows_data: Predicted joint positions.
                sequence.markers_output_windows_data: Predicted marker positions.
        """
        ##########step 1: Get the batch data from the windows data######################################################
        sequence.get_batch_data()
        _markers_batch_data = sequence.input_batch_data
        _batch_num = len(_markers_batch_data)
        _jts_rot_batch_data = list()
        _jts_pos_batch_data = list()
        _markers_output_batch_data = list()
        
        self.model.model.auto_encoder.eval()
        self.model.model.mc_encoder.eval()
        self.model.model.static_encoder.eval()
        self.model.model.Marker_encoder.eval() ## add 
        for _batch_idx in range(_batch_num):
            _input_data = _markers_batch_data[_batch_idx]
            _batch_size = _input_data.shape[0]
            ############### step 2: Scale the input data to normal. ####################################################
            _input_data = (_input_data - self.marker_statistic_data[0]) / self.marker_statistic_data[1]
            _input_data = _input_data.reshape(_batch_size, sequence.window_size, sequence.marker_num, 3)
            _input_data = np.array(_input_data, dtype=np.float32)
            _input_data = torch.Tensor(_input_data)
            ############### step 3: Predict the joint rotations, positions, marker positions using the neural network ##
            with torch.no_grad():
                motion_code0, offsets_code, res_first_rot, marker_config_code = self.model.predict(_input_data)
                #_pred_markers, _pred_jts_rot, _pred_jts_pos = self.model.predict3(motion_code0, offsets_code, res_first_rot, marker_config_code)
                _pred_markers, _pred_jts_rot, _pred_jts_pos, _, _, _, _ = self.model.predict4(motion_code0, offsets_code, res_first_rot, marker_config_code)
            # [_, _pred_jts_rot, _pred_jts_pos, _pred_markers, _, _, _, _] = self.model.predict(_input_data)
            _pred_markers = _pred_markers.cpu().numpy()
            _pred_jts_rot = _pred_jts_rot.cpu().numpy()
            _pred_jts_pos = _pred_jts_pos.cpu().numpy()
            _jts_rot_batch_data.append(_pred_jts_rot)
            _jts_pos_batch_data.append(_pred_jts_pos)
            _markers_output_batch_data.append(_pred_markers)
        ################## step 4: transfer the batch data into windows data ###########################################
        sequence.jts_rot_windows_data = np.concatenate(_jts_rot_batch_data, axis=0)
        sequence.jts_pos_windows_data = np.concatenate(_jts_pos_batch_data, axis=0)
        sequence.markers_output_windows_data = np.concatenate(_markers_output_batch_data, axis=0)
        ################## step 5: scale the output joint position data ################################################
        sequence.jts_pos_windows_data = sequence.jts_pos_windows_data
        return sequence
