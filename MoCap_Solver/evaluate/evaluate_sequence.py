import os
import argparse
import glob
import numpy as np
from MoCap_Solver.utils.Sequence import Sequence
from MoCap_Solver.utils.Normalization import Normalization
from MoCap_Solver.utils.PostProcessing import PostProcessing
from MoCap_Solver.utils.Evaluation import Evaluation
from MoCap_Solver.evaluate.predict import PredictClass
from utils.utils import FRAMENUM, MARKERNUM, JOINTNUM, BATCHSIZE, STEPSIZE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def evaluate_sequence():
    parser = argparse.ArgumentParser(description='mannual to this script')
    parser.add_argument('--GPU_USE', type=bool, default=True)
    args = parser.parse_args()
    GPU_USE = args.GPU_USE

    if GPU_USE == False:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    raw_dir = os.path.join('data', 'test_sample_data')
    raw_window_dir = os.path.join('MoCap_Solver/data/Testing_dataset', 'mocap_dataset')

    print(raw_window_dir)
    result_dir = os.path.join('result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    pd_class = PredictClass()
    
    window_files = glob.glob(os.path.join(raw_window_dir, '*.npz'))
    print('file length: ' + str(len(window_files)))
    
    error_sum = np.array([0, 0, 0])
    error_num = 0

    idx = 0
    for file in window_files[:]:
        _file_window_name = os.path.basename(file).split('.')[0]
        _file_name = _file_window_name[:-2]
        _raw_file_path = os.path.join(raw_dir, _file_name + '.npz')

        h = np.load(_raw_file_path)
        h1 = np.load(file)
        _Markers = h['M1']
        gt_Markers = h['M']
        gt_jt_pos = h['J_t']
        gt_jt_rot = h1['J_R']
        rigidR = h1['RigidR']  #N, 3, 3
        rigidT = h1['RigidT']  #N, 3, 1

        _fps = 120.
        _start_time = 0.
        _Markers = np.array(_Markers)

        # Build a data structure to represent the sequence
        sequence = Sequence(_file_name, _Markers, None, batch_size=BATCHSIZE, window_size=FRAMENUM, fps=_fps, step_size=32)

        sequence.windows_raw_markers = h1['M1'] # window data
        sequence.Rigid_R_inv = [np.linalg.inv(rigidR[i]) for i in range(rigidR.shape[0]) ]
        sequence.Rigid_T_inv = [sequence.Rigid_R_inv[i].dot(-rigidT[i]) for i in range(rigidT.shape[0])]

        sequence = pd_class.predict(sequence)
        pred_jt_rot_res = np.copy(sequence.jts_rot_windows_data)

        normalization = Normalization()
        sequence = normalization.denorm(sequence)

        # Post-processing module: smooth the predicted results of marker positions, joint rotations and positions.
        post_poc = PostProcessing(marker_num=MARKERNUM, joint_num=JOINTNUM)
        sequence = post_poc.post_processing(sequence)

        _eval = Evaluation(marker_num=MARKERNUM)
        marker_error = _eval.evaluate_marker(sequence.pred_marker_data, gt_Markers) #_eval.evaluate(sequence, gt_seq)
        jt_pos_error = _eval.evaluate_pos(sequence.pred_trans_data, gt_jt_pos)
        jt_angle_error = _eval.evaluate_rot(pred_jt_rot_res, gt_jt_rot)
            
        window_num = gt_Markers.shape[0]
        error_sum[0] += marker_error * window_num
        error_sum[1] += jt_angle_error * window_num
        error_sum[2] += jt_pos_error * window_num
        error_num += window_num

        if idx % 10 == 0:
            mean_error = error_sum / error_num 
            print(str(idx) + "/" + str(str(len(window_files))) + ".  current mean error:  marker: %.3f(mm)  angle: %.3f(deg) pos: %.3f(mm)"  % (mean_error[0] * 1000.,mean_error[1] ,mean_error[2] * 1000.))

        idx += 1
    mean_error = error_sum / error_num 
    print("Total mean error:  marker: %.3f(mm) angle: %.3f(deg)  pos: %.3f(mm)"  % (mean_error[0] * 1000.,mean_error[1] ,mean_error[2] * 1000.))
