import numpy as np
import torch
from utils.utils import angle_loss

class Evaluation(object):
    def __init__(self, marker_num=56):
        self.marker_num = marker_num
        self.criteria_angle_loss = angle_loss()

    def evaluate(self, seq1, seq2):
        markers_1 = seq1.pred_marker_data
        markers_2 = seq2.raw_markers
        e_v = markers_1 - markers_2
        _vertex_error = np.average(np.sqrt(np.sum(e_v ** 2, axis=-1)))
        return _vertex_error

    def evaluate_marker(self, pred_marker, gt_marker):
        return np.mean(np.sqrt(np.sum((pred_marker - gt_marker) ** 2, axis=-1)))

    def evaluate_pos(self, pred_pos, gt_pos):
        return np.mean(np.sqrt(np.sum((pred_pos - gt_pos) ** 2, axis=-1)))

    def evaluate_rot(self, pred_rot, gt_rot):
        return self.criteria_angle_loss(torch.tensor(pred_rot, dtype=torch.float32),
                                             torch.tensor(gt_rot, dtype=torch.float32))