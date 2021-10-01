from MoCap_Solver.model.MC_Encoder import MC_AE, TS_AE
import os
import torch
from MoCap_Solver.model.skeleton import build_edge_topology

class IntegratedModel:
    """
    IntegratedModel: This module is the integration of marker configuration encoder.
    """
    def __init__(self, args, joint_topology,  device):
        self.args = args
        self.joint_topology = joint_topology
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
        self.auto_encoder = MC_AE(self.edges).to(device)
        self.ts_auto_encoder = TS_AE(self.edges).to(device)

    def parameters(self):
        return list(self.auto_encoder.parameters())

    def save(self, path):
        torch.save(self.auto_encoder.state_dict(), os.path.join(path))
        print('Save at {} succeed!'.format(path))

    def load(self, path):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')
        self.auto_encoder.load_state_dict(torch.load(os.path.join(path),
                                                     map_location=self.args.cuda_device))
        print('load succeed!')

    def template_skeleton_load(self, path):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')
        self.ts_auto_encoder.load_state_dict(torch.load(os.path.join(path),
                                                        map_location=self.args.cuda_device))
        print('load succeed!')