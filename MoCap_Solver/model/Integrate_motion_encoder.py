import torch
from MoCap_Solver.model.enc_and_dec import AE
from MoCap_Solver.model.MC_Encoder import TS_AE
from MoCap_Solver.model.skeleton import build_edge_topology
from MoCap_Solver.model.Kinematics import ForwardKinematics
import os

class IntegratedModel:
    def __init__(self, args, joint_topology, origin_offsets: torch.Tensor, device, characters):
        self.args = args
        self.joint_topology = joint_topology
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
        self.fk = ForwardKinematics(self.edges)
        self.window_size = args.window_size

        self.auto_encoder = AE(topology=self.edges).to(device)
        self.static_encoder = TS_AE(self.edges).to(device)
        self.static_encoder.load_state_dict(
            torch.load(os.path.join('models', 'ts_encoder.pt'),
                       map_location=self.args.cuda_device))

    def parameters(self):
        return list(self.auto_encoder.parameters())

    def save(self, path):
        torch.save(self.auto_encoder.state_dict(), path)
        print('Save at {} succeed!'.format(path))

    def load(self, path):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')

        self.auto_encoder.load_state_dict(torch.load(os.path.join(path),
                                                     map_location=self.args.cuda_device))
        print('load succeed!')