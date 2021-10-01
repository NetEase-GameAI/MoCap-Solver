import torch
from MoCap_Solver.model.enc_and_dec import AE, StaticEncoder
from MoCap_Solver.model.MC_Encoder import TS_AE, MarkerAE, MC_AE
from MoCap_Solver.model.skeleton import build_edge_topology
from MoCap_Solver.model.Kinematics import ForwardKinematics, GlobalTransform, Skinning
from utils.utils import CUDADIVICE
import os



class IntegratedModel:
    """
    IntegratedModel: This module is the integration of MoCap Solver.
    """
    def __init__(self,  joint_topology, origin_offsets: torch.Tensor, device):
        self.joint_topology = joint_topology
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
        self.fk = ForwardKinematics(self.edges)
        self.gt = GlobalTransform()
        self.skin = Skinning()

        self.auto_encoder = AE(topology=self.edges).to(device)
        self.auto_encoder.load_state_dict(
            torch.load(os.path.join('models', 'motion_encoder.pt'),
                       map_location=CUDADIVICE))
        self.Marker_encoder = MarkerAE().to(device)
        self.mc_encoder = MC_AE(self.edges).to(device)
        self.mc_encoder.load_state_dict(
            torch.load(os.path.join('models', 'mc_encoder.pt'),
                       map_location=CUDADIVICE))
        self.static_encoder = TS_AE(self.edges).to(device)
        self.static_encoder.load_state_dict(
            torch.load(os.path.join('models', 'ts_encoder.pt'),
                       map_location=CUDADIVICE))

    def parameters(self):
        return list(self.Marker_encoder.parameters())

    def save(self, path, epoch):
        from utils.option_parser import try_mkdir
        try_mkdir(path)
        path = os.path.join(path, 'MoCap_Solver_'+str(epoch)+'.pt')
        torch.save(self.Marker_encoder.state_dict(), path)
        print('Save at {} succeed!'.format(path))

    def save1(self, path):
        torch.save(self.Marker_encoder.state_dict(), path)

    def load1(self, path):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')
        self.Marker_encoder.load_state_dict(torch.load(os.path.join(path),
                                                     map_location=CUDADIVICE))
        print('load succeed!')

    def load(self, path, epoch):
        print('loading from', path)
        path = os.path.join(path, 'MoCap_Solver_' + str(epoch) + '.pt')
        if not os.path.exists(path):
            raise Exception('Unknown loading path')
        self.Marker_encoder.load_state_dict(torch.load(os.path.join(path),
                                                     map_location=CUDADIVICE))
        print('load succeed!')