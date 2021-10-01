import torch
from torch.nn import Linear, BatchNorm1d, LeakyReLU, Flatten
import torch.nn as nn
from MoCap_Solver.model.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear
import numpy as np
from utils.utils import _dict_marker, _dict_joints, LAYERSNUM, SKELETONDIST, MARKERNUM, JOINTNUM, FRAMENUM
gama_choose = True
compact_mode = '32'

class TS_enc(nn.Module):
    def __init__(self,  edges):
        super(TS_enc, self).__init__()
        self.channel_list = []
        self.topologies = [edges]
        self.edge_num = [len(edges) + 1]
        self.layers = nn.ModuleList()
        self.pooling_list = []
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3

        for i in range(LAYERSNUM):
            neighbor_list = find_neighbor(edges, SKELETONDIST)
            seq = []
            if i == 0: self.channel_list.append(channels * len(neighbor_list))

            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))
            if i < LAYERSNUM - 1:
                pool = SkeletonPool(edges, channels_per_edge=channels * 2, pooling_mode='mean')
                seq.append(pool)
                edges = pool.new_edges
                self.pooling_list.append(pool.pooling_list)
                self.topologies.append(pool.new_edges)
                self.edge_num.append(len(self.topologies[-1]) + 1)
            seq.append(activation)
            self.channel_list.append(channels * 2 * len(neighbor_list))
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input):
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze())
        return output


class TS_dec(nn.Module):
    def __init__(self,  enc):
        super(TS_dec, self).__init__()
        self.layers = nn.ModuleList()
        self.enc = enc
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3
        self.channel_list = []
        self.mclayers = nn.ModuleList()
        self.mcnormLayers = nn.ModuleList()
        self.mcactLayers = nn.ModuleList()
        for i in range(LAYERSNUM):
            in_channels = enc.channel_list[LAYERSNUM - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[LAYERSNUM - i - 1], SKELETONDIST)
            seq = []

            if i > 0:
                unpool = SkeletonUnpool(enc.pooling_list[LAYERSNUM - i - 1], in_channels // len(neighbor_list))

                seq.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
                in_channels = in_channels * 2
                seq.append(unpool)
                out_channels = enc.channel_list[LAYERSNUM - i - 1]

            seq.append(SkeletonLinear(neighbor_list, in_channels=in_channels,
                                      out_channels=out_channels, extra_dim1=True))
            # if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input):
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze())
        return output


class TS_AE(nn.Module):
    def __init__(self,  edges):
        super(TS_AE, self).__init__()
        self.enc = TS_enc(edges)
        self.dec = TS_dec(self.enc)

    def reparamize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        latent = self.enc(x)[-1]
        return latent

    def decode(self, z):
        result = self.dec(z)[-1]
        return result

    def decode1(self, z):
        result = self.dec(z)
        return [result[2], result[1], result[0]]

    def forward(self, input):
        z = self.encode(input)
        result = self.decode(z)
        return result, z


class MC_enc(nn.Module):
    def __init__(self):
        super(MC_enc, self).__init__()
        self.layers = nn.ModuleList()
        self.normLayers = nn.ModuleList()
        self.actLayers = nn.ModuleList()
        self.prevLayer0 = Flatten()
        self.prevLayer = nn.Sequential(
            Linear(MARKERNUM * JOINTNUM * 3 + JOINTNUM * 3, 1024))
        self.prevLayer1 = nn.Sequential(
            Linear(1024 + 168, 1024))
        self.layers.append(Linear(1024, 1024))
        self.normLayers.append(BatchNorm1d(1024))
        self.actLayers.append(LeakyReLU(0.5))
        self.layers.append(Linear(1024, 1024))
        self.normLayers.append(BatchNorm1d(1024))
        self.actLayers.append(LeakyReLU(0.5))


    def forward(self, input, TS_input, TS_latent):
        input3 = self.prevLayer0(TS_latent)
        input0 = self.prevLayer0(input)
        input1 = self.prevLayer0(TS_input)
        input2 = torch.cat([input0, input1], dim=1)
        input2 = self.prevLayer(input2)
        x0 = self.normLayers[0](input2)
        x0 = self.actLayers[0](x0)
        x = self.layers[0](x0)
        m0 = x + x0

        m0 = torch.cat([m0, input3], dim=1)
        m0 = self.prevLayer1(m0)

        x1 = self.normLayers[1](m0)
        x1 = self.actLayers[1](x1)
        x = self.layers[1](x1)
        m1 = x + x1

        return m1


class MC_dec(nn.Module):
    def __init__(self):
        super(MC_dec, self).__init__()
        # self.enc = mc_enc
        self.configlayers = nn.ModuleList()
        self.confignormLayers = nn.ModuleList()
        self.configactLayers = nn.ModuleList()
        self.prevLayer0 = Flatten()
        self.prevLayer1 = nn.Sequential(
            Linear(1024 + 168, 1024))
        self.prevLayer2 = nn.Sequential(
            Linear(1024 + 24 * 3, 1024))

        self.configlayers.append(Linear(1024, 1024))
        self.confignormLayers.append(BatchNorm1d(1024))
        self.configactLayers.append(LeakyReLU(0.5))

        self.configlayers.append(Linear(1024, 1024))
        self.confignormLayers.append(BatchNorm1d(1024))
        self.configactLayers.append(LeakyReLU(0.5))

        self.configactLayers.append(LeakyReLU(0.5))
        self.configlayers.append(Linear(1024, MARKERNUM * JOINTNUM * 3))

        # self.configSoftmax = nn.Softmax(dim=2)

    def forward(self, input, TS_input, TS_latent):
        TS_input = self.prevLayer0(TS_input)
        TS_latent = self.prevLayer0(TS_latent)
        input0 = torch.cat([input, TS_latent], dim=1)
        input1 = self.prevLayer1(input0)
        x_vector = self.confignormLayers[0](input1)
        x_vector = self.configactLayers[0](x_vector)
        x_vector1 = self.configlayers[0](x_vector)
        x_vector = x_vector + x_vector1

        x_vector = torch.cat([x_vector, TS_input], dim=1)
        x_vector = self.prevLayer2(x_vector)

        x_vector = self.confignormLayers[1](x_vector)
        x_vector = self.configactLayers[1](x_vector)
        x_vector1 = self.configlayers[1](x_vector)
        x_vector = x_vector + x_vector1

        x_vector = self.configactLayers[2](x_vector)
        mrk_config_x_o = self.configlayers[2](x_vector)
        mrk_config_x_o = mrk_config_x_o.reshape([mrk_config_x_o.shape[0], MARKERNUM, JOINTNUM, 3])
        return mrk_config_x_o


class MC_AE(nn.Module):
    def __init__(self, edges):
        super(MC_AE, self).__init__()
        self.enc = MC_enc()
        self.dec = MC_dec()

    def reparamize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, TS_input, TS_latent):
        latent = self.enc(x, TS_input, TS_latent)
        return latent

    def decode(self, z, TS_input, TS_latent):
        latent = z
        result = self.dec(latent, TS_input, TS_latent)
        return result

    def forward(self, input, TS_input, TS_latent):
        z = self.encode(input, TS_input, TS_latent)
        result = self.decode(z, TS_input, TS_latent)
        return result, z


class MarkerEncoder(nn.Module):
    def __init__(self):
        super(MarkerEncoder, self).__init__()
        self.gamma_para = 0.001
        self.HUBER_DELTA = 200
        self.lambda_para = np.ones((MARKERNUM, 3))
        self.lambda_jt_para = np.ones((JOINTNUM, 3))
        self.head_idx = [_dict_marker["ARIEL"], _dict_marker["LBHD"], _dict_marker["LFHD"], _dict_marker["RFHD"],
                    _dict_marker["RBHD"]]
        self.head_jt_idx = [_dict_joints["Head"]]
        self.shoulder_idx = [_dict_marker["LTSH"], _dict_marker["LBSH"], _dict_marker["LFSH"], _dict_marker["RTSH"],
                        _dict_marker["RFSH"], _dict_marker["RBSH"]]
        self.shoulder_jt_idx = [_dict_joints['L_Collar'], _dict_joints['R_Collar'], _dict_joints['L_Shoulder'], _dict_joints['R_Shoulder']]
        self.arm_idx = [
            _dict_marker["LUPA"], _dict_marker["LIEL"], _dict_marker["LELB"], _dict_marker["LWRE"],
            _dict_marker["RUPA"], _dict_marker["RIEL"], _dict_marker["RELB"], _dict_marker["RWRE"]
        ]
        self.arm_jt_idx = [_dict_joints['L_Elbow'], _dict_joints['R_Elbow']]
        self.wrist_hand_idx = [
            _dict_marker["LOWR"], _dict_marker["LIWR"], _dict_marker["LIHAND"], _dict_marker["LOHAND"],
            _dict_marker["ROWR"], _dict_marker["RIWR"], _dict_marker["RIHAND"], _dict_marker["ROHAND"]
        ]
        self.wrist_hand_jt_idx = [_dict_joints['L_Wrist'], _dict_joints['R_Wrist'], _dict_joints['L_Hand'], _dict_joints['R_Hand']]

        self.torso_idx = [
            _dict_marker["CLAV"], _dict_marker["STRN"], _dict_marker["C7"], _dict_marker["T10"], _dict_marker["L4"],
            _dict_marker["LMWT"], _dict_marker["LFWT"], _dict_marker["LBWT"],
            _dict_marker["RMWT"], _dict_marker["RFWT"], _dict_marker["RBWT"]
        ]

        self.torso_jt_idx = [_dict_joints['Pelvis'],_dict_joints['Spine1'], _dict_joints['Spine2'], _dict_joints['Spine3'], _dict_joints['L_Hip'], _dict_joints['R_Hip'], _dict_joints['Neck']]
        self.thigh_idx = [
            _dict_marker["LKNI"], _dict_marker["LKNE"], _dict_marker["LHIP"], _dict_marker["LSHN"],
            _dict_marker["RKNI"], _dict_marker["RKNE"], _dict_marker["RHIP"], _dict_marker["RSHN"]
        ]
        self.thigh_jt_idx = [_dict_joints['L_Knee'], _dict_joints['R_Knee']]

        self.foots_idx = [
            _dict_marker["LANK"], _dict_marker["LHEL"], _dict_marker["LMT1"], _dict_marker["LTOE"],
            _dict_marker["LMT5"],
            _dict_marker["RANK"], _dict_marker["RHEL"], _dict_marker["RMT1"], _dict_marker["RTOE"], _dict_marker["RMT5"]
        ]
        self.foots_jt_idx = [_dict_joints['L_Ankle'], _dict_joints['R_Ankle'], _dict_joints['L_Foot'], _dict_joints['R_Foot']]

        self.lambda_para[[self.head_idx]] = self.lambda_para[[self.head_idx]] * 10  # head
        self.lambda_para[[self.shoulder_idx]] = self.lambda_para[[self.shoulder_idx]] * 5  # shoulder
        self.lambda_para[[self.arm_idx]] = self.lambda_para[[self.arm_idx]] * 8  # arm
        self.lambda_para[[self.wrist_hand_idx]] = self.lambda_para[[self.wrist_hand_idx]] * 10  # wrist
        self.lambda_para[[self.torso_idx]] = self.lambda_para[[self.torso_idx]] * 5  # torso
        self.lambda_para[[self.thigh_idx]] = self.lambda_para[[self.thigh_idx]] * 8  # thigh
        self.lambda_para[[self.foots_idx]] = self.lambda_para[[self.foots_idx]] * 10  # foots

        self.lambda_jt_para[[self.head_jt_idx]] = self.lambda_jt_para[[self.head_jt_idx]] * 10  # head
        self.lambda_jt_para[[self.shoulder_jt_idx]] = self.lambda_jt_para[[self.shoulder_jt_idx]] * 5  # shoulder
        self.lambda_jt_para[[self.arm_jt_idx]] = self.lambda_jt_para[[self.arm_jt_idx]] * 8  # arm
        self.lambda_jt_para[[self.wrist_hand_jt_idx]] = self.lambda_jt_para[[self.wrist_hand_jt_idx]] * 10  # wrist
        self.lambda_jt_para[[self.torso_jt_idx]] = self.lambda_jt_para[[self.torso_jt_idx]] * 5  # torso
        self.lambda_jt_para[[self.thigh_jt_idx]] = self.lambda_jt_para[[self.thigh_jt_idx]] * 8  # thigh
        self.lambda_jt_para[[self.foots_jt_idx]] = self.lambda_jt_para[[self.foots_jt_idx]] * 10  # foots
        self.joints_nums = JOINTNUM
        self.marker_nums = MARKERNUM
        self.layers = nn.ModuleList()
        self.normLayers = nn.ModuleList()
        self.actLayers = nn.ModuleList()
        self.prevLayer = nn.Sequential(Flatten(),
                                       Linear(FRAMENUM * MARKERNUM * 3, 2048))
        self.layers.append(Linear(2048, 2048))
        self.normLayers.append(BatchNorm1d(2048))
        self.actLayers.append(LeakyReLU(0.5))
        self.layers.append(Linear(2048, 2048))
        self.normLayers.append(BatchNorm1d(2048))
        self.actLayers.append(LeakyReLU(0.5))
        self.layers.append(Linear(2048, 2048))
        self.normLayers.append(BatchNorm1d(2048))
        self.actLayers.append(LeakyReLU(0.5))

        self.actLayers.append(LeakyReLU(0.5))
        self.layers.append(Linear(2048, 2048))
        self.actLayers.append(LeakyReLU(0.5))

    def forward(self, input):
        input1 = self.prevLayer(input)
        x0 = self.normLayers[0](input1)
        x0 = self.actLayers[0](x0)
        x = self.layers[0](x0)
        m0 = x + x0
        x1 = self.normLayers[1](m0)
        x1 = self.actLayers[1](x1)
        x = self.layers[1](x1)
        m1 = x + x1
        x2 = self.normLayers[2](m1)
        x2 = self.actLayers[2](x2)
        x = self.layers[2](x2)
        m2 = x + x2
        x3 = self.normLayers[2](m2)
        x3 = self.actLayers[3](x3)
        x3 = self.layers[3](x3)
        x = self.actLayers[4](x3)
        return x


class MarkerDecoder(nn.Module):
    def __init__(self):
        super(MarkerDecoder, self).__init__()
        self.configlayers = nn.ModuleList()
        self.confignormLayers = nn.ModuleList()
        self.configactLayers = nn.ModuleList()
        self.mdlayers = nn.ModuleList()
        self.mdnormLayers = nn.ModuleList()
        self.mdactLayers = nn.ModuleList()
        self.offsetlayers = nn.ModuleList()
        self.offsetnormLayers = nn.ModuleList()
        self.offsetactLayers = nn.ModuleList()
        self.transLayers = nn.ModuleList()
        self.transactLayers = nn.ModuleList()

        self.configlayers.append(Linear(2048, 2048))
        self.confignormLayers.append(BatchNorm1d(2048))
        self.configactLayers.append(LeakyReLU(0.5))

        self.configlayers.append(Linear(2048, 2048))
        self.confignormLayers.append(BatchNorm1d(2048))
        self.configactLayers.append(LeakyReLU(0.5))

        self.configactLayers.append(LeakyReLU(0.5))
        self.configlayers.append(Linear(2048, 1024))

        self.mdlayers.append(Linear(2048, 2048))
        self.mdnormLayers.append(BatchNorm1d(2048))
        self.mdactLayers.append(LeakyReLU(0.5))

        self.mdlayers.append(Linear(2048, 2048))
        self.mdnormLayers.append(BatchNorm1d(2048))
        self.mdactLayers.append(LeakyReLU(0.5))



        self.mdlayers.append(Linear(2048, 2048))
        self.mdnormLayers.append(BatchNorm1d(2048))
        self.mdactLayers.append(LeakyReLU(0.5))

        self.mdactLayers.append(LeakyReLU(0.5))
        self.mdlayers.append(Linear(2048, 2048))


        self.offsetlayers.append(Linear(2048, 2048))
        self.offsetnormLayers.append(BatchNorm1d(2048))
        self.offsetactLayers.append(LeakyReLU(0.5))

        self.offsetlayers.append(Linear(2048, 2048))
        self.offsetnormLayers.append(BatchNorm1d(2048))
        self.offsetactLayers.append(LeakyReLU(0.5))

        self.offsetlayers.append(Linear(2048, 2048))
        self.offsetnormLayers.append(BatchNorm1d(2048))
        self.offsetactLayers.append(LeakyReLU(0.5))

        self.offsetactLayers.append(LeakyReLU(0.5))
        self.offsetlayers.append(Linear(2048, 168))


    def forward(self, input):
        ######marker config#######################
        x_vector = self.confignormLayers[0](input)
        x_vector = self.configactLayers[0](x_vector)
        x_vector1 = self.configlayers[0](x_vector)
        x_vector = x_vector + x_vector1

        x_vector = self.confignormLayers[1](x_vector)
        x_vector = self.configactLayers[1](x_vector)
        x_vector1 = self.configlayers[1](x_vector)
        x_vector = x_vector + x_vector1

        x_vector = self.configactLayers[2](x_vector)
        mrk_config_x_o = self.configlayers[2](x_vector)
        mrk_config_x_o = mrk_config_x_o.reshape([mrk_config_x_o.shape[0], 1024])

        ######motion data#############################
        md_vector = self.mdnormLayers[0](input)
        md_vector = self.mdactLayers[0](md_vector)
        md_vector1 = self.mdlayers[0](md_vector)
        md_vector = md_vector + md_vector1

        md_vector = self.mdnormLayers[1](md_vector)
        md_vector = self.mdactLayers[1](md_vector)
        md_vector1 = self.mdlayers[1](md_vector)
        md_vector = md_vector + md_vector1

        md_vector = self.mdnormLayers[2](md_vector)
        md_vector = self.mdactLayers[2](md_vector)
        md_vector1 = self.mdlayers[2](md_vector)
        md_vector = md_vector + md_vector1

        md_vector5 = self.mdactLayers[3](md_vector)
        md_x_o = self.mdlayers[3](md_vector5)
        md_x_o = md_x_o.reshape([md_x_o.shape[0], 2048])


        ######offset data#############################
        offset_vector = self.offsetnormLayers[0](input)
        offset_vector = self.offsetactLayers[0](offset_vector)
        offset_vector1 = self.offsetlayers[0](offset_vector)
        offset_vector = offset_vector + offset_vector1

        offset_vector = self.offsetnormLayers[1](offset_vector)
        offset_vector = self.offsetactLayers[1](offset_vector)
        offset_vector1 = self.offsetlayers[1](offset_vector)
        offset_vector = offset_vector + offset_vector1

        offset_vector = self.offsetnormLayers[2](offset_vector)
        offset_vector = self.offsetactLayers[2](offset_vector)
        offset_vector1 = self.offsetlayers[2](offset_vector)
        offset_vector = offset_vector + offset_vector1

        offset_vector = self.offsetactLayers[3](offset_vector)
        offset_x_o = self.offsetlayers[3](offset_vector)
        offset_x_o = offset_x_o.reshape([offset_x_o.shape[0], 168])

        return mrk_config_x_o, md_x_o, offset_x_o


class MarkerAE(nn.Module):
    def __init__(self):
        super(MarkerAE, self).__init__()
        self.enc = MarkerEncoder()
        self.dec = MarkerDecoder()

    def forward(self, input):
        latent = self.enc(input)
        mrkconfig, motionlatent, offset = self.dec(latent)
        return latent, mrkconfig, motionlatent, offset
