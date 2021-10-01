import torch
import torch.nn as nn
from MoCap_Solver.model.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear
from utils.utils import LAYERSNUM, KERNELSIZE, SKELETONDIST, EXTRACONV

class Encoder(nn.Module):
    def __init__(self,  topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
        self.channel_base = [4]
        self.channel_list = []
        self.edge_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.convs = []

        kernel_size = KERNELSIZE
        padding = (kernel_size - 1) // 2
        bias = True
        add_offset = True

        for i in range(LAYERSNUM):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(LAYERSNUM):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], SKELETONDIST)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            for _ in range(EXTRACONV):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode='reflection', bias=bias))
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                    padding=padding, padding_mode='reflection', bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == LAYERSNUM - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode='mean',
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == LAYERSNUM - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)

        for i, layer in enumerate(self.layers):
            if offset is not None:
                self.convs[i].set_offset(offset[i])
            input = layer(input)
        return input


class Decoder(nn.Module):
    def __init__(self,  enc):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.enc = enc
        self.convs = []

        kernel_size = KERNELSIZE
        padding = (kernel_size - 1) // 2

        add_offset = True

        for i in range(LAYERSNUM):
            seq = []
            in_channels = enc.channel_list[LAYERSNUM - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[LAYERSNUM - i - 1], SKELETONDIST)

            if i != 0 and i != LAYERSNUM - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[LAYERSNUM - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
            seq.append(self.unpools[-1])
            for _ in range(EXTRACONV):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=enc.edge_num[LAYERSNUM - i - 1], kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding, padding_mode='reflection', bias=bias))
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[LAYERSNUM - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode='reflection', bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * enc.channel_base[LAYERSNUM - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])
            if i != LAYERSNUM - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
        # throw the padded rwo for global position
        input = input[:, :-1, :]

        return input


class AE(nn.Module):
    def __init__(self, topology):
        super(AE, self).__init__()
        self.enc = Encoder(topology)
        self.dec = Decoder(self.enc)

    def forward(self, input, offset=None):
        latent = self.enc(input, offset)
        result = self.dec(latent, offset)
        return latent, result


# eoncoder for static part, i.e. offset part
class StaticEncoder(nn.Module):
    def __init__(self, edges):
        super(StaticEncoder, self).__init__()
        self.layers = nn.ModuleList()
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3

        for i in range(LAYERSNUM):
            neighbor_list = find_neighbor(edges, SKELETONDIST)
            seq = []
            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))
            if i < LAYERSNUM - 1:
                pool = SkeletonPool(edges, channels_per_edge=channels*2, pooling_mode='mean')
                seq.append(pool)
                edges = pool.new_edges
            seq.append(activation)
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input):
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze())
        return output
