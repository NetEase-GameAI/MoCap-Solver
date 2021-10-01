import torch
import torch.nn as nn
import numpy as np
import math
'''
This script is about the implemention about Forward Kinematics, Inverse Kinematics, and Skinning. This script is borrowed from [Aberman et al. 2020].
'''

class ForwardKinematics:
    def __init__(self,  edges):
        self.topology = [-1] * (len(edges) + 1)
        self.topology1 = self.topology[1:]
        self.rotation_map = []
        for i, edge in enumerate(edges):
            self.topology[edge[1]] = edge[0]
            self.rotation_map.append(edge[1])

        self.world = 0
        self.pos_repr = '3d'
        self.quater = True

    def forward_from_raw(self, raw, offset, world=None, quater=None):
        if world is None: world = self.world
        if quater is None: quater = self.quater
        if self.pos_repr == '3d':
            position = raw[:, -3:, :]
            rotation = raw[:, :-3, :]
        elif self.pos_repr == '4d':
            raise Exception('Not support')
        if quater:
            rotation = rotation.reshape((rotation.shape[0], -1, 4, rotation.shape[-1]))
            identity = torch.tensor((1, 0, 0, 0), dtype=torch.float, device=raw.device)
        else:
            rotation = rotation.reshape((rotation.shape[0], -1, 3, rotation.shape[-1]))
            identity = torch.zeros((3, ), dtype=torch.float, device=raw.device)
        identity = identity.reshape((1, 1, -1, 1))
        new_shape = list(rotation.shape)
        new_shape[1] += 1
        new_shape[2] = 1
        rotation_final = identity.repeat(new_shape)
        for i, j in enumerate(self.rotation_map):
            rotation_final[:, j, :, :] = rotation[:, i, :, :]
        return self.forward(rotation_final, position, offset, world=world, quater=quater)

    def forward_from_raw1(self, raw, offset, world=None, quater=None):
        if world is None: world = self.world
        if quater is None: quater = self.quater
        if self.pos_repr == '3d':
            position = raw[:, -3:, :]
            rotation = raw[:, :-3, :]
        elif self.pos_repr == '4d':
            raise Exception('Not support')
        if quater:
            rotation = rotation.reshape((rotation.shape[0], -1, 4, rotation.shape[-1]))
            identity = torch.tensor((1, 0, 0, 0), dtype=torch.float, device=raw.device)
        else:
            rotation = rotation.reshape((rotation.shape[0], -1, 3, rotation.shape[-1]))
            identity = torch.zeros((3, ), dtype=torch.float, device=raw.device)
        identity = identity.reshape((1, 1, -1, 1))
        new_shape = list(rotation.shape)
        new_shape[1] += 1
        new_shape[2] = 1
        rotation_final = identity.repeat(new_shape)
        for i, j in enumerate(self.rotation_map):
            rotation_final[:, j, :, :] = rotation[:, i, :, :]
        return self.forward1(rotation_final, position, offset, world=world, quater=quater)


    def forward_from_raw2(self, raw, offset, firstrot=None, world=None, quater=None):
        if world is None: world = self.world
        if quater is None: quater = self.quater
        firstrot = firstrot.permute(0,2,1)
        if self.pos_repr == '3d':
            position = raw[:, -3:, :]
            rotation = raw[:, :-3, :]
        elif self.pos_repr == '4d':
            raise Exception('Not support')
        if quater:
            rotation = rotation.reshape((rotation.shape[0], -1, 4, rotation.shape[-1]))
            identity = torch.tensor((1, 0, 0, 0), dtype=torch.float, device=raw.device)
        else:
            rotation = rotation.reshape((rotation.shape[0], -1, 3, rotation.shape[-1]))
            identity = torch.zeros((3, ), dtype=torch.float, device=raw.device)
        identity = identity.reshape((1, 1, -1, 1))
        new_shape = list(rotation.shape)
        new_shape[1] += 1
        new_shape[2] = 1
        rotation_final = identity.repeat(new_shape)
        for i, j in enumerate(self.rotation_map):
            rotation_final[:, j, :, :] = rotation[:, i, :, :]
        rotation_final[:, 0, :, :] = firstrot[:,:,:]
        return self.forward1(rotation_final, position, offset, world=world, quater=quater)

    def forward_from_raw3(self, raw, offset, firstrot=None, world=None, quater=None):
        if world is None: world = self.world
        if quater is None: quater = self.quater
        firstrot = firstrot.permute(0,2,1)
        if self.pos_repr == '3d':
            position = raw[:, -3:, :]
            rotation = raw[:, :-3, :]
        elif self.pos_repr == '4d':
            raise Exception('Not support')
        if quater:
            rotation = rotation.reshape((rotation.shape[0], -1, 4, rotation.shape[-1]))
            identity = torch.tensor((1, 0, 0, 0), dtype=torch.float, device=raw.device)
        else:
            rotation = rotation.reshape((rotation.shape[0], -1, 3, rotation.shape[-1]))
            identity = torch.zeros((3, ), dtype=torch.float, device=raw.device)
        identity = identity.reshape((1, 1, -1, 1))
        new_shape = list(rotation.shape)
        new_shape[1] += 1
        new_shape[2] = 1
        rotation_final = identity.repeat(new_shape)
        for i, j in enumerate(self.rotation_map):
            rotation_final[:, j, :, :] = rotation[:, i, :, :]
        return self.forward3(rotation_final, position, offset, firstrot, world=world, quater=quater)

    '''
    rotation should have shape batch_size * Joint_num * (3/4) * Time
    position should have shape batch_size * 3 * Time
    offset should have shape batch_size * Joint_num * 3
    output have shape batch_size * Time * Joint_num * 3
    '''
    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False, world=True):
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        result = torch.empty(rotation.shape[:-1] + (3, ), device=position.device)


        norm = torch.norm(rotation, dim=-1, keepdim=True)
        #norm[norm < 1e-10] = 1
        rotation = rotation / norm


        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.topology):
            if pi == -1:
                assert i == 0
                continue

            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :], transform[..., i, :, :])
            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze()
            if world: result[..., i, :] += result[..., pi, :]
        return result

    def forward1(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False, world=True):
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        rotation = rotation.contiguous()
        position = position.permute(0, 2, 1)
        result = torch.empty(rotation.shape[:-1] + (3, ), device=position.device)

        # print(rotation.device)

        norm = torch.norm(rotation, dim=-1, keepdim=True)
        # print(norm.device)
        #norm[norm < 1e-10] = 1
        rotation = rotation / norm


        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.topology):
            if pi == -1:
                assert i == 0
                continue

            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :], transform[..., i, :, :])
            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze()
            if world: result[..., i, :] += result[..., pi, :]
        return result, transform

    def forward3(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, first_rot: torch.Tensor, order='xyz', quater=False, world=True):
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        first_rot = first_rot.permute(0, 2, 1)
        rotation = rotation.contiguous()
        position = position.permute(0, 2, 1)
        result = torch.empty(rotation.shape[:-1] + (3, ), device=position.device)

        # print(rotation.device)

        norm = torch.norm(rotation, dim=-1, keepdim=True)
        # print(norm.device)
        #norm[norm < 1e-10] = 1
        rotation = rotation / norm


        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)
        first_rot_transform = self.transform_from_r6d(first_rot)
        transform[:, :, 0, :, :] = first_rot_transform

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.topology):
            if pi == -1:
                assert i == 0
                continue

            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :], transform[..., i, :, :])
            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze()
            if world: result[..., i, :] += result[..., pi, :]
        return result, transform

    def from_local_to_world(self, res: torch.Tensor):
        res = res.clone()
        for i, pi in enumerate(self.topology):
            if pi == 0 or pi == -1:
                continue
            res[..., i, :] += res[..., pi, :]
        return res

    @staticmethod
    def transform_from_euler(rotation, order):
        rotation = rotation / 180 * math.pi
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m

    @staticmethod
    def normalize_vector(v, return_mag=False):
        batch = v.shape[:-1]
        v_mag = torch.sqrt(v.pow(2).sum(axis=-1))  # batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(-1, 64, 1)
        v = v / v_mag
        if (return_mag == True):
            return v, v_mag[..., 0]
        else:
            return v

    @staticmethod
    def cross_product(u, v):
        batch = u.shape[:-1]
        # print (u.shape)
        # print (v.shape)
        i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
        j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
        k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


        out = torch.cat((i.view(-1, 64, 1), j.view(-1, 64, 1), k.view(-1, 64 , 1)), 2)  # batch*3

        return out

    def transform_from_r6d(self, r6d: torch.Tensor):
        x_raw = r6d[..., 0:3]  # batch*3
        y_raw = r6d[..., 3:6]  # batch*3
        x = self.normalize_vector(x_raw)  # batch*3
        z = self.cross_product(x, y_raw)  # batch*3
        z = self.normalize_vector(z)  # batch*3
        y = self.cross_product(z, x)  # batch*3

        batch = x.shape[:-1]

        x = x.view(-1, 64, 3, 1)
        y = y.view(-1, 64, 3, 1)
        z = z.view(-1, 64, 3, 1)
        matrix = torch.cat((x, y, z), 3)  # batch*3*3
        return matrix


class InverseKinematics:
    def __init__(self, rotations: torch.Tensor, positions: torch.Tensor, offset, parents, constrains):
        self.rotations = rotations
        self.rotations.requires_grad_(True)
        self.position = positions
        self.position.requires_grad_(True)

        self.parents = parents
        self.offset = offset
        self.constrains = constrains

        self.optimizer = torch.optim.Adam([self.position, self.rotations], lr=1e-3, betas=(0.9, 0.999))
        self.crit = nn.MSELoss()

    def step(self):
        self.optimizer.zero_grad()
        glb = self.forward(self.rotations, self.position, self.offset, order='', quater=True, world=True)
        loss = self.crit(glb, self.constrains)
        loss.backward()
        self.optimizer.step()
        self.glb = glb
        return loss.item()

    def tloss(self, time):
        return self.crit(self.glb[time, :], self.constrains[time, :])

    def all_loss(self):
        res = [self.tloss(t).detach().numpy() for t in range(self.constrains.shape[0])]
        return np.array(res)

    '''
        rotation should have shape batch_size * Joint_num * (3/4) * Time
        position should have shape batch_size * 3 * Time
        offset should have shape batch_size * Joint_num * 3
        output have shape batch_size * Time * Joint_num * 3
    '''

    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False,
                world=True):
        '''
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        '''
        result = torch.empty(rotation.shape[:-1] + (3,), device=position.device)

        norm = torch.norm(rotation, dim=-1, keepdim=True)
        rotation = rotation / norm

        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.parents):
            if pi == -1:
                assert i == 0
                continue

            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze()
            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :], transform[..., i, :, :])
            if world: result[..., i, :] += result[..., pi, :]
        return result

    @staticmethod
    def transform_from_euler(rotation, order):
        rotation = rotation / 180 * math.pi
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m


class GlobalTransform:
    def __init__(self):
        self.u = 1

    def forward_from_markers(self, raw, global_rot):
        transform = self.transform_from_quaternion(global_rot)
        batch_num = raw.shape[0]
        window_size = raw.shape[1]
        marker_num = raw.shape[-2]
        transform = transform.view(-1, 3, 3)
        raw = raw.view(-1, marker_num, 3, 1)
        for i in range(marker_num):
            raw[:, i, :, :] = torch.matmul(transform, raw[:, i, :, :])
        raw = raw.view(batch_num, window_size, marker_num, -1)
        return raw

    def forward_from_jt_rot(self, raw, global_rot):
        transform = self.transform_from_quaternion(global_rot)
        jt_num = raw.shape[-3]
        batch_num = raw.shape[0]
        windows_size = raw.shape[1]
        transform = transform.view(-1, 3, 3)
        raw = raw.view(-1, jt_num, 3, 3)
        for i in range(jt_num):
            raw[:, i, :, :] = torch.matmul(transform, raw[:, i, :, :])
        raw = raw.view(batch_num, windows_size, jt_num, 3, 3)
        return raw

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m

class Skinning:
    def __init__(self):
        self.u = 1

    def skinning(self, mrk_config, weights, joint_rot, joint_trans):
        batchsize = joint_trans.shape[0]
        windowssize = joint_trans.shape[1]
        marker_num = mrk_config.shape[1]
        jt_num = mrk_config.shape[2]
        mrk_config= mrk_config.view(batchsize, 1, marker_num, jt_num, 3, 1)
        mrk_config = mrk_config.repeat(1, windowssize, 1, 1, 1, 1)
        mrk_list = []
        joint_rot = joint_rot.view(-1, 3, 3)
        joint_trans = joint_trans.view(-1, 3, 1)
        for i in range(marker_num):
            mc = mrk_config[:, :, i, :, :, :].reshape(-1, 3, 1)
            mrk = (torch.matmul(joint_rot, mc) + joint_trans).reshape(batchsize, windowssize, 1, jt_num, 3)
            weight = weights[:,i, :].reshape(-1, 1, 1, jt_num, 1)
            mrk = mrk * weight
            mrk = torch.sum(mrk, dim=3)
            mrk_list.append(mrk)
        mrks = torch.cat(mrk_list, dim=2)
        return mrks
