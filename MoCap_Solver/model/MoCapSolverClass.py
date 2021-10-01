from MoCap_Solver.model.Integrate_mocap_solver import IntegratedModel as Integrate_mocap_solver
from torch import optim
import torch
from utils.utils import Criterion_EE, Eval_Criterion, vertex_loss, angle_loss, HuberLoss
from MoCap_Solver.model.base_model import BaseModel
import torch.nn.functional as F
import os
import numpy as np
from utils.utils import _dict_joints, _dict_marker, FRAMENUM, MARKERNUM, JOINTNUM

class MoCapSolverModel(BaseModel):
    """
    This class is the module of MoCap Solver.
    """
    def __init__(self, args, topology):
        super(MoCapSolverModel, self).__init__(args)
        self.joint_topology = topology
        self.n_topology = 1
        self.models = []
        self.para = []
        self.motions_input = []
        self.args = args
        self.vertex_criteria = vertex_loss()
        self.train_marker_data = np.load(os.path.join('models', 'train_marker_data.npy'))
        self.train_marker_data = torch.tensor(self.train_marker_data, dtype=torch.float32).to(self.device)
        self.train_motion_code_data = np.load(os.path.join('models', 'train_motion_code_data.npy'))
        self.train_motion_code_data = torch.tensor(self.train_motion_code_data, dtype=torch.float32).to(self.device)
        self.train_offset_code_data = np.load(os.path.join('models', 'train_offset_code_data.npy'))
        self.train_offset_code_data = torch.tensor(self.train_offset_code_data, dtype=torch.float32).to(self.device)
        self.train_mc_code_data = np.load(os.path.join('models', 'train_mc_code_data.npy'))
        self.train_mc_code_data = torch.tensor(self.train_mc_code_data, dtype=torch.float32).to(self.device)
        self.train_offset_data = np.load(os.path.join('models', 'train_data_ts.npy'))
        self.train_offset_data = torch.tensor(self.train_offset_data, dtype=torch.float32).to(self.device)
        self.train_motion_data = np.load(os.path.join('models', 'train_motion_data.npy'))
        self.train_motion_data = torch.tensor(self.train_motion_data, dtype=torch.float32).to(self.device)
        self.train_mc_data = np.load(os.path.join('models', 'train_data_marker_config.npy'))
        self.train_mc_data = torch.tensor(self.train_mc_data, dtype=torch.float32).to(self.device)
        self.train_first_rot_data = np.load(os.path.join('models', 'train_first_rot_data.npy'))
        self.train_first_rot_data = torch.tensor(self.train_first_rot_data, dtype=torch.float32).to(self.device)
        self.weights = np.load(os.path.join('models', 'weights.npy'))
        self.weights = torch.tensor(self.weights, dtype=torch.float32).to(self.device)

        model = Integrate_mocap_solver(self.joint_topology, None, self.device)
        self.model = model
        self.models.append(model)
        self.para = model.parameters()

        if self.is_train:
            self.optimizer = optim.Adam(self.para, args.learning_rate)
            self.optimizers = [self.optimizer]
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
            self.schedulerss = [self.scheduler]
            self.criterion_rec = torch.nn.MSELoss()
            self.criterion_smooth_l1_skeleton_pos_loss_0 = HuberLoss(100)
            self.criterion_smooth_l1_first_rot_error_average_0 = HuberLoss(100)
            self.criterion_smooth_l1_marker_configuration_loss_0 = HuberLoss(100)
            self.criterion_smooth_l1_marker_pos_loss_0 = HuberLoss(100)
            self.criterion_smooth_l1_offsets_loss_0 = HuberLoss(100)
            self.criterion_smooth_l1_first_rot_transform_loss_0 = HuberLoss(100)
            self.criterion_smooth_l1_rec_loss_motion_0 = HuberLoss(100)
            self.criterion_smooth_l1_rec_motion_transform_loss_0 = HuberLoss(100)
            self.criterion_smooth_l1_res_mc_code_error_average_0 = HuberLoss(100)
            self.criterion_smooth_l1_res_offset_code_error_average_0 = HuberLoss(100)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_ee = Criterion_EE(args, torch.nn.MSELoss())
            self.criter_angle_loss = angle_loss()
        else:
            self.err_crit = []
            self.err_crit.append(Eval_Criterion(self.joint_topology))
        self.marker_num = MARKERNUM
        self.joint_num = JOINTNUM
        self.lambda_para = self.models[0].Marker_encoder.enc.lambda_para
        self.lambda_para_jt = self.models[0].Marker_encoder.enc.lambda_jt_para
        self.lambda_para = self.lambda_para.reshape(1, 1, self.marker_num, 3)
        self.lambda_para_jt = self.lambda_para_jt.reshape(1, 1, self.joint_num, 3)
        self.lambda_para_jt1 = self.lambda_para_jt.reshape(1, 1, self.joint_num, 3, 1)
        self.lambda_para = torch.tensor(self.lambda_para, dtype=torch.float32).to(self.device)
        self.lambda_para_jt = torch.tensor(self.lambda_para_jt, dtype=torch.float32).to(self.device)
        self.lambda_para_jt1 = torch.tensor(self.lambda_para_jt1, dtype=torch.float32).to(self.device)

    def set_input(self, motions):
        self.motions_input = motions
        return

    def predict1(self, motion, offset, first_rot, mrkconfig):
        motion = motion.to(self.device)
        offset = offset.to(self.device)
        first_rot = first_rot.to(self.device)
        mrkconfig = mrkconfig.to(self.device)
        motion_input = (motion - self.train_motion_data[0]) / self.train_motion_data[1]
        offset_input = (offset - self.train_offset_data[0]) / self.train_offset_data[1]
        offsetlatentcode = self.models[0].static_encoder.encode(offset_input)
        offset_output = self.models[0].static_encoder.decode1(offsetlatentcode)
        offset_output[0] = offset_output[0].view(-1, 24, 3)
        offset_output[0] = offset_output[0] * self.train_offset_data[1] + self.train_offset_data[0]
        res_offset_output_input = (offset_output[0] - self.train_offset_data[0]) / self.train_offset_data[1]
        marker_config_input = (mrkconfig - self.train_mc_data[0]) / self.train_mc_data[1]
        marker_config_all, _ = self.models[0].mc_encoder(marker_config_input, res_offset_output_input, offsetlatentcode)
        res_marker_config_all = marker_config_all * self.train_mc_data[1] + self.train_mc_data[0]
        res_marker_config = res_marker_config_all[:, :, :, :3]
        skinning_weights = self.weights.view(1, MARKERNUM, JOINTNUM)
        _, res_motion_out = self.models[0].auto_encoder(motion_input, offset_output)
        res_motion_out = res_motion_out * self.train_motion_data[1] + self.train_motion_data[0]
        res_skeleton_pos, res_transform = self.models[0].fk.forward_from_raw2(res_motion_out, offset_output[0],
                                                                              first_rot, world=True,
                                                                              quater=True)
        res_markers = self.models[0].skin.skinning(res_marker_config, skinning_weights, res_transform, res_skeleton_pos)
        return res_markers, res_transform, res_skeleton_pos


    def predict4(self, motioncode, offsetcode, first_rot, mrkconfigcode):
        motioncode = motioncode.to(self.device)
        offsetcode = offsetcode.to(self.device)
        first_rot = first_rot.to(self.device)
        mrkconfigcode = mrkconfigcode.to(self.device)

        offset_output = self.models[0].static_encoder.decode1(offsetcode)
        offset_output[0] = offset_output[0].view(-1, 24, 3)
        offset_output[0] = offset_output[0] * self.train_offset_data[1] + self.train_offset_data[0]
        res_offset_output_input = (offset_output[0] - self.train_offset_data[0]) / self.train_offset_data[1]

        marker_config_all = self.models[0].mc_encoder.dec(mrkconfigcode, res_offset_output_input, offsetcode)
        res_marker_config_all = marker_config_all * self.train_mc_data[1] + self.train_mc_data[0]
        res_marker_config = res_marker_config_all[:, :, :, :3]
        skinning_weights = self.weights.view(1, MARKERNUM, JOINTNUM)
        motioncode = motioncode.view(-1, 112, 16)

        res_motion_out = self.models[0].auto_encoder.dec(motioncode, offset_output)
        res_motion_out = res_motion_out * self.train_motion_data[1] + self.train_motion_data[0]

        res_skeleton_pos, res_transform = self.models[0].fk.forward_from_raw2(res_motion_out, offset_output[0],
                                                                              first_rot, world=True,
                                                                              quater=True)
        res_markers = self.models[0].skin.skinning(res_marker_config, skinning_weights, res_transform, res_skeleton_pos)
        return res_markers, res_transform, res_skeleton_pos, res_motion_out, offset_output[
            0], res_marker_config, skinning_weights

    def predict3(self, motioncode, offsetcode, first_rot, mrkconfigcode):
        motioncode = motioncode.to(self.device)
        offsetcode = offsetcode.to(self.device)
        first_rot = first_rot.to(self.device)
        mrkconfigcode = mrkconfigcode.to(self.device)

        offset_output = self.models[0].static_encoder.decode1(offsetcode)
        offset_output[0] = offset_output[0].view(-1, 24, 3)
        offset_output[0] = offset_output[0] * self.train_offset_data[1] + self.train_offset_data[0]
        res_offset_output_input = (offset_output[0] - self.train_offset_data[0]) / self.train_offset_data[1]

        marker_config_all = self.models[0].mc_encoder.dec(mrkconfigcode, res_offset_output_input, offsetcode)
        res_marker_config_all = marker_config_all * self.train_mc_data[1] + self.train_mc_data[0]
        res_marker_config = res_marker_config_all[:, :, :, :3]
        skinning_weights = self.weights.view(1, MARKERNUM, JOINTNUM)
        motioncode = motioncode.view(-1, 112, 16)

        res_motion_out = self.models[0].auto_encoder.dec(motioncode, offset_output)
        res_motion_out = res_motion_out * self.train_motion_data[1] + self.train_motion_data[0]

        res_skeleton_pos, res_transform = self.models[0].fk.forward_from_raw2(res_motion_out, offset_output[0],
                                                                              first_rot, world=True,
                                                                              quater=True)
        res_markers = self.models[0].skin.skinning(res_marker_config, skinning_weights, res_transform, res_skeleton_pos)
        return res_markers, res_transform, res_skeleton_pos

    def predict2(self, motion, offset, first_rot, mrkconfig):
        motion = motion.to(self.device)
        offset = offset.to(self.device)
        first_rot = first_rot.to(self.device)
        mrkconfig = mrkconfig.to(self.device)
        motion_input = (motion - self.train_motion_data[0]) / self.train_motion_data[1]
        offset_input = (offset - self.train_offset_data[0]) / self.train_offset_data[1]
        offsetlatentcode = self.models[0].static_encoder.encode(offset_input)
        offset_output = self.models[0].static_encoder.decode1(offsetlatentcode)
        offset_output[0] = offset_output[0].view(-1, 24, 3)
        offset_output[0] = offset_output[0] * self.train_offset_data[1] + self.train_offset_data[0]
        res_offset_output_input = (offset_output[0] - self.train_offset_data[0]) / self.train_offset_data[1]
        marker_config_input = (mrkconfig - self.train_mc_data[0]) / self.train_mc_data[1]
        marker_config_all, marker_config_all_latent = self.models[0].mc_encoder(marker_config_input,
                                                                                res_offset_output_input,
                                                                                offsetlatentcode)
        res_marker_config_all = marker_config_all * self.train_mc_data[1] + self.train_mc_data[0]
        res_marker_config = res_marker_config_all[:, :, :, :3]
        skinning_weights = self.weights.view(1, MARKERNUM, JOINTNUM)

        res_motion_latent, res_motion_out = self.models[0].auto_encoder(motion_input, offset_output)
        res_motion_out = res_motion_out * self.train_motion_data[1] + self.train_motion_data[0]

        res_skeleton_pos, res_transform = self.models[0].fk.forward_from_raw2(res_motion_out, offset_output[0],
                                                                              first_rot, world=True,
                                                                              quater=True)
        res_markers = self.models[0].skin.skinning(res_marker_config, skinning_weights, res_transform, res_skeleton_pos)
        return offsetlatentcode, marker_config_all_latent, res_motion_latent, res_markers, res_transform, res_skeleton_pos

    def predict1(self, motioncode, offsetcode, first_rot, mrkconfigcode):
        motioncode = motioncode.to(self.device)
        offsetlatentcode = offsetcode.to(self.device)
        first_rot = first_rot.to(self.device)
        mrkconfigcode = mrkconfigcode.to(self.device)
        offset_output = self.models[0].static_encoder.decode1(offsetlatentcode)
        offset_output[0] = offset_output[0].view(-1, 24, 3)
        offset_output[0] = offset_output[0] * self.train_offset_data[1] + self.train_offset_data[0]
        res_offset_output_input = (offset_output[0] - self.train_offset_data[0]) / self.train_offset_data[1]
        marker_config_all = self.models[0].mc_encoder.decode(mrkconfigcode, res_offset_output_input, offsetlatentcode)
        res_marker_config_all = marker_config_all * self.train_mc_data[1] + self.train_mc_data[0]
        res_marker_config = res_marker_config_all[:, :, :, :3]
        skinning_weights = self.weights.view(1, MARKERNUM, JOINTNUM)
        motioncode = motioncode.view(-1, 112, 16)
        res_motion_out = self.models[0].auto_encoder.dec(motioncode, offset_output)
        res_skeleton_pos, res_transform = self.models[0].fk.forward_from_raw2(res_motion_out, offset_output[0],
                                                                              first_rot, world=True,
                                                                              quater=True)
        res_markers = self.models[0].skin.skinning(res_marker_config, skinning_weights, res_transform, res_skeleton_pos)
        return res_markers, res_transform, res_skeleton_pos

    def predict(self, raw_markers):
        raw_marker = raw_markers.to(self.device)
        _, marker_config_code, motion_code, offsets_code = self.models[0].Marker_encoder(raw_marker)
        marker_config_code = marker_config_code * self.train_mc_code_data[1] + self.train_mc_code_data[0]
        offsets_code = offsets_code * self.train_offset_code_data[1] + self.train_offset_code_data[0]
        motion_code0 = motion_code[:, 256:]
        motion_code0 = motion_code0 * self.train_motion_code_data[1] + self.train_motion_code_data[0]
        motion_code0 = motion_code0.view(-1, 112, 16)
        motion_code1 = motion_code[:, :256].view(-1, 64, 4)
        res_first_rot = motion_code1 * self.train_first_rot_data[1] + self.train_first_rot_data[0]
        res_first_rot = F.normalize(res_first_rot, dim=2)
        return motion_code0, offsets_code, res_first_rot, marker_config_code

    def forward(self):
        self.motions = []
        self.raw_markers = []
        self.clean_markers = []
        self.offsets = []
        self.marker_configs = []
        self.skinning_weights = []
        self.skeleton_pos = []
        self.first_rot = []
        self.res_offsets = []
        self.res_first_rot = []
        self.res_motions = []
        self.res_skeleton_pos = []
        self.res_transform = []
        self.transform = []
        self.res_markers = []
        self.res_marker_config = []
        self.res_skinning_weights = []
        self.coder_res_skel_pos = []
        self.res_mc_code = []
        self.mc_code = []
        self.res_offset_code = []
        self.offset_code = []
        self.first_rot_transform = []
        self.res_first_rot_transform = []

        # reconstruct
        for i in range(self.n_topology):
            # motion, offset_idx = self.motions_input[i]
            raw_marker, clean_marker, skeleton_pos, motion, offsets, marker_config, first_rot, of_code, mc_code, transform = self.motions_input
            motion = motion.to(self.device)
            raw_marker = raw_marker.to(self.device)
            clean_marker = clean_marker.to(self.device)
            offsets = offsets.to(self.device)
            marker_config = marker_config.to(self.device)
            first_rot = first_rot.to(self.device)
            skeleton_pos = skeleton_pos.to(self.device)
            of_code = of_code.to(self.device)
            mc_code = mc_code.to(self.device)
            self.skeleton_pos.append(skeleton_pos)
            transform = transform.to(self.device)
            self.transform.append(transform)
            self.motions.append(motion)
            self.raw_markers.append(raw_marker)
            self.clean_markers.append(clean_marker)
            self.offsets.append(offsets)
            self.marker_configs.append(marker_config[:, :, :, :3])
            self.first_rot.append(first_rot)
            self.offset_code.append(of_code)
            self.mc_code.append(mc_code)


            _, marker_config_code, motion_code, offsets_code = self.models[i].Marker_encoder(raw_marker)
            marker_config_code = marker_config_code * self.train_mc_code_data[1] + self.train_mc_code_data[0]
            self.res_mc_code.append(marker_config_code)
            offsets_code = offsets_code * self.train_offset_code_data[1] + self.train_offset_code_data[0]
            self.res_offset_code.append(offsets_code)
            motion_code0 = motion_code[:, 256:]
            motion_code0 = motion_code0 * self.train_motion_code_data[1] + self.train_motion_code_data[0]
            motion_code0 = motion_code0.view(-1, 112, 16)
            motion_code1 = motion_code[:, :256].view(-1, 64, 4)
            res_first_rot = motion_code1 * self.train_first_rot_data[1] + self.train_first_rot_data[0]
            res_first_rot = F.normalize(res_first_rot, dim=2)
            # trans_out =
            res_markers, res_transform, res_skeleton_pos, res_motion_out, offset_output, res_marker_config, skinning_weights \
                = self.predict4(motion_code0, offsets_code, res_first_rot, marker_config_code)
            res_first_rot_transform = self.models[i].fk.transform_from_quaternion(res_first_rot)
            first_rot_transform = self.models[i].fk.transform_from_quaternion(first_rot)
            self.res_first_rot_transform.append(res_first_rot_transform)
            self.first_rot_transform.append(first_rot_transform)

            self.res_offsets.append(offset_output)
            self.res_motions.append(res_motion_out)
            self.res_first_rot.append(res_first_rot)
            self.res_skeleton_pos.append(res_skeleton_pos)
            self.res_transform.append(res_transform)
            self.res_markers.append(res_markers)
            self.res_marker_config.append(res_marker_config)
        return

    def backward_G(self):

        self.rec_losses = []
        self.rec_loss = 0
        self.cycle_loss = 0
        self.loss_G = 0
        self.ee_loss = 0
        self.loss_G_total = 0
        self.rec_pos_error = 0
        for i in range(self.n_topology):
            ############## motion loss #####################################################
            rec_motion_loss = self.criterion_smooth_l1_rec_loss_motion_0 (self.motions[i][..., :-3], self.res_motions[i][..., :-3])
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_loss_motion_{}'.format(i), rec_motion_loss)

            rec_motion_trans_loss = self.criterion_smooth_l1_rec_loss_motion_0 (self.motions[i][:, -3:, :].permute(0, 2, 1), self.res_motions[i][:, -3:, :].permute(0,2,1))
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_motion_trans_loss_{}'.format(i), rec_motion_trans_loss)

            res_motion_trans_ave_err = self.vertex_criteria(self.motions[i][:, -3:, :].permute(0, 2, 1), self.res_motions[i][:, -3:, :].permute(0, 2, 1))
            self.loss_recoder.add_scalar('res_motion_trans_ave_err_{}'.format(i), res_motion_trans_ave_err)

            rec_motion_transform_loss = self.criterion_smooth_l1_rec_motion_transform_loss_0 (self.transform[i]* self.lambda_para_jt1, self.res_transform[i]* self.lambda_para_jt1)
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_motion_transform_loss_{}'.format(i), rec_motion_transform_loss)

            rec_motion_angle_loss = self.criter_angle_loss(self.transform[i], self.res_transform[i])
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_motion_angle_loss_{}'.format(i), rec_motion_angle_loss)

            ############## skeleton pos loss ################################################
            rec_skel_pos_loss = self.criterion_smooth_l1_skeleton_pos_loss_0 (self.skeleton_pos[i]* self.lambda_para_jt, self.res_skeleton_pos[i]* self.lambda_para_jt)
            self.loss_recoder.add_scalar('skeleton_pos_loss_{}'.format(i), rec_skel_pos_loss)

            ############## marker pos loss ###################################################
            res_mrk_pos_loss = self.criterion_smooth_l1_marker_pos_loss_0 (self.clean_markers[i]* self.lambda_para, self.res_markers[i]* self.lambda_para)
            self.loss_recoder.add_scalar('marker_pos_loss_{}'.format(i), res_mrk_pos_loss)

            ############## offsets loss ######################################################
            res_offset_loss = self.criterion_smooth_l1_offsets_loss_0 (self.offsets[i]* self.lambda_para_jt, self.res_offsets[i]* self.lambda_para_jt)
            self.loss_recoder.add_scalar('offsets_loss_{}'.format(i), res_offset_loss)

            ############## marker configuration loss #########################################
            res_mrk_config_loss = self.criterion_smooth_l1_marker_configuration_loss_0 (self.marker_configs[i], self.res_marker_config[i])
            self.loss_recoder.add_scalar('marker_configuration_loss_{}'.format(i), res_mrk_config_loss)


            ############# marker pos error average ############################################
            res_mrk_pos_ave_err = self.vertex_criteria(self.clean_markers[i], self.res_markers[i])
            self.loss_recoder.add_scalar('marker_pos_error_average_{}'.format(i), res_mrk_pos_ave_err)

            ############# skel pos error average ############################################
            res_skel_pos_ave_err = self.vertex_criteria(self.skeleton_pos[i] , self.res_skeleton_pos[i])
            self.loss_recoder.add_scalar('skel_pos_error_average_{}'.format(i), res_skel_pos_ave_err)

            res_first_rot_ave_err = self.criterion_smooth_l1_first_rot_error_average_0(self.first_rot[i], self.res_first_rot[i])
            self.loss_recoder.add_scalar('first_rot_error_average_{}'.format(i), res_first_rot_ave_err)

            rec_first_rot_transform_loss = self.criterion_smooth_l1_first_rot_transform_loss_0 (self.first_rot_transform[i], self.res_first_rot_transform[i])
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_first_rot_transform_loss_{}'.format(i), rec_first_rot_transform_loss)

            rec_first_rot_angle = self.criter_angle_loss(self.first_rot_transform[i], self.res_first_rot_transform[i])
            self.loss_recoder.add_scalar('rec_first_rot_angle_{}'.format(i), rec_first_rot_angle)

            res_mc_code_ave_arr = self.criterion_smooth_l1_res_mc_code_error_average_0 (self.res_mc_code[i], self.mc_code[i])
            self.loss_recoder.add_scalar('res_mc_code_error_average_{}'.format(i), res_mc_code_ave_arr)

            res_offset_code_ave_arr = self.criterion_smooth_l1_res_offset_code_error_average_0 (self.res_offset_code[i], self.offset_code[i])
            self.loss_recoder.add_scalar('res_offset_code_error_average_{}'.format(i), res_offset_code_ave_arr)
            rec_loss = 20. * res_mrk_pos_loss +  50 * rec_skel_pos_loss  + 1000. * rec_first_rot_transform_loss + 1. * rec_motion_transform_loss + 2. * res_offset_code_ave_arr + 10. * rec_motion_loss + 5000. * rec_motion_trans_loss + 1. * res_mc_code_ave_arr + 100. * res_mrk_config_loss + 100. * res_offset_loss
            self.rec_losses.append([rec_loss, res_mrk_pos_ave_err, rec_motion_angle_loss, res_skel_pos_ave_err])
            self.rec_loss += rec_loss

        self.loss_G_total = self.rec_loss  # * self.args.lambda_rec
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()
        return

    def test_data(self):
        self.rec_losses = []
        self.rec_loss = 0
        self.cycle_loss = 0
        self.loss_G = 0
        self.ee_loss = 0
        self.loss_G_total = 0
        self.rec_pos_error = 0
        for i in range(self.n_topology):
            ############## motion loss #####################################################
            rec_motion_loss = self.criterion_smooth_l1_rec_loss_motion_0 (self.motions[i][..., :-3], self.res_motions[i][..., :-3])
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('test_rec_loss_motion_{}'.format(i), rec_motion_loss)

            rec_motion_trans_loss = self.criterion_smooth_l1_rec_loss_motion_0 (self.motions[i][:, -3:, :].permute(0, 2, 1), self.res_motions[i][:, -3:, :].permute(0, 2, 1))
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('test_rec_motion_trans_loss_{}'.format(i), rec_motion_trans_loss)

            res_motion_trans_ave_err = self.vertex_criteria(self.motions[i][:, -3:, :].permute(0, 2, 1), self.res_motions[i][:, -3:, :].permute(0, 2, 1))
            self.loss_recoder.add_scalar('test_res_motion_trans_ave_err_{}'.format(i), res_motion_trans_ave_err)

            rec_motion_transform_loss = self.criterion_smooth_l1_rec_motion_transform_loss_0 (self.transform[i]* self.lambda_para_jt1, self.res_transform[i]* self.lambda_para_jt1)
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('test_rec_motion_transform_loss_{}'.format(i), rec_motion_transform_loss)

            rec_motion_angle_loss = self.criter_angle_loss(self.transform[i], self.res_transform[i])
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('test_rec_motion_angle_loss_{}'.format(i), rec_motion_angle_loss)

            ############## skeleton pos loss ################################################
            rec_skel_pos_loss = self.criterion_smooth_l1_skeleton_pos_loss_0 (self.skeleton_pos[i]* self.lambda_para_jt, self.res_skeleton_pos[i]* self.lambda_para_jt)
            self.loss_recoder.add_scalar('test_skeleton_pos_loss_{}'.format(i), rec_skel_pos_loss)

            ############## marker pos loss ###################################################
            res_mrk_pos_loss = self.criterion_smooth_l1_marker_pos_loss_0 (self.clean_markers[i]* self.lambda_para, self.res_markers[i]* self.lambda_para)
            self.loss_recoder.add_scalar('test_marker_pos_loss_{}'.format(i), res_mrk_pos_loss)

            ############## offsets loss ######################################################
            res_offset_loss = self.criterion_smooth_l1_offsets_loss_0 (self.offsets[i]* self.lambda_para_jt, self.res_offsets[i]* self.lambda_para_jt)
            self.loss_recoder.add_scalar('test_offsets_loss_{}'.format(i), res_offset_loss)

            ############## marker configuration loss #########################################
            res_mrk_config_loss = self.criterion_smooth_l1_marker_configuration_loss_0 (self.marker_configs[i], self.res_marker_config[i])
            self.loss_recoder.add_scalar('test_marker_configuration_loss_{}'.format(i), res_mrk_config_loss)


            ############# marker pos error average ############################################
            res_mrk_pos_ave_err = self.vertex_criteria(self.clean_markers[i], self.res_markers[i])
            self.loss_recoder.add_scalar('test_marker_pos_error_average_{}'.format(i), res_mrk_pos_ave_err)

            ############# skel pos error average ############################################
            res_skel_pos_ave_err = self.vertex_criteria(self.skeleton_pos[i] , self.res_skeleton_pos[i])
            self.loss_recoder.add_scalar('test_skel_pos_error_average_{}'.format(i), res_skel_pos_ave_err)

            res_first_rot_ave_err = self.criterion_smooth_l1_first_rot_error_average_0(self.first_rot[i], self.res_first_rot[i])
            self.loss_recoder.add_scalar('test_first_rot_error_average_{}'.format(i), res_first_rot_ave_err)

            rec_first_rot_transform_loss = self.criterion_smooth_l1_first_rot_transform_loss_0 (self.first_rot_transform[i], self.res_first_rot_transform[i])
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('test_rec_first_rot_transform_loss_{}'.format(i), rec_first_rot_transform_loss)

            rec_first_rot_angle = self.criter_angle_loss(self.first_rot_transform[i], self.res_first_rot_transform[i])
            self.loss_recoder.add_scalar('test_rec_first_rot_angle_{}'.format(i), rec_first_rot_angle)

            res_mc_code_ave_arr = self.criterion_smooth_l1_res_mc_code_error_average_0 (self.res_mc_code[i], self.mc_code[i])
            self.loss_recoder.add_scalar('test_res_mc_code_error_average_{}'.format(i), res_mc_code_ave_arr)

            res_offset_code_ave_arr = self.criterion_smooth_l1_res_offset_code_error_average_0 (self.res_offset_code[i], self.offset_code[i])
            self.loss_recoder.add_scalar('test_res_offset_code_error_average_{}'.format(i), res_offset_code_ave_arr)
            rec_loss = 20. * res_mrk_pos_loss +  50 * rec_skel_pos_loss  + 1000. * rec_first_rot_transform_loss + 1. * rec_motion_transform_loss + 2. * res_offset_code_ave_arr + 10. * rec_motion_loss + 5000. * rec_motion_trans_loss + 1. * res_mc_code_ave_arr + 100. * res_mrk_config_loss + 100. * res_offset_loss
            self.rec_losses.append([rec_loss, res_mrk_pos_ave_err, rec_motion_angle_loss, res_skel_pos_ave_err])
        return

    def verbose(self):
        res = {'Total_loss': self.rec_losses[0][0].item(),
            'joint_position_error': self.rec_losses[0][1].item(),
               'marker_pos_error_average_:': self.rec_losses[0][2].item(),
               'skel_pos_error_average_:': self.rec_losses[0][3].item(),
               }
        return res.items()

    def verbose1(self):
        L = [self.rec_losses[0][0].item(), self.rec_losses[0][1].item(), self.rec_losses[0][2].item(), self.rec_losses[0][3].item()]
        return L

    def save(self):
        for i, model in enumerate(self.models):
            model.save(os.path.join(self.model_save_dir), self.epoch_cnt)
            model.save1(os.path.join('models', 'mocap_solver.pth'))
        return

    def load(self, epoch=None):
        for i, model in enumerate(self.models):
            model.load(os.path.join(self.model_save_dir), epoch)

        self.epoch_cnt = epoch
        return

    def load2(self):
        for i, model in enumerate(self.models):
            model.load1(os.path.join('models', 'mocap_solver.pth'))
        return


