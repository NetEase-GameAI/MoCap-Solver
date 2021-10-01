from MoCap_Solver.model.Integrate_motion_encoder import IntegratedModel
from torch import optim
import torch
from utils.utils import  Criterion_EE, Eval_Criterion, angle_loss, vertex_loss
from MoCap_Solver.model.base_model import BaseModel
import os

class MotionEncoderClass(BaseModel):
    def __init__(self, args, character_names, dataset, test_dataset):
        super(MotionEncoderClass, self).__init__(args)
        self.character_names = character_names
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.n_topology = len(character_names)
        self.models = []
        self.D_para = []
        self.G_para = []
        self.args = args
        self.vertex_criteria = vertex_loss()
        self.criter_angle_loss = angle_loss()
        self.model_save_dir = os.path.join('models', 'checkpoints', 'Motion_encoder')

        for i in range(self.n_topology):
            model = IntegratedModel(args, dataset.joint_topologies[i], None, self.device, character_names[i])
            self.models.append(model)
            # self.D_para += model.D_parameters()
            self.G_para += model.parameters()

        if self.is_train:
            self.fake_pools = []
            self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizers = [self.optimizerG]
            self.criterion_rec = torch.nn.MSELoss()
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_ee = Criterion_EE(args, torch.nn.MSELoss())
        else:
            self.err_crit = []
            for i in range(self.n_topology):
                self.err_crit.append(Eval_Criterion(dataset.joint_topologies[i]))
            self.id_test = 0

    def set_input(self, motions):
        self.motions_input = motions

        if not self.is_train:
            self.motion_backup = []
            for i in range(self.n_topology):
                self.motion_backup.append(motions[i][0].clone())
                self.motions_input[i][0][1:] = self.motions_input[i][0][0]
                self.motions_input[i][1] = [0] * len(self.motions_input[i][1])

    def forward(self):
        self.latents = []
        self.offset_repr = []
        self.pos_ref = []
        self.ee_ref = []
        self.res = []
        self.res_denorm = []
        self.res_pos = []
        self.fake_res = []
        self.fake_res_denorm = []
        self.fake_pos = []
        self.fake_ee = []
        self.fake_latent = []
        self.motions = []
        self.motion_denorm = []
        self.rnd_idx = []
        self.res_transform = []
        self.transform = []

        for i in range(self.n_topology):
            offsetinput = self.dataset.offset_norm(self.dataset.offsets[i])
            latentcode = self.models[i].static_encoder.encode(offsetinput)
            output = self.models[i].static_encoder.decode1(latentcode)
            output[0] = self.dataset.offset_denorm(output[0].view(-1, 24, 3))
            self.offset_repr.append(output)
            # self.offset_repr.append(self.models[i].static_encoder(self.dataset.offsets[i]))

        # reconstruct
        for i in range(self.n_topology):
            motion, offset_idx = self.motions_input[i]
            motion = motion.to(self.device)
            self.motions.append(motion)

            motion_denorm = self.dataset.denorm(i, offset_idx, motion)
            self.motion_denorm.append(motion_denorm)
            offsets = [self.offset_repr[i][p][offset_idx] for p in range(self.args.num_layers + 1)]
            latent, res = self.models[i].auto_encoder(motion, offsets)
            res_denorm = self.dataset.denorm(i, offset_idx, res)
            res_pos, res_transform = self.models[i].fk.forward_from_raw1(res_denorm, self.dataset.offsets[i][offset_idx], world=True)
            self.res_transform.append(res_transform)
            self.res_pos.append(res_pos)
            self.latents.append(latent)
            self.res.append(res)
            self.res_denorm.append(res_denorm)

            pos, transform = self.models[i].fk.forward_from_raw1(motion_denorm, self.dataset.offsets[i][offset_idx], world=True)
            pos = pos.detach()
            transform = transform.detach()

            self.pos_ref.append(pos)
            self.transform.append(transform)

    def forward1(self):
        self.latents = []
        self.offset_repr = []
        self.pos_ref = []
        self.ee_ref = []
        self.res = []
        self.res_denorm = []
        self.res_pos = []
        self.fake_res = []
        self.fake_res_denorm = []
        self.fake_pos = []
        self.fake_ee = []
        self.fake_latent = []
        self.motions = []
        self.motion_denorm = []
        self.rnd_idx = []
        self.res_transform = []
        self.transform = []

        for i in range(self.n_topology):
            offsetinput = self.test_dataset.offset_norm(self.test_dataset.offsets[i])
            latentcode = self.models[i].static_encoder.encode(offsetinput)
            output = self.models[i].static_encoder.decode1(latentcode)
            output[0] = self.test_dataset.offset_denorm(output[0].view(-1, 24, 3))
            self.offset_repr.append(output)
            # self.offset_repr.append(self.models[i].static_encoder(self.dataset.offsets[i]))

        # reconstruct
        for i in range(self.n_topology):
            motion, offset_idx = self.motions_input[i]
            motion = motion.to(self.device)
            self.motions.append(motion)

            motion_denorm = self.test_dataset.denorm(i, offset_idx, motion)
            self.motion_denorm.append(motion_denorm)
            offsets = [self.offset_repr[i][p][offset_idx] for p in range(self.args.num_layers + 1)]
            latent, res = self.models[i].auto_encoder(motion, offsets)
            res_denorm = self.test_dataset.denorm(i, offset_idx, res)
            res_pos, res_transform = self.models[i].fk.forward_from_raw1(res_denorm, self.test_dataset.offsets[i][offset_idx], world=True)
            self.res_pos.append(res_pos)
            self.res_transform.append(res_transform)
            self.latents.append(latent)
            self.res.append(res)
            self.res_denorm.append(res_denorm)

            pos, transform = self.models[i].fk.forward_from_raw1(motion_denorm, self.test_dataset.offsets[i][offset_idx], world=True)
            pos = pos.detach()
            transform = transform.detach()
            self.transform.append(transform)
            self.pos_ref.append(pos)


    def test1(self):
        #rec_loss and gan loss
        self.rec_losses = []
        self.rec_loss = 0
        self.cycle_loss = 0
        self.loss_G = 0
        self.ee_loss = 0
        self.loss_G_total = 0
        self.rec_pos_error = 0
        for i in range(self.n_topology):
            rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('test_rec_loss_quater_{}'.format(i), rec_loss1)

            input_pos = self.motion_denorm[i][:, -3:, :]
            rec_pos = self.res_denorm[i][:, -3:, :]
            rec_loss2 = self.criterion_rec(input_pos, rec_pos)
            self.loss_recoder.add_scalar('test_rec_loss_global_{}'.format(i), rec_loss2)

            pos_ref_global = self.pos_ref[i]
            #self.models[i].fk.from_local_to_world(self.pos_ref[i])
            res_pos_global = self.res_pos[i]
            #self.models[i].fk.from_local_to_world(self.res_pos[i])
            rec_loss3 = self.criterion_rec(pos_ref_global, res_pos_global)
            self.loss_recoder.add_scalar('test_rec_loss_position_{}'.format(i), rec_loss3)
            rec_loss4 = self.vertex_criteria(pos_ref_global, res_pos_global)
            self.loss_recoder.add_scalar('test_rec_loss_vertex_per_{}'.format(i), rec_loss4)

            rec_motion_angle_loss = self.criter_angle_loss(self.transform[i], self.res_transform[i])
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('test_rec_motion_angle_loss_{}'.format(i), rec_motion_angle_loss)

            rec_loss = rec_loss1 + (rec_loss2 * self.args.lambda_global_pose +
                                    rec_loss3 * self.args.lambda_position) * 100.

            self.rec_losses.append([rec_loss, rec_loss4, rec_motion_angle_loss])



    def backward_G(self):
        #rec_loss and gan loss
        self.rec_losses = []
        self.rec_loss = 0
        self.cycle_loss = 0
        self.loss_G = 0
        self.ee_loss = 0
        self.loss_G_total = 0
        self.rec_pos_error = 0
        for i in range(self.n_topology):
            rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_loss_quater_{}'.format(i), rec_loss1)

            input_pos = self.motion_denorm[i][:, -3:, :]
            rec_pos = self.res_denorm[i][:, -3:, :]
            rec_loss2 = self.criterion_rec(input_pos, rec_pos)
            self.loss_recoder.add_scalar('rec_loss_global_{}'.format(i), rec_loss2)

            pos_ref_global = self.pos_ref[i]
            #self.models[i].fk.from_local_to_world(self.pos_ref[i])
            res_pos_global = self.res_pos[i]
            #self.models[i].fk.from_local_to_world(self.res_pos[i])
            rec_loss3 = self.criterion_rec(pos_ref_global, res_pos_global)
            self.loss_recoder.add_scalar('rec_loss_position_{}'.format(i), rec_loss3)
            rec_loss4 = self.vertex_criteria(pos_ref_global, res_pos_global)
            self.loss_recoder.add_scalar('rec_loss_vertex_per_{}'.format(i), rec_loss4)

            rec_motion_angle_loss = self.criter_angle_loss(self.transform[i], self.res_transform[i])
            # rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_motion_angle_loss_{}'.format(i), rec_motion_angle_loss)

            rec_loss = rec_loss1 + (rec_loss2 * self.args.lambda_global_pose +
                                    rec_loss3 * self.args.lambda_position) * 100.

            self.rec_losses.append([rec_loss, rec_loss4, rec_motion_angle_loss])
            self.rec_loss += rec_loss

        self.loss_G_total = self.rec_loss * self.args.lambda_rec
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

    def verbose(self):
        res = {'Total_loss': self.rec_losses[0][0].item(),
               'skeleton_position_error': self.rec_losses[0][1].item(),
               'skeleton_rotation_error': self.rec_losses[0][2].item(),
               }
        return res.items()

    def verbose1(self):
        L = [self.rec_losses[0][0].item(), self.rec_losses[0][1].item(), self.rec_losses[0][2].item()]
        return L

    def save(self):
        for i, model in enumerate(self.models):
            model.save(os.path.join(self.model_save_dir, 'topology{}'.format(i)) + '_' + str(self.epoch_cnt) + '.pt')
            model.save(os.path.join('models', 'motion_encoder.pt'))

        # for i, optimizer in enumerate(self.optimizers):
        #     file_name = os.path.join(self.model_save_dir,'optimizers'+'_{}'.format(self.epoch_cnt) +'_{}.pt'.format( i))
        #     torch.save(optimizer.state_dict(), file_name)

    def load(self, epoch=None):
        for i, model in enumerate(self.models):
            print(os.path.join(self.model_save_dir, 'topology{}'.format(i) + '_' + str(epoch) + '.pt'))
            model.load(os.path.join(self.model_save_dir, 'topology{}'.format(i) + '_' + str(epoch) + '.pt'))
        self.epoch_cnt = epoch

    def load2(self):
        for i, model in enumerate(self.models):
            model.load(os.path.join('models', 'motion_encoder.pt'))
