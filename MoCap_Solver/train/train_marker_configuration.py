import os
import utils.option_parser as option_parser
import torch
from MoCap_Solver.model.skeleton import build_edge_topology
import numpy as np
from utils.utils import topology, MARKERNUM, JOINTNUM

from MoCap_Solver.model.Integrate_marker_config_encoder import IntegratedModel as Marker_Config_Model

def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay

def train_marker_configuration():
    if not os.path.exists(os.path.join('models', 'checkpoints')):
        os.mkdir(os.path.join('models', 'checkpoints'))
    if not os.path.exists(os.path.join('models', 'checkpoints', 'Marker_configuration_encoder')):
        os.mkdir(os.path.join('models', 'checkpoints', 'Marker_configuration_encoder'))
    edges = build_edge_topology(topology, torch.zeros((len(topology), 3)))
    args = option_parser.get_args()
    device = torch.device('cuda:0')
    model_class = Marker_Config_Model(args, topology, device)
    model = model_class.auto_encoder
    model_class.template_skeleton_load(os.path.join('models', 'ts_encoder.pt'))
    ts_model = model_class.ts_auto_encoder
    learning_rate = 2e-4
    from MoCap_Solver.io.Marker_configuration_dataloader import Marker_configuration_dataloader
    dataset = Marker_configuration_dataloader(args)
    batch_size = 256
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    train_data = np.load(os.path.join('models','train_data_marker_config.npy'))
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_ts_data = np.load(os.path.join('models', 'train_data_ts.npy'))
    train_ts_data = torch.tensor(train_ts_data, dtype=torch.float32).to(device)
    weights = np.load(os.path.join('models', 'weights.npy')).reshape(MARKERNUM, JOINTNUM, 1)
    weights = np.tile(weights, (1, 1, 3))
    weights = np.array(weights, dtype=np.float32)
    weights = torch.tensor(weights)
    weights = weights.to(device)
    params_decay, params_no_decay = split_parameters(model_class.auto_encoder)
    optimizer = torch.optim.Adam([{'params': params_decay, 'weight_decay': 0.0005},
                                  {'params': params_no_decay}], lr=learning_rate)
    num_epochs = 100000
    last_loss = 1e10
    model_class.ts_auto_encoder.eval()
    model_class.auto_encoder.train()

    for epoch in range(0, num_epochs):
        for i, x in enumerate(data_loader):
            x0 = x[0]
            x1 = x[1]
            x0 = x0.to(device).view(-1, MARKERNUM, JOINTNUM, 3)
            mc_gt = x0[:, :, :, :3]
            x1 = x1.to(device).view(-1, JOINTNUM, 3)
            x0 = (x0 - train_data[0])/train_data[1]
            x1 = (x1 - train_ts_data[0])/train_ts_data[1]
            x1_reconst, x1_latent = model_class.ts_auto_encoder(x1)
            x1_reconst = x1_reconst.view(-1, JOINTNUM, 3)
            x_reconst, latent = model_class.auto_encoder(x0, x1_reconst, x1_latent)
            x_reconst = x_reconst.view(-1, MARKERNUM, JOINTNUM, 3)
            x_reconst = x_reconst * train_data[1] + train_data[0]
            x0 = x0 * train_data[1] + train_data[0]
            x_reconst1 = x_reconst[:, :, :, :3]
            x_reconst1 = x_reconst1 * weights
            x_reconst1 = torch.sum(x_reconst1, dim=2)
            weights_gt = weights.view(-1, MARKERNUM, JOINTNUM, 1)
            x_gt = mc_gt * weights
            x_gt = torch.sum(x_gt, dim=2)

            reconst_loss1 = 2. *torch.nn.SmoothL1Loss() (x_reconst[:,:, :, :3], x0[:,:,:,:3])
            reconst_loss = 20. * torch.nn.SmoothL1Loss()(x_reconst1, x_gt)

            loss = reconst_loss + reconst_loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model_class.auto_encoder.eval()
            test_data = torch.tensor(dataset.test_data).to(device).view(-1, MARKERNUM, JOINTNUM, 3)
            test_data = (test_data-train_data[0])/train_data[1]
            test_ts_data = torch.tensor(dataset.test_ts_data).to(device).view(-1, JOINTNUM, 3)
            test_ts_data = (test_ts_data - train_ts_data[0])/train_ts_data[1]
            xx, yy = model_class.ts_auto_encoder(test_ts_data)
            xx = xx.view(-1, JOINTNUM, 3)
            z, _ = model_class.auto_encoder(test_data, xx, yy)

            out = z.view(-1, MARKERNUM, JOINTNUM, 3)
            test_data = test_data * train_data[1] + train_data[0]
            test_data0 = test_data[:, :, :, :3]
            out = out * train_data[1] + train_data[0]
            out0 = out[:, :, :, :3]
            test_data0 = test_data0 * weights
            test_data0 = torch.sum(test_data0, dim=2)
            out0 = out0 * weights
            out0 = torch.sum(out0, dim=2)

            error = out0 - test_data0
            error1 = torch.mean(torch.sqrt(torch.sum(torch.pow(error,2),dim = 2))).detach()
            print ('evaluate epoch%d'%epoch + ':  ' + str(error1 * 1000) + 'mm' )
            model_class.auto_encoder.train()

        if error1 < last_loss:
            last_loss = error1
            model_class.save(os.path.join('models', 'checkpoints', 'Marker_configuration_encoder', '%d.pt'%epoch))
            model_class.save(os.path.join('models', 'mc_encoder.pt'))

