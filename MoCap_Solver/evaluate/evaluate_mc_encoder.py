import os
import utils.option_parser as option_parser
import torch
from MoCap_Solver.model.skeleton import build_edge_topology
import numpy as np
from utils.utils import topology, MARKERNUM, JOINTNUM

from MoCap_Solver.model.Integrate_marker_config_encoder import IntegratedModel as Marker_Config_Model

def evaluate_mc_encoder():
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
    model_class.load(os.path.join('models', 'mc_encoder.pt'))
    last_loss = 1e10
    model_class.ts_auto_encoder.eval()
    model_class.auto_encoder.train()
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
        error1 = torch.mean(torch.sqrt(torch.sum(torch.pow(error,2),dim = 2))).detach().cpu().numpy()
        print ('Test loss: marker position error: ' + str(error1 * 1000) + 'mm' )
