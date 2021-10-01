import os
import utils.option_parser as option_parser
import torch
from MoCap_Solver.model.skeleton import build_edge_topology
import torch.nn.functional as F
from utils.utils import prev_list, JOINTNUM, topology
from MoCap_Solver.model.Integrate_template_skeleton_encoder import IntegratedModel as Template_Skeleton_Model
from MoCap_Solver.io.Template_skeleton_dataloader import Template_skeleton_dataloader
import numpy as np


def fk(ts):
    for i in range(1, JOINTNUM):
        ts[:, i, :] = ts[:, prev_list[i], :] + ts[:, i, :]
    return ts

def evaluate_ts_encoder():
    edges = build_edge_topology(topology, torch.zeros((len(topology), 3)))
    args = option_parser.get_args()
    device = torch.device('cuda:0')
    model_class = Template_Skeleton_Model(args, topology, device)
    model_class.load(os.path.join('models', 'ts_encoder.pt'))
    dataset = Template_skeleton_dataloader(args)
    batch_size = 512
    train_data = np.load(os.path.join('models', 'train_data_ts.npy'))
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    num_epochs = 200
    last_loss = 1e10
    if not os.path.exists(os.path.join('models', 'checkpoints')):
        os.mkdir(os.path.join('models', 'checkpoints'))
    if not os.path.exists(os.path.join('models', 'checkpoints', 'Template_skeleton_encoder')):
        os.mkdir(os.path.join('models', 'checkpoints', 'Template_skeleton_encoder'))
    model_class.auto_encoder.train()
    with torch.no_grad():
        model_class.auto_encoder.eval()
        test_data = torch.tensor(dataset.test_set).to(device).view(-1, JOINTNUM, 3)
        test_data = (test_data - train_data[0]) / train_data[1]
        z = model_class.auto_encoder.encode(test_data)
        out = model_class.auto_encoder.decode(z).view(-1, JOINTNUM, 3)
        out = out.view(-1, JOINTNUM, 3)
        test_data = test_data * train_data[1] + train_data[0]
        out = out * train_data[1] + train_data[0]
        test_data = fk(test_data)
        out = fk(out)
        error = out - test_data
        error1 = torch.mean(torch.sqrt(torch.sum(torch.pow(error, 2), dim=2))).detach().cpu().numpy()
        print('Test Loss: Skleton position error:' + str(error1 * 1000.) + 'mm')
        model_class.auto_encoder.train()
