import sys
from torch.utils.data.dataloader import DataLoader
import utils.option_parser as option_parser
import os
from utils.option_parser import try_mkdir
import time
import torch
import numpy as np
from MoCap_Solver.io.MoCap_dataloader import TrainData, ValData
from utils.utils import topology
from MoCap_Solver.model.MoCapSolverClass import MoCapSolverModel
from MoCap_Solver.io.NoiseAdd import add_noise

def train_mocap_solver():
    args1 = option_parser.get_args()
    args1.verbose = 1
    args1.save_dir = os.path.join('models', 'checkpoints', 'MoCap_solver')
    try_mkdir(args1.save_dir)
    args1.learning_rate = 1. *(1e-4)
    args1.scheduler == 'Step_LR'

    with open(os.path.join(args1.save_dir, 'para.txt'), 'w') as para_file:
        para_file.write(' '.join(sys.argv))

    train_dataset = TrainData(args1 , statistic_on=False)
    test_dataset = ValData(args1)
    train_data_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, drop_last=True)
    print('data loader over!')
    model = MoCapSolverModel(args1, topology)
    print('model init over!')
    args1.epoch_begin = 0
    args1.epoch_num = 800

    if os.path.exists(os.path.join('models', 'mocap_solver.pth')):
        model.load2()
        print("load pretrained!")
    model.setup()
    print('model set up over!')
    start_time = time.time()

    model.model.auto_encoder.eval()
    model.model.mc_encoder.eval()
    model.model.static_encoder.eval()
    test_loss = 1e10
    for epoch in range(args1.epoch_begin, args1.epoch_num):
        model.model.Marker_encoder.train()
        for step, motions in enumerate(train_data_loader):
            model.set_input(motions)
            model.optimize_parameters()

            if args1.verbose:
                res = model.verbose()
                print('[{}/{}]\t[{}/{}]\t'.format(epoch, args1.epoch_num, step, len(train_data_loader)), res)
        statistic_data = np.zeros((4))
        with torch.no_grad():
            model.model.Marker_encoder.eval()
            num = 0
            for step, motions in enumerate(test_data_loader):
                model.set_input(motions)
                model.forward()
                model.test_data()
                res = model.verbose1()
                statistic_data[0] += res[0]
                statistic_data[1] += res[1]
                statistic_data[2] += res[2]
                statistic_data[3] += res[3]
                num += 1
            statistic_data = statistic_data/num
            print('############# Test loss of epoch {} ##################'.format(epoch))
            print('Total loss: ' + str(statistic_data[0]) + '  marker position error: ' + str(1000. * statistic_data[1])
                  +' mm  skeleton rotation error: ' + str(statistic_data[2]) + ' deg  skeleton position error: ' + str(1000. * statistic_data[3]) + ' mm')
            model.model.Marker_encoder.train()
        if statistic_data[0] < test_loss:
            print("Validation loss decress from: " + str(test_loss) + " to " + str(statistic_data[0]))
            test_loss = statistic_data[0]

        model.save()
        model.epoch()

    end_time = time.time()
    print('training time', end_time - start_time)

