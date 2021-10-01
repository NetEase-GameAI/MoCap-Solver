import sys
from torch.utils.data.dataloader import DataLoader
from MoCap_Solver.model.MotionEncoderClass import MotionEncoderClass
from MoCap_Solver.io.Motion_dataloader import MixedData as Motion_dataloader
import utils.option_parser as option_parser
import os
from utils.option_parser import try_mkdir
import time
import glob
import torch
from utils.utils import FRAMENUM
import numpy as np
def main():
    args = option_parser.get_args()
    args.window_size = FRAMENUM
    npzfiles = glob.glob(os.path.join('MoCap_Solver', 'data', 'Training_dataset', 'motion_dataset',  '*.npz'))
    test_npzfiles = glob.glob(os.path.join('MoCap_Solver', 'data', 'Testing_dataset', 'motion_dataset',  '*.npz'))

    characters = [[npzfile for npzfile in npzfiles[:]]]
    test_characters = [[npzfile for npzfile in test_npzfiles[:]]]
    args.verbose = 1
    args.save_dir = os.path.join('models', 'checkpoints', 'Motion_encoder')

    log_path = os.path.join(args.save_dir, 'logs')
    try_mkdir(args.save_dir)
    try_mkdir(log_path)
    try_mkdir(os.path.join('models', 'checkpoints'))

    with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file:
        para_file.write(' '.join(sys.argv))

    dataset = Motion_dataloader(args, characters, False)
    # sys.exit()
    test_dataset = Motion_dataloader(args, test_characters, False)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MotionEncoderClass(args, characters, dataset, test_dataset)
    model.load2()

    model.setup()

    start_time = time.time()
    test_loss = 1e10

    model.models[0].static_encoder.eval()

    statistic_data = np.zeros((3))
    with torch.no_grad():
        model.models[0].auto_encoder.eval()
        num = 0
        for step, motions in enumerate(test_data_loader):
            model.set_input(motions)
            model.forward1()
            model.test1()
            res = model.verbose1()
            statistic_data[0] += res[0]
            statistic_data[1] += res[1]
            statistic_data[2] += res[2]
            num += 1
        statistic_data = statistic_data / num
        print('############# Test loss of Motion Encoder ##################')
        print('skeleton position error: ' + str(1000. * statistic_data[1])
              +'mm  skeleton rotation error: ' + str(statistic_data[2]) + 'deg')



def evaluate_motion_encoder():
    main()
