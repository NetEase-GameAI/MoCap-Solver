import logging
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import keras as K
from keras import backend as K1
from keras.layers import Input, Add,Dropout,PReLU,LeakyReLU,Conv2D,Conv2DTranspose,Concatenate, Lambda,  Multiply

from Holden2018.io.TrainDataGenerator import  ValGenerator
from Holden2018.io.train import Schedule, CustomModelCheckpoint, LossHistory
from keras.callbacks import LearningRateScheduler
from utils.utils import MARKERNUM, JOINTNUM
import numpy as np


def eval_Holden():
    nb_epochs = 500
    lr = 0.001
    steps = 10000
    batch_size = 512
    history = LossHistory()
    output_path = os.path.join('models', 'checkpoints', 'Holden')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ############### 1. Load the training data and the testing data. ###########################
    weights_file_path = os.path.join('models', 'weights.npy')
    weights = np.load(weights_file_path, allow_pickle=True) #56 * 24
    print("weights shape: " + str(weights.shape))

    val_generator = ValGenerator(weights_file_path)
    ############### 2. Load the model. #######################################################
    from Holden2018.model.model_holden import get_Denoise_Markers_Model
    model = get_Denoise_Markers_Model(input_channel_num=2, marker_nums=MARKERNUM, joints_nums=JOINTNUM, lr=lr, weights=weights)
    pre_train_model = os.path.join('models', 'Holden.hdf5')
    if os.path.exists(pre_train_model):
        print('model_load! ' + pre_train_model)
        model.load_weights(pre_train_model, by_name=True)
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    ################ 4. Start training. ##########################################################
    ##x_vec input:noise marker + mean mrk config (N,56,3,2), y_vec1:noise marker (N,56,3) y_vec2:clean marker （N,56,3） y_vec3:mrkconfig （N,56,24,3）
    val_x_vec,val_y_vec0, val_y_vec1, val_y_vec2, val_y_vec3 = val_generator.get_all_vectors()

    hist = model.evaluate(
        [val_x_vec,val_y_vec3],
        [val_y_vec0,val_y_vec1,val_y_vec2],
        batch_size=batch_size,
        verbose=1
    )
    print(len(hist))
    print(hist)
    print('################# Final Holden model test loss ##########################################')
    print('Skeleton position error:' + str(hist[4]*1000.) + ' mm' )
    print('Skeleton rotation error:' + str(hist[5]) + ' deg')
    print('Marker error:' + str(hist[6]*1000.) + ' mm')