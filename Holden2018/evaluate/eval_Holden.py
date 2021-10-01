import logging
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import keras as K

from Holden2018.io.TrainDataGenerator import  ValGenerator
from Holden2018.io.train import Schedule, CustomModelCheckpoint, LossHistory
from keras.callbacks import LearningRateScheduler
from utils.utils import MARKERNUM, JOINTNUM

def eval_Holden():
    nb_epochs = 400
    lr = 0.001
    steps = 10000
    batch_size = 512
    history = LossHistory()
    output_path = os.path.join('models', 'checkpoints', 'Holden')
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    ############### 1. Load the training data and the testing data. ###########################
    weights_file_path = os.path.join('models', 'weights.npy')
    val_generator = ValGenerator(weights_file_path)
    ############### 2. Load the model. #######################################################
    from Holden2018.model.model_holden import get_Denoise_Markers_Model
    model = get_Denoise_Markers_Model(input_channel_num=2, marker_nums=MARKERNUM, joints_nums=JOINTNUM, lr=lr)
    pre_train_model = os.path.join('models', 'Holden.hdf5')
    if os.path.exists(pre_train_model):
        print('model_load!')
        model.load_weights(pre_train_model, by_name=True)
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    ################ 4. Start training. ##########################################################
    val_x_vec, val_y_vec0 = val_generator.get_all_vectors()
    hist = model.evaluate(
        val_x_vec,
        val_y_vec0,
        batch_size=batch_size,
        verbose=1
    )
    print('################# Final Holden model test loss ##########################################')
    print('Skeleton position error:' + str(hist[1]*1000.) + ' mm')
