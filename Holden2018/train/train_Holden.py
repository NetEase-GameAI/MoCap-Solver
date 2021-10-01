import logging
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import keras as K

from Holden2018.io.TrainDataGenerator import TrainGenerator, ValGenerator
from Holden2018.io.train import Schedule, CustomModelCheckpoint, LossHistory
from keras.callbacks import LearningRateScheduler
from utils.utils import MARKERNUM, JOINTNUM

def train_Holden():
    nb_epochs = 400
    lr = 0.001
    steps = 10000
    batch_size = 512
    history = LossHistory()
    if not os.path.exists(os.path.join('models', 'checkpoints')):
        os.mkdir(os.path.join('models', 'checkpoints'))
    output_path = os.path.join('models', 'checkpoints', 'Holden')
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    ############### 1. Load the training data and the testing data. ###########################
    weights_file_path = os.path.join('models', 'weights.npy')
    train_generator = TrainGenerator(weights_file_path, statistic_on=True)
    val_generator = ValGenerator(weights_file_path)
    ############### 2. Load the model. #######################################################
    from Holden2018.model.model_holden import get_Denoise_Markers_Model
    model = get_Denoise_Markers_Model(input_channel_num=2, marker_nums=MARKERNUM, joints_nums=JOINTNUM, lr=lr)
    pre_train_model = os.path.join(output_path, 'previous.hdf5')
    if os.path.exists(pre_train_model):
        print('model_load!')
        model.load_weights(pre_train_model, by_name=True)
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    prev_str = os.path.join(output_path, 'weights.')
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    ############### 3. specify the tensorboard log file dir. ####################################
    tbCallBack = K.callbacks.TensorBoard(log_dir=log_dir,
                                         histogram_freq=0,
                                         write_graph=False,
                                         write_images=True
                                         )
    callbacks = [
        LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
        CustomModelCheckpoint(model, prev_str + "weights.{epoch:03d}-{val_loss:.3f}.hdf5"),
        tbCallBack
    ]
    ################ 4. Start training. ##########################################################
    data_x_vec, data_y_vec0= train_generator.get_all_vectors()
    val_x_vec, val_y_vec0 = val_generator.get_all_vectors()

    print(len(data_x_vec))
    hist = model.fit(
        data_x_vec,
        data_y_vec0,
        batch_size=batch_size,
        epochs=nb_epochs,
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(
            val_x_vec, val_y_vec0)
    )
