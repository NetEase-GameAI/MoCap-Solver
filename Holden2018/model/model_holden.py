import os
os.environ['KERAS_BACKEND']= 'tensorflow'
from keras.models import Model
from keras.layers import Input, Add,Dropout,PReLU,LeakyReLU,Conv2D,Conv2DTranspose,Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation,Flatten,Reshape
from keras import regularizers
from keras import backend as K
import keras
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np

from utils.utils import _dict_marker, _dict_joints, MARKERNUM, JOINTNUM


def get_Denoise_Markers_Model(input_channel_num = 2,marker_nums=MARKERNUM, joints_nums=JOINTNUM, lr = 0.0001):
    train_joints_data = np.load(os.path.join('models', "train_data_joint_holden.npy"))
    gama_para = 0.001
    gama_choose = True
    lambda_para =  np.ones((marker_nums,3))
    lambda_jt_para = np.ones((joints_nums,3, 4))

    head_idx = [_dict_marker["ARIEL"],_dict_marker["LBHD"],_dict_marker["LFHD"],_dict_marker["RFHD"],_dict_marker["RBHD"]]
    head_jt_idx =  [_dict_joints["Head"]]
    shoulder_idx = [_dict_marker["LTSH"],_dict_marker["LBSH"],_dict_marker["LFSH"],_dict_marker["RTSH"],_dict_marker["RFSH"],_dict_marker["RBSH"]]
    shoulder_jt_idx = [_dict_joints['L_Collar'], _dict_joints['R_Collar'], _dict_joints['L_Shoulder'], _dict_joints['R_Shoulder']]
    arm_idx = [
                _dict_marker["LUPA"],_dict_marker["LIEL"],_dict_marker["LELB"],_dict_marker["LWRE"],
                _dict_marker["RUPA"],_dict_marker["RIEL"],_dict_marker["RELB"],_dict_marker["RWRE"]
              ]
    arm_jt_idx = [_dict_joints['L_Elbow'], _dict_joints['R_Elbow']]
    wrist_hand_idx = [
                        _dict_marker["LOWR"],_dict_marker["LIWR"],_dict_marker["LIHAND"],_dict_marker["LOHAND"],
                        _dict_marker["ROWR"],_dict_marker["RIWR"],_dict_marker["RIHAND"],_dict_marker["ROHAND"]
                     ]
    wrist_hand_jt_idx = [_dict_joints['L_Wrist'], _dict_joints['R_Wrist'], _dict_joints['L_Hand'], _dict_joints['R_Hand']]

    torso_idx = [
                    _dict_marker["CLAV"],_dict_marker["STRN"],_dict_marker["C7"],_dict_marker["T10"],_dict_marker["L4"],
                    _dict_marker["LMWT"],_dict_marker["LFWT"],_dict_marker["LBWT"],
                    _dict_marker["RMWT"],_dict_marker["RFWT"],_dict_marker["RBWT"]
                ]

    torso_jt_idx = [_dict_joints['Pelvis'],_dict_joints['Spine1'], _dict_joints['Spine2'], _dict_joints['Spine3'], _dict_joints['L_Hip'], _dict_joints['R_Hip'], _dict_joints['Neck']]
    thigh_idx = [
                    _dict_marker["LKNI"],_dict_marker["LKNE"],_dict_marker["LHIP"],_dict_marker["LSHN"],
                    _dict_marker["RKNI"],_dict_marker["RKNE"],_dict_marker["RHIP"],_dict_marker["RSHN"]
                ]
    thigh_jt_idx = [_dict_joints['L_Knee'], _dict_joints['R_Knee']]

    foots_idx = [
                    _dict_marker["LANK"],_dict_marker["LHEL"],_dict_marker["LMT1"],_dict_marker["LTOE"],_dict_marker["LMT5"],
                    _dict_marker["RANK"],_dict_marker["RHEL"],_dict_marker["RMT1"],_dict_marker["RTOE"],_dict_marker["RMT5"]
                ]
    foots_jt_idx = [_dict_joints['L_Ankle'], _dict_joints['R_Ankle'], _dict_joints['L_Foot'], _dict_joints['R_Foot']]

    lambda_para[[head_idx]] = lambda_para[[head_idx]]*10 #head
    lambda_para[[shoulder_idx]] = lambda_para[[shoulder_idx]]*5 # shoulder
    lambda_para[[arm_idx]] = lambda_para[[arm_idx]]*8 #arm
    lambda_para[[wrist_hand_idx]] = lambda_para[[wrist_hand_idx]]*10 #wrist
    lambda_para[[torso_idx]] = lambda_para[[torso_idx]]*5 #torso
    lambda_para[[thigh_idx]] = lambda_para[[thigh_idx]]*8 #thigh
    lambda_para[[foots_idx]] = lambda_para[[foots_idx]]*10 #foots
    
    lambda_jt_para[[head_jt_idx]] = lambda_jt_para[[head_jt_idx]]*10 #head
    lambda_jt_para[[shoulder_jt_idx]] = lambda_jt_para[[shoulder_jt_idx]]*5 # shoulder
    lambda_jt_para[[arm_jt_idx]] = lambda_jt_para[[arm_jt_idx]]*8 #arm
    lambda_jt_para[[wrist_hand_jt_idx]] = lambda_jt_para[[wrist_hand_jt_idx]]*10 #wrist
    lambda_jt_para[[torso_jt_idx]] = lambda_jt_para[[torso_jt_idx]]*5 #torso
    lambda_jt_para[[thigh_jt_idx]] = lambda_jt_para[[thigh_jt_idx]]*8 #thigh
    lambda_jt_para[[foots_jt_idx]] = lambda_jt_para[[foots_jt_idx]]*10 #foots

    def _residual_block(inputs,feature_dim):
        x_0 = BatchNormalization()(inputs)
        x_0 = LeakyReLU(alpha=0.5)(x_0)
        if gama_choose == True:        
            x = Dense(feature_dim,kernel_initializer='he_normal',activity_regularizer = regularizers.l2(gama_para))(x_0)  
        else:
            x = Dense(feature_dim,kernel_initializer='he_normal')(x_0)        
        m = Add()([x, x_0])
        return m

    inputs = Input(shape=(marker_nums,3,input_channel_num))
    features = Flatten()(inputs)
    if gama_choose == True:   
        x = Dense(2048,kernel_initializer='he_normal',activity_regularizer = regularizers.l2(gama_para))(features)
    else:
        x = Dense(2048,kernel_initializer='he_normal')(features)
    x = _residual_block(x,2048)

    x = _residual_block(x,2048)
	
    x = _residual_block(x,2048)
	
    x = _residual_block(x,2048)

    x = _residual_block(x,2048)

    x_o = LeakyReLU(alpha=0.5)(x) 

    if gama_choose == True:
        x_o = Dense(joints_nums*3*4,kernel_initializer='he_normal',activity_regularizer = regularizers.l2(gama_para))(x_o)
    else:
        x_o = Dense(joints_nums*3*4,kernel_initializer='he_normal')(x_o)
    
    x_o = Reshape((joints_nums, 3, 4), input_shape = (joints_nums * 3 * 4,), name = 'x_output')(x_o)

    HUBER_DELTA = 200

    def smoothL1(y_true, y_pred):
        y_pred = y_pred * train_joints_data[1] + train_joints_data[0]
        error_mat = lambda_jt_para * K.abs(y_true - y_pred)
        _error_ = K.sqrt(K.sum(K.square(error_mat)))
        x = K.switch(_error_ < HUBER_DELTA, \
                     0.5 * _error_ ** 2, \
                     0.5 * HUBER_DELTA ** 2 + HUBER_DELTA * (_error_ - HUBER_DELTA))
        return x

    def joint_trans_error_loss(y_true, y_pred):
        y_pred = y_pred * train_joints_data[1] + train_joints_data[0]
        y_pred = y_pred[:, :, :, 3]
        y_true = y_true[:, :, :, 3]
        error_mat = K.square(y_true - y_pred)
        error_mat = K.sum(error_mat, axis = 2)
        error_mat = K.sqrt(error_mat)
        error = K.mean(error_mat)
        return error

    model = Model(inputs=inputs, outputs= x_o)
    opt = Adam(lr=lr) 

    model.compile(optimizer=opt, loss= {'x_output': smoothL1}, loss_weights={'x_output':3}, metrics={'x_output': joint_trans_error_loss})

    return model
if __name__ == "__main__":
    print (get_Denoise_Markers_Model().summary())
