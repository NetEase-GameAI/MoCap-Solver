import os
os.environ['KERAS_BACKEND']= 'tensorflow'
from keras.models import Model
from keras.layers import Input, Add,Dropout,PReLU,LeakyReLU,Conv2D,Conv2DTranspose,Concatenate, Lambda,  Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation,Flatten,Reshape
from keras import regularizers
from keras import backend as K
import keras
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np

from utils.utils import _dict_marker, _dict_joints, MARKERNUM, JOINTNUM


def computeMarker(jtransform,mrkconfig, weights, train_joints_data):
    
    jtransform = Reshape((JOINTNUM, 3, 4))(jtransform)
    jtransform = Lambda(lambda x: x * train_joints_data[1] + train_joints_data[0])(jtransform)
    jtrans = Lambda(lambda x: x[:,:,:,3])(jtransform)
    jtrans = Reshape((1, JOINTNUM, 3))(jtrans)
    jtrans = Lambda(lambda x:K.tile(x, (1,MARKERNUM,1,1)))(jtrans)
    jrot = Lambda(lambda x: x[:,:,:,0:3])(jtransform)
    jrot = Reshape((1, JOINTNUM, 3, 3))(jrot)
    jrot = Lambda(lambda x:K.tile(x, (1, MARKERNUM, 1, 1, 1)))(jrot)

    weight = weights.reshape(1, MARKERNUM, JOINTNUM, 1)

    mrkconfig = Reshape((MARKERNUM, JOINTNUM, 1, 3))(mrkconfig)
    mrkconfig = Lambda(lambda x:K.tile(x, (1, 1, 3, 1)))(mrkconfig)
    offset = Lambda(lambda x:x * mrkconfig)(jrot)   # , Marker * join * 3*3
    offset = Lambda(lambda x:K.sum(x, axis = 4))(offset)
    offset = Add()([offset, jtrans])   #  Marker * join * 3
    offset = Lambda(lambda x:x * weight)(offset) # Marker * join * 1
    offset = Lambda(lambda x:K.sum(x, axis = 2), name = 'x_output2')(offset)
    return offset


def get_Denoise_Markers_Model(input_channel_num = 2, marker_nums=MARKERNUM, joints_nums=JOINTNUM, lr = 0.0001, weights = None):

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
    input_mrk_config = Input(shape=(marker_nums,joints_nums,3))
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
    
    x_o0 = Reshape((joints_nums, 3, 4), input_shape = (joints_nums * 3 * 4,), name = 'x_output')(x_o)
    x_o1 = Reshape((joints_nums, 3, 4), input_shape = (joints_nums * 3 * 4,), name = 'x_output1')(x_o)
    x_o3 = computeMarker(x_o, input_mrk_config, weights, train_joints_data) 
    
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
    def marker_error_loss(y_true, y_pred):
        error_mat = K.square(y_true - y_pred)
        error_mat = K.sum(error_mat, axis = 2)
        error_mat = K.sqrt(error_mat)
        error = K.mean(error_mat)
        return error

    # angle_dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2))
    def joint_angle_error_loss(y_true, y_pred):
        y_pred = y_pred * train_joints_data[1] + train_joints_data[0]
        y_pred = y_pred[:, :, :, 0:3]
        y_true = y_true[:, :, :, 0:3]

        y_pred1 = y_pred[:,  :, 0, :]
        y_pred2 = y_pred[:,  :, 1, :]
        y_pred3 = y_pred[:,  :, 2, :]
        y_pred1 = K.reshape(y_pred1, (-1, JOINTNUM, 1, 3))
        y_pred2 = K.reshape(y_pred2, (-1, JOINTNUM, 1, 3))
        y_pred3 = K.reshape(y_pred3, (-1, JOINTNUM, 1, 3))
        y_pred1 = K.tile(y_pred1, (1, 1, 3, 1))
        y_pred2 = K.tile(y_pred2, (1, 1, 3, 1))
        y_pred3 = K.tile(y_pred3, (1, 1, 3, 1))
        z_pred1 = y_true * y_pred1
        z_pred2 = y_true * y_pred2
        z_pred3 = y_true * y_pred3
        z_pred1 = K.sum(z_pred1, axis = 3)
        z_pred2 = K.sum(z_pred2, axis = 3)
        z_pred3 = K.sum(z_pred3, axis = 3)
        z_pred1 = K.reshape(z_pred1, (-1, JOINTNUM, 3, 1))
        z_pred2 = K.reshape(z_pred2, (-1, JOINTNUM, 3, 1))
        z_pred3 = K.reshape(z_pred3, (-1, JOINTNUM, 3, 1))
        z_pred = K.concatenate([z_pred1, z_pred2, z_pred3])
        z_pred_trace = tf.linalg.trace(z_pred) 
        z_pred_trace = (z_pred_trace - 1.)/2.000000000
        z_pred_trace = K.clip(z_pred_trace, -1.0, 1.0)
        z_pred_trace = tf.acos(z_pred_trace) 
        z_pred_trace = K.abs(z_pred_trace * 180 /3.141592653)   #* 10 #
        error = K.mean(z_pred_trace)
        return error

    model = Model(inputs=[inputs,input_mrk_config], outputs= [x_o0, x_o1, x_o3])
    opt = Adam(lr=lr) 

    model.compile(optimizer=opt, loss= {'x_output': smoothL1, 'x_output1': smoothL1, 'x_output2': marker_error_loss}, loss_weights={'x_output':3, 'x_output1':0,'x_output2':0}, metrics={'x_output': joint_trans_error_loss, 'x_output1': joint_angle_error_loss,  'x_output2': marker_error_loss})
    return model
if __name__ == "__main__":
    print (get_Denoise_Markers_Model().summary())
