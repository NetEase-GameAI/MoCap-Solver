import json
import os
import copy
import numpy as np


class JsonClass(object):
    """
    This class is the data structure of Json files.
    """

    def __init__(self, marker_num=56, joint_num=24):
        # this formate is subject to the name of the markers in file which stores the data of original markers
        self._dict_marker = {
            "ARIEL": 0, "C7": 1, "CLAV": 2, "L4": 3, "LANK": 4, "LBHD": 5, "LBSH": 6, "LBWT": 7, "LELB": 8, "LFHD": 9,
    "LFSH": 10, "LFWT": 11, "LHEL": 12, "LHIP": 13,
    "LIEL": 14, "LIHAND": 15, "LIWR": 16, "LKNE": 17, "LKNI": 18, "LMT1": 19, "LMT5": 20, "LMWT": 21, "LOHAND": 22,
    "LOWR": 23, "LSHN": 24, "LTOE": 25, "LTSH": 26,
    "LUPA": 27, "LWRE": 28, "RANK": 29, "RBHD": 30, "RBSH": 31, "RBWT": 32, "RELB": 33, "RFHD": 34, "RFSH": 35,
    "RFWT": 36, "RHEL": 37, "RHIP": 38, "RIEL": 39, "RIHAND": 40,
    "RIWR": 41, "RKNE": 42, "RKNI": 43, "RMT1": 44, "RMT5": 45, "RMWT": 46, "ROHAND": 47, "ROWR": 48, "RSHN": 49,
    "RTOE": 50, "RTSH": 51, "RUPA": 52, "RWRE": 53, "STRN": 54,
    "T10": 55}
        self._dict_joint = {"Pelvis": 0, "L_Hip": 1, "R_Hip": 2, "Spine1": 3, "L_Knee": 4, "R_Knee": 5, "Spine2": 6,
                            "L_Ankle": 7, "R_Ankle": 8, "Spine3": 9, "L_Foot": 10, "R_Foot": 11, "Neck": 12,
                            "L_Collar": 13, "R_Collar": 14, "Head": 15, "L_Shoulder": 16, "R_Shoulder": 17,
                            "L_Elbow": 18, "R_Elbow": 19, "L_Wrist": 20, "R_Wrist": 21, "L_Hand": 22, "R_Hand": 23}
        self._dict_marker.update(dict([reversed(_item) for _item in self._dict_marker.items()]))
        self._dict_joint.update(dict([reversed(_item) for _item in self._dict_joint.items()]))
        self.mrk_num = marker_num
        self.joint_nun = joint_num

    def generate_a_dict(self, seqname, fps, start_time, joint_on=False):
        """
        Generate a dictionary of one frame.
        Args:
            seqname: the name of sequence.
            fps: the frame per second.
            start_time: The start time of the sequence.
            joint_on: determine if adding the information of skeleton joints.

        Returns:
            final_dict: The dictionary of one frame.
        """
        _marker_pos_list = []
        _jt_list = []
        for m_idx in range(self.mrk_num):
            mrk_dict = dict(marker_idx=self._dict_marker[m_idx], x=0.0, z=0.0, y=0.0)
            _marker_pos_list.append(mrk_dict)
        if joint_on:
            for _j_idx in range(self.joint_nun):
                rotation_dict = dict(y=0.0, x=0.0, z=0.0, w=0.0)
                translation_dict = dict(y=0.0, x=0.0, z=0.0)
                _bone_dict = dict(Rotation=rotation_dict, Translation=translation_dict,
                                  BoneName=self._dict_joint[_j_idx])
                _jt_list.append(_bone_dict)
        final_dict = dict(FrameNo=0, BoneInfo=_jt_list, SeqName=seqname, Markerpos=_marker_pos_list, fps=fps,
                          start_time=start_time)
        return final_dict

    def save_markers_result(self, seq_name, result_dir, marker_list, pred_joints_list=[], fps=60., start_time=0.,
                            joints_on=False):
        """
        Save the marker positions and joint positions and rotations into json file.
        Args:
            seq_name: The name of the sequence.
            result_dir: The dir of the result json file.
            marker_list: The positions of markers.
            pred_joints_list: The joint positions and rotations of joints
            fps: frames per second.
            start_time: The start time of the sequence.
            joints_on: determine whether to save the joint positions and rotations.

        """
        marker_list = np.array(marker_list)
        load_dict = [self.generate_a_dict(seq_name, fps, start_time, joints_on)]
        dict1 = load_dict[0]
        ani_frame_num = len(marker_list)
        marker_num = marker_list.shape[1]
        if not (marker_num == self.mrk_num):
            print('Error: the input marker num is not equal to the predefined marker num!')
            return False
        if joints_on:
            if not (len(marker_list) == len(pred_joints_list)):
                print('Error: the input seq num of Joint motion data is not equal to the markers!')
                return False
        load_dict = [copy.deepcopy(dict1) for xxxx in range(ani_frame_num)]

        for frame_idx, mkr in enumerate(load_dict):
            mkr['FrameNo'] = frame_idx

            for marker_idx, pos in enumerate(mkr['Markerpos']):
                pos['x'], pos['y'], pos['z'] = float(
                    marker_list[frame_idx][marker_idx][0]), float(
                    marker_list[frame_idx][marker_idx][1]), float(
                    marker_list[frame_idx][marker_idx][2])
            if joints_on:
                for marker_idx, joints in enumerate(mkr['BoneInfo']):
                    _bone_info = pred_joints_list[frame_idx][marker_idx]
                    joints['Translation']['x'], joints['Translation']['y'], joints['Translation']['z'] = float(
                        _bone_info[0][0]), float(_bone_info[0][1]), float(
                        _bone_info[0][2])
                    joints['Rotation']['w'], joints['Rotation']['x'], joints['Rotation']['y'], joints['Rotation'][
                        'z'] = float(_bone_info[1][0]), float(_bone_info[1][1]), float(_bone_info[1][2]), float(
                        _bone_info[1][3])
            # #save json file
        with open(os.path.join(result_dir, seq_name + '.json'), 'w') as write_f:
            json.dump(load_dict, write_f)

    def read_json(self, file_path, joint_on=False):
        """
        Read the json file and return the marker positions, joint positions and rotations.
        Args:
            file_path: The path of the json file.
            joint_on: determine whether to read the joint information of the json file.

        Returns:
            frame_vector_list: The marker positions of the json file.
            frame_joints_list: The joint positions and rotations of the json file.
            fps: Frames per second.
            start: The start time of the sequence.

        """
        with open(file_path, 'r') as fp:
            load_dict = json.load(fp)
        ani_frame_num = len(load_dict)
        frame_vector_list = list()
        frame_joints_list = list()
        start_time = 0.
        fps = 30.
        for frame_idx, mkr in enumerate(load_dict):
            _markers_list = list()
            _joints_list = list()
            # check
            try:
                mrknum = len(mkr['Markerpos'])
                if mrknum != self.mrk_num:
                    raise Exception('The marker num is not correct! File: ' + file_path)
            except:
                raise Exception('There is no Markerpos info in the file: ' + file_path)
            try:
                for pos in mkr['Markerpos']:
                    _markers_list.append([pos['x'], pos['y'], pos['z']])
            except:
                raise Exception('The marker positions are not correct! File: ' + file_path)
            if frame_idx == 0:
                # try:
                # keys = mkr.keys()
                if 'start_time' in mkr.keys():
                    start_time = mkr['start_time']
                else:
                    start_time = 0.
                if 'fps' in mkr.keys():
                    fps = mkr['fps']
                else:
                    fps = 30.
                # except:
                #     raise Exception('The start time and fps are not correct! File: ' + file_path)
            frame_vector_list.append(_markers_list)
            if (joint_on):
                try:
                    bone_num = len(mkr['BoneInfo'])
                    if bone_num != self.joint_nun:
                        raise Exception('The bone number is not correct! File: ' + file_path)
                except:
                    raise Exception('There is no BoneInfo info in the file: ' + file_path)

                try:
                    for pos in mkr['BoneInfo']:
                        _joints_list.append({'Translation': np.array(
                            [pos['Translation']['x'], pos['Translation']['y'], pos['Translation']['z']]),
                            'Rotation': np.array(
                                [pos['Rotation']['w'], pos['Rotation']['x'], pos['Rotation']['y'],
                                 pos['Rotation']['z']]), 'BoneName': pos['BoneName']}
                        )
                except:
                    raise Exception('The BoneInfo information is not correct! File: ' + file_path)
                frame_vector_list.append(_joints_list)
        return frame_vector_list, frame_joints_list, fps, start_time
