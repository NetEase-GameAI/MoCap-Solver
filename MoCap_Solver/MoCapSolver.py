import numpy as np

class MoCapSolver():
    def __init__(self):
        return

    def extract_encoder_data(self):
        ############# Extract the data for training Template Skeleton Encoder ######################
        from MoCap_Solver.extract.generate_ts_dataset import generate_ts_dataset
        generate_ts_dataset()
        ############# Extract the data for training Marker Configuration Encoder ###################
        from MoCap_Solver.extract.generate_mc_dataset import generate_mc_dataset
        generate_mc_dataset()
        # ############# Convert the mocap data into temporal window data #############################
        from MoCap_Solver.extract.generate_train_windows_data import generate_train_windows_data
        from MoCap_Solver.extract.generate_test_windows_data import generate_test_windows_data
        generate_train_windows_data()
        generate_test_windows_data()
        ############# Extract the data for training Motion Encoder ##################################
        from MoCap_Solver.extract.generate_motion_dataset import generate_motion_dataset
        generate_motion_dataset()
        return

    def train_encoders(self):
        ##################### Train Template Skeleton Encoder #####################################
        from MoCap_Solver.train.train_template_skeleton import train_template_skeleton
        train_template_skeleton()
        # ##################### Train Marker Configuration Encoder ##################################
        from MoCap_Solver.train.train_marker_configuration import train_marker_configuration
        train_marker_configuration()
        ##################### Train Motion Encoder ################################################
        from MoCap_Solver.train.train_motion import train_motion
        train_motion()
        return

    def evaluate_encoders(self):
        ##################### Evaluate Template Skeleton Encoder ##################################
        print('############## Evaluate Template Skeleton Encoder #######################')
        from MoCap_Solver.evaluate.evaluate_ts_encoder import evaluate_ts_encoder
        evaluate_ts_encoder()
        ##################### Evaluate Marker Configuration Encoder ###############################
        print('############## Evaluate Marker Configuration Encoder ####################')
        from MoCap_Solver.evaluate.evaluate_mc_encoder import evaluate_mc_encoder
        evaluate_mc_encoder()
        ##################### Evaluate Motion Encoder #############################################
        print('############## Evaluate Motion Encoder ##################################')
        from MoCap_Solver.evaluate.evaluate_motion_encoder import evaluate_motion_encoder
        evaluate_motion_encoder()
        return

    def extract_solver_data(self):
        ###################### Extract the data for training MoCap-Solver #########################
        from MoCap_Solver.extract.generate_moc_sol_dataset import generate_moc_sol_dataset
        generate_moc_sol_dataset()
        from MoCap_Solver.extract.statistic import statistic
        statistic()
        return

    def train_solver(self):
        ###################### Train MoCap-Solver ##################################################
        from MoCap_Solver.train.train_mocap_solver import train_mocap_solver
        train_mocap_solver()
        return

    def evaluate_solver(self):
        ###################### Evaluate MoCap-Solver ################################################
        from MoCap_Solver.evaluate.evaluate_sequence import evaluate_sequence
        evaluate_sequence()
        return