from MoCap_Solver.MoCapSolver import MoCapSolver

model_class = MoCapSolver()

print('1. Start extracting traning data of MoCap-Encoders!')
model_class.extract_encoder_data()

print('2. Start training MoCap-Encoders!')
model_class.train_encoders()

print('3. Evaluate MoCap-Encoders!')
model_class.evaluate_encoders()

print('3. Start extracting mocap-solver data')
model_class.extract_solver_data()

print('4. Start training Mocap-Solver')
model_class.train_solver()

print('5. Start evaluate MoCap-Solver')
model_class.evaluate_solver()

