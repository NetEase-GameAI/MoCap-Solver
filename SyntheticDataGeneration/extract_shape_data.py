import json
import os
import numpy as np

def extract_shape_data():
    h = np.load(os.path.join('external', 'smpl_data.npz'))
    maleshapes = h['maleshapes']
    N = maleshapes.shape[0]
    if not os.path.exists(os.path.join('external', 'male_shape_data')):
        os.mkdir(os.path.join('external', 'male_shape_data'))
    output_folder = os.path.join('external', 'male_shape_data')
    for i in range(N):
        L = dict(betas=maleshapes[i].tolist())
        with open(os.path.join(output_folder, '%04d.json'%i), 'w') as write_f:
            json.dump(L, write_f)
    return
