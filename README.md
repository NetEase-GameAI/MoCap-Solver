## 1. Description

This depository contains the sourcecode of MoCap-Solver and the baseline method [Holden 2018].

MoCap-Solver  is  a data-driven-based robust marker denoising method, which takes raw mocap markers as input and outputs corresponding clean markers and skeleton motions. It is based on our work published in SIGGRAPH 2021:

MoCap-Solver: A Neural Solver for Optical Motion Capture Data. 


To configurate this project, run the following commands in Anaconda:
```
conda create -n MoCapSolver pip python=3.6
conda activate MoCapSolver
conda install cudatoolkit=10.1.243
conda install cudnn=7.6.5
conda install numpy=1.17.0
conda install matplotlib=3.1.3
conda install json5=0.9.1
conda install pyquaternion=0.9.9
conda install h5py=2.10.0
conda install tqdm=4.56.0
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install tensorboard==1.15.1
```

## 2. Genrate synthetic dataset

Download the project [SMPLPYTORCH](https://github.com/gulvarol/smplpytorch) with SMPL models downloaded and configurated and put the subfolder "smplpytorch" into the folder "external".

Put the CMU mocap dataset from [AMASS](https://amass.is.tue.mpg.de/) dataset into the folder
```
external/CMU
```
and download the 'smpl_data.npz' from the project [SURREAL](https://github.com/gulvarol/surreal) and put it into "external".

Finally, run the following scripts to generate training dataset and testing dataset.
```
python generate_dataset.py
```

We use a SEED to randomly select train dataset and test dataset and randomly generate noises. 
You can set the number of SEED to generate different datasets.

If you need to generate the training data of your own mocap data sequence, we need three kinds of data for each mocap data sequence: raw data, clean data and the bind pose. 
For each sequence, we should prepare these three kinds of data.
-  **The raw data**: the animations of raw markers that are captured by the optical mocap devices.
- **The clean data**: The corresponding ground-truth skinned mesh animations containing clean markers and skeleton animation. 
**The skeletons of each mocap sequences must be homogenious**, 
that is to say, the numbers of skeletons and the hierarchy must be consistent. 
The clean markers is skinned on the skeletons. **The skinning weights of each mocap sequence must be consistent**.
- **The bind pose**: The bind pose contains the positions of skeletons and the corresponding clean markers, 
as the Section 3 illustrated.

```
M: the marker global positions of cleaned mocap sequence. N * 56 * 3
M1: the marker global positions of raw mocap sequence. N * 56 * 3
J_R: The global rotation matrix of each joints of mocap sequence. N *  24 * 3 * 3
J_t: The joint global positions of mocap sequence. N * 24 * 3
J: The joint positions of T-pose. 24 * 3
Marker_config: The marker configuration of the bind-pose, meaning the local position of each marker with respect to the local frame of each joints. 56 * 24 * 3
```

The order of the markers and skeletons we process in our algorithm is as follows:

```
Marker_order = {
            "ARIEL": 0, "C7": 1, "CLAV": 2, "L4": 3, "LANK": 4, "LBHD": 5, "LBSH": 6, "LBWT": 7, "LELB": 8, "LFHD": 9,
            "LFSH": 10, "LFWT": 11, "LHEL": 12, "LHIP": 13,
            "LIEL": 14, "LIHAND": 15, "LIWR": 16, "LKNE": 17, "LKNI": 18, "LMT1": 19, "LMT5": 20, "LMWT": 21,
            "LOHAND": 22, "LOWR": 23, "LSHN": 24, "LTOE": 25, "LTSH": 26,
            "LUPA": 27, "LWRE": 28, "RANK": 29, "RBHD": 30, "RBSH": 31, "RBWT": 32, "RELB": 33, "RFHD": 34, "RFSH": 35,
            "RFWT": 36, "RHEL": 37, "RHIP": 38, "RIEL": 39, "RIHAND": 40,
            "RIWR": 41, "RKNE": 42, "RKNI": 43, "RMT1": 44, "RMT5": 45, "RMWT": 46, "ROHAND": 47, "ROWR": 48,
            "RSHN": 49, "RTOE": 50, "RTSH": 51, "RUPA": 52, "RWRE": 53, "STRN": 54, "T10": 55} // The order of markers

Skeleton_order = {"Pelvis": 0, "L_Hip": 1, "L_Knee": 2, "L_Ankle": 3, "L_Foot": 4, "R_Hip": 5, "R_Knee": 6, "R_Ankle": 7,
            "R_Foot": 8, "Spine1": 9, "Spine2": 10, "Spine3": 11, "L_Collar": 12, "L_Shoulder": 13, "L_Elbow": 14,
            "L_Wrist": 15, "L_Hand": 16, "Neck": 17, "Head": 18, "R_Collar": 19, "R_Shoulder": 20, "R_Elbow": 21,
            "R_Wrist": 22, "R_Hand": 23}// The order of skeletons.
```


## 3. Train and evaluate 

### 3.1 MoCap-Solver

We can train and evaluate MoCap-Solver by running this script.

```
python train_and_evaluate_MoCap_Solver.py
```

## 3.2 Train and evaluate [Holden 2018]

We also provide our implement version of [Holden 2018], which is the baseline of mocap data solving. 


Once prepared mocap dataset, we can train and evaluate the model [Holden 2018] by running the following script:

```
python train_and_evaluate_Holden2018.py
```


We set the SEED number to 100, 200, 300, 400 respectively, and generated four different datasets. We trained MoCap-Solver and [Holden 2018] on these four datasets and evaluated the errors on the test dataset, the evaluation result is showed on the table.

We gave the pretained corresponding MoCap-Encoders in "models". 

In our original implementation of MoCap-Solver and [Holden 2018] in our paper, markers and skeletons were normalized using the average bone length of the dataset. However, it is problematic when deploying this algorithm to the production environment, since the groundtruth skeletons of test data were actually unknown information. So in our released version, such normalization is removed and the evaluation error is slightly higher than our original implementation since the task has become more complex.

| [Holden 2018]      | SEED100     | SEED200     | SEED300     | SEED400     | 
|----------------|---------------|---------------|---------------:|
| Mean skeleton position error(mm) | 18.7 | 17.9 | 18.1 | 18.2|
| Mean skeleton rotation error(deg)| 7.72| 7.29 | 7.66 | 7.71|
| Mean skeleton rotation error(deg)| 19.86| 18.83 | 19.29 | 19.29|


| Ours      | SEED100     | SEED200     | SEED300     | SEED400     | 
|----------------|---------------|---------------|---------------:|
| Mean skeleton position error(mm) | 10.08 | 10.11 | 10.28 | 9.69|
| Mean skeleton rotation error(deg)| 3.42| 3.44 | 3.47 | 3.32|
| Mean skeleton rotation error(deg)| 10.51| 10.57 | 10.86 | 10.22|



## 4. Typos

The loss function (3-4) of our paper: The first term of this function (i.e. alpha_1*D(Y, X)), X denotes the groundtruth clean markers and Y the predicted clean markers. 


## 5. Citation
If you use this code for your research, please cite our paper:

```
@article{kang2021mocapsolver,
  author = {Chen, Kang and Wang, Yupan and Zhang, Song-Hai and Xu, Sen-Zhe and Zhang, Weidong and Hu, Shi-Min},
  title = {MoCap-Solver: A Neural Solver for Optical Motion Capture Data},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {40},
  number = {4},
  pages = {84},
  year = {2021},
  publisher = {ACM}
}
```