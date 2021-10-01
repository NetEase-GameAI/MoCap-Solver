import numpy as np
import matplotlib.pyplot as plt

def find_peak_index(dis_img,_height_threshold = 50, _bincount_threshold=40):
    '''find the peaks every row in the distance image(depending on scipy.signal)
    Arguments:
        dis_img {2D numpy array} -- distance image
        _height_threshold {int} -- the threshold of the height of peak
        _bincount_threshold {int} -- the count of peak position which occurs in every row

        _peaks_idx {1D numpy array} -- the peak idx in the markers
        _ret_peaks_max_dis {1D numpy array} -- the max peak value in the markers
    '''
    #extend the both boundaries of distance image for detecting the peak at idx 0 and final idx
    from scipy.signal import find_peaks
    _peak_dis_img = np.c_[dis_img[:,1],dis_img,dis_img[:,-2]]
    _peaks_group = np.array([],dtype=np.int64)
    _peaks_max_dis = np.zeros(shape=_peak_dis_img.shape[1],dtype=np.int64) 
    for idx in range(len(dis_img[0])):
        _peaks,_peaks_val = find_peaks(_peak_dis_img[idx],height=_height_threshold)
        _peaks_max_dis[_peaks] = np.maximum(_peaks_max_dis[_peaks],_peaks_val['peak_heights'])
        _peaks_group = np.append(_peaks_group,_peaks)
    _bincount = np.bincount(_peaks_group)
    _ret_peaks_max_dis = _peaks_max_dis[np.where(_bincount>=_bincount_threshold)]
    _peaks_idx = np.array([p-1 for p in np.where(_bincount>=_bincount_threshold)])[0]
    return _peaks_idx,_ret_peaks_max_dis

def plot_peak(dis_img):
    x = range(0,len(dis_img[0]))
    _peaks_idx,_max_dis = find_peak_index(dis_img)
    for idx in range(len(dis_img[0])):
        plt.plot(x,dis_img[idx]) 
    print (_max_dis)
    plt.scatter(_peaks_idx, _max_dis, marker = 'o', color = 'r', label='outliers', s = 50)
    print (_peaks_idx)
    plt.legend(loc = 'upper right')
    plt.show()


