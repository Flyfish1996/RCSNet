
'''
Usage:


'''

import numpy as np
from scipy.io import loadmat
import mne
import os
from scipy.signal import resample

#############################################################################
########################  Motor Imagery   ###################################
#############################################################################
def load_BNCI2014_001(file_path, subject, win_sel):
    '''
    This four class motor imagery data set was originally released as data set 2a of the BCI Competition IV.
    :param file_path:
    :param subject:
    :param win_sel:
    :return:
    '''
    data_subject = {}
    stimcodes = ('769', '770', '771', '772')  # 注意读数据集说明，‘768’才是一个trial的开始，也就意味着对应0时刻
    channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
    tmin = win_sel[0]
    tmax = win_sel[1]

    file_to_load = file_path + '/A0' + str(subject) + 'T.gdf'
    raw_data = mne.io.read_raw_gdf(file_to_load)
    fs = raw_data.info.get('sfreq')
    events, event_ids = mne.events_from_annotations(raw_data)
    stims = [value for key, value in event_ids.items() if key in stimcodes]
    epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                        baseline=None, preload=True, proj=False, reject_by_annotation=False)
    epochs = epochs.drop_channels(channels_to_remove)
    class_return = epochs.events[:, -1] - min(epochs.events[:, -1])
    data_return = epochs.get_data() * 1e6
    data_return = data_return - np.mean(data_return, axis=1, keepdims=True)


    # channel_index = np.array([2,4,6,8,10,12,14,16,18]) - 1
    channel_index = np.arange(0, 22)

    data_subject['train'] = {'X': data_return[:, channel_index, :],
                             'Y': class_return}

    ################################# 测试集数据读取  ####################################################
    file_to_load = file_path + '/A0' + str(subject) + 'E.gdf'
    raw_data = mne.io.read_raw_gdf(file_to_load)
    fs = raw_data.info.get('sfreq')
    labels_mat = loadmat(file_path + '/Data sets 2a_true_labels/' + 'A0' + str(subject) + 'E.mat')
    class_all = labels_mat['classlabel'][:, 0]
    events, event_ids = mne.events_from_annotations(raw_data)
    index_type = [value for key, value in event_ids.items() if key in '783']
    events_index = np.where(events[:, 2] == np.array(index_type))[0]
    events = events[events_index, :]
    events[:, 2] = class_all
    stims = list(np.array(np.unique(class_all)))
    epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                        baseline=None, preload=True, proj=False, reject_by_annotation=False)
    epochs = epochs.drop_channels(channels_to_remove)
    class_return = epochs.events[:, -1] - min(epochs.events[:, -1])
    data_return = epochs.get_data() * 1e6
    data_return = data_return - np.mean(data_return, axis=1, keepdims=True)

    data_subject['test'] = {'X': data_return[:, channel_index, :],
                            'Y': class_return}

    data_subject['fs'] = 250
    data_subject['win_sel'] = win_sel
    return data_subject

def load_data_KU(data_path, subject, training):

    if training:
        if subject < 10:
            file_to_load = data_path + '\sess01_subj0' + str(subject) + '_EEG_MI.mat'
        else:
            file_to_load = data_path + '\sess01_subj' + str(subject) + '_EEG_MI.mat'


        raw_eeg_subject = loadmat(file_to_load)

        x_data1 = raw_eeg_subject['EEG_MI_train']['smt'][0, 0]
        labels1 = raw_eeg_subject['EEG_MI_train']['y_dec'][0, 0][0]
        ch_names = raw_eeg_subject['EEG_MI_train']['chan'][0, 0][0]
        ch_names_list = [str(x[0]) for x in ch_names]

        x_data2 = raw_eeg_subject['EEG_MI_test']['smt'][0, 0]
        labels2 = raw_eeg_subject['EEG_MI_test']['y_dec'][0, 0][0]
        x_data = np.concatenate((x_data1, x_data2), axis=1)
        labels = np.concatenate((labels1, labels2), axis=0)

        class_return = labels - np.min(labels)
        data_return = np.transpose(x_data, axes=[1, 2, 0])

    else:
        if subject < 10:
            file_to_load = data_path + '\sess02_subj0' + str(subject) + '_EEG_MI.mat'
        else:
            file_to_load = data_path + '\sess02_subj' + str(subject) + '_EEG_MI.mat'

        raw_eeg_subject = loadmat(file_to_load)

        x_data1 = raw_eeg_subject['EEG_MI_train']['smt'][0, 0]
        labels1 = raw_eeg_subject['EEG_MI_train']['y_dec'][0, 0][0]
        ch_names = raw_eeg_subject['EEG_MI_train']['chan'][0, 0][0]
        ch_names_list = [str(x[0]) for x in ch_names]

        x_data2 = raw_eeg_subject['EEG_MI_test']['smt'][0, 0]
        labels2 = raw_eeg_subject['EEG_MI_test']['y_dec'][0, 0][0]
        x_data = np.concatenate((x_data1, x_data2), axis=1)
        labels = np.concatenate((labels1, labels2), axis=0)

        class_return = labels - np.min(labels)
        data_return = np.transpose(x_data, axes=[1, 2, 0])
    return data_return, class_return


def load_Lee2019_MI(file_path, subject, win_sel):
    import pickle

    if subject < 10:
        file_to_load = file_path + '/subj0' + str(subject) + '_EEG_MI.pkl'
    else:
        file_to_load = file_path + '/subj' + str(subject) + '_EEG_MI.pkl'
    with open(file_to_load, 'rb') as f:
        data_subject = pickle.load(f)

    # X_train, y_train = load_data_KU(file_path, subject, True)
    # X_test, y_test = load_data_KU(file_path, subject, False)
    #
    # # channel_Sel = np.array([33, 34,
    # #                         13, 14, 15,
    # #                         39,  40, 41]) - 1
    # channel_Sel = np.array([4, 5, 6,
    #                         8, 33, 9, 10, 34, 11,
    #                         35, 13, 36, 14, 37, 15, 38,
    #                         18, 39, 19, 40, 20, 41, 21,
    #                         24, 42, 25, 43, 26]) - 1  # 28个通道
    #
    # from scipy.signal import resample
    # data_subject = {}
    # X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
    # X_test = X_test - np.mean(X_test, axis=1, keepdims=True)
    #
    # data_subject['train'] = {'X': X_train,
    #                          'Y': y_train}
    # data_subject['test'] = {'X': X_test,
    #                         'Y': y_test}
    # data_subject['train']['X'] = data_subject['train']['X'][:, channel_Sel, :]
    # data_subject['test']['X'] = data_subject['test']['X'][:, channel_Sel, :]
    # data_subject['train']['X'] = resample(data_subject['train']['X'], 1000, axis=-1)
    # data_subject['test']['X'] = resample(data_subject['test']['X'], 1000, axis=-1)
    # data_subject['fs'] = 250
    # data_subject['win_sel'] = win_sel

    return data_subject



def load_WBCIC_MI_2C(file_path, subject, win_sel):
    '''

    数据总通道是64，去除眼电等，还剩余59个通道，
    其中PZ作为参考电极，进行重参考，因此余下58个通道

    :param file_path:
    :param subject:
    :param win_sel:
    :return:
    '''


    data_subject = {}
    labels = []
    all_EEG = []
    for sessioni in range(1, 4):
        # 构建subject和session的文件夹名称
        if subject<11:
            mat_file = f"sub-{subject:03d}_ses-{sessioni:02d}_task-motorimagery_eeg.mat"
            mat_file_path = os.path.join(file_path, f"sub-{subject:03d}", f"ses-{sessioni:02d}", 'eeg', mat_file)
        else:
            mat_file = f"sub-{subject+1:03d}_ses-{sessioni:02d}_task-motorimagery_eeg.mat"
            mat_file_path = os.path.join(file_path, f"sub-{subject:03d}", f"ses-{sessioni:02d}", 'eeg', mat_file)

        mat_data = loadmat(mat_file_path)
        si_labels = mat_data['labels'] - np.min(np.unique(mat_data['labels']))
        labels.append(np.squeeze(si_labels, axis=0))
        all_EEG.append(np.transpose(mat_data['data'], axes=[2, 0, 1]))

    channel_name = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
                    'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                    'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
                    'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
                    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
                    'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                    'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2']


    data_subject['train'] = {'X': all_EEG,
                             'Y': labels}

    data_subject['channel'] = channel_name
    data_subject['fs'] = 250
    data_subject['win_sel'] = win_sel

    return data_subject


def load_WBCIC_MI_3C(file_path, subject, win_sel):
    '''

    数据总通道是64，去除眼电等，还剩余59个通道，
    其中PZ作为参考电极，进行重参考，因此余下58个通道

    :param file_path:
    :param subject:
    :param win_sel:
    :return:
    '''


    data_subject = {}
    labels = []
    all_EEG = []
    for sessioni in range(1, 4):
        # 构建subject和session的文件夹名称

        mat_file = f"sub-{subject:03d}_ses-{sessioni:02d}_task-motorimagery_eeg.mat"
        mat_file_path = os.path.join(file_path, f"sub-{subject:03d}", f"ses-{sessioni:02d}", 'eeg', mat_file)

        mat_data = loadmat(mat_file_path)
        si_labels = mat_data['labels'] - np.min(np.unique(mat_data['labels']))
        labels.append(np.squeeze(si_labels))
        all_EEG.append(np.transpose(mat_data['data'], axes=[2, 0, 1]))

    channel_name = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
                    'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                    'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
                    'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
                    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
                    'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                    'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2']


    data_subject['train'] = {'X': all_EEG,
                             'Y': labels}

    data_subject['channel'] = channel_name
    data_subject['fs'] = 250
    data_subject['win_sel'] = win_sel

    return data_subject









if __name__ == "__main__":
    # data_all = load_BNCI2014_001(file_path='H:\All_Dataset\Motor Imagery\BNCI2014_001', subject=1, win_sel=[-1, 4])
    # data_all = load_Lee2019_MI(file_path='H:\All_Dataset\Motor Imagery\Lee2019_MI', subject=1, win_sel=[-1, 4])

    print('test')