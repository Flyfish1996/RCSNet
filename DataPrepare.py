#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DataPrepare.py
# @Time      :2024/1/5 16:24
# @Author    :Guoqing Cai

'''
Usage:
'''



from dataload_func import *
import numpy as np






# data_path = {'path',
#              'path',
#              'path',
#              'path',
#              }

data_path = {'BNCI2014_001': '/home/caiguoqing/All_Data/Motor Imagery/BNCI2014_001',
             'WBCIC_MI_2C': '/home/caiguoqing/All_Data/Motor Imagery/WBCIC_MI/derivatives/2C dataset_processeddata',
             'WBCIC_MI_3C': '/home/caiguoqing/All_Data/Motor Imagery/WBCIC_MI/derivatives/3C dataset_processeddata',
             'Lee2019_MI': '/home/caiguoqing/All_Data/Motor Imagery/Lee2019_MI/Lee2019_MI',
             }

subject_name = {'BNCI2014_001': np.arange(1, 10, 1),
                'WBCIC_MI_2C':  np.arange(1, 52, 1),
                'WBCIC_MI_3C':  np.arange(1, 12, 1),
                'Lee2019_MI': np.arange(1, 55, 1),
                }


win_sel = {'BNCI2014_001': [-2, 5],
           'Lee2019_MI': [0, 4],
           'WBCIC_MI_2C': [0, 4],
           'WBCIC_MI_3C': [0, 4],
           }


data_load_funciton = {'BNCI2014_001': load_BNCI2014_001,
                      'Lee2019_MI': load_Lee2019_MI,
                      'WBCIC_MI_2C': load_WBCIC_MI_2C,
                      'WBCIC_MI_3C': load_WBCIC_MI_3C,
                      }

class PrepareData:
    def __init__(self, data_sel, win_classify):
        self.data_sel = data_sel
        self.data_path = data_path[data_sel]
        self.sub_name = subject_name[data_sel]
        self.win_classify = win_classify
        self.win_sel = win_sel[data_sel]

    def load_data_sub(self, target_sub):
        selected_function = data_load_funciton[self.data_sel]
        data_subject = selected_function(file_path=self.data_path, subject=target_sub, win_sel=self.win_sel)
        return data_subject




if __name__ == "__main__":

    data_prepare = PrepareData(data_sel='HandWriting', win_classify=[0, 0.5])
    data_prepare.load_data_sub(target_sub=data_prepare.sub_name[5])

    print(data_prepare)
    print(data_sub)



