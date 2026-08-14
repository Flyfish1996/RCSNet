#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ModelPrepare.py
# @Time      :2024/1/18 9:26
# @Author    :Guoqing Cai

'''
Usage:


'''


def model_prepare(model_sel, model_para):
    '''


    :param model_sel:

        all model: EEGNet: " EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces",
                   DeepNet:
    :param model_para:
    :return:
    '''

    if model_sel in ['EEGNet']:
        from compared_model.EEGNet import EEGNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = EEGNet(nChan=nChan, nTime=nTime, nClass=nClass)

    elif model_sel in ['ShallowNet']:
        from compared_model.ShallowConvNet import ShallowConvNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = ShallowConvNet(num_classes=nClass, chans=nChan, samples=nTime)


    elif model_sel in ['EEGConformer']:
        from compared_model.EEGConformer import Conformer
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        drop_p1 = model_para['dropout1']
        drop_p2 = model_para['dropout2']
        depth = model_para['depth']

        model = Conformer(nChan=nChan, nTime=nTime, nClass=nClass,
                          drop_p1=drop_p1, drop_p2=drop_p2, emb_size=40, depth=depth)


    elif model_sel in ['FBCNet']:
        from compared_model.FBCNet import FBCNet

        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        nBands = model_para['band']
        spa_filter = model_para['spa_filter']
        model = FBCNet(nChan=nChan, nTime=nTime, nClass=nClass, nBands=nBands, m=spa_filter)

    elif model_sel in ['IFNet']:
        from compared_model.IFNet import IFNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        nBands = model_para['band']
        model = IFNet(in_planes=nChan, out_planes=64, kernel_size=63, radix=nBands,
                      patch_size=125, time_points=nTime, num_classes=nClass)

    elif model_sel in ['IFNetV2']:
        from compared_model.IFNet import IFNetV2
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = IFNetV2(in_planes=nChan, out_planes=64, kernel_size=63, radix=2,
                       patch_size=125, time_points=nTime, num_classes=nClass)
        
    elif model_sel in ['TransNet']:
        from compared_model.TransNet import TransNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = TransNet(num_classes=nClass, num_samples=nTime, num_channels=nChan)

    elif model_sel in ['EDPNet']:
        from compared_model.EDPNet import EDPNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = EDPNet(chans=nChan, samples=nTime, num_classes=nClass,)

    elif model_sel in ['LightConvNet']:
        from compared_model.LightConvNet import LightConvNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        nBands = model_para['band']
        model = LightConvNet(num_classes=nClass, num_samples=nTime, num_channels=nChan, num_bands=nBands)

    elif model_sel in ['TSFCNet']:
        from compared_model.TSFCNet import TSFCNet4a
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        nBands = model_para['band']
        model = TSFCNet4a(nChan=nChan, nTime=nTime, nClass=nClass, nBands=nBands)


    elif model_sel in ['MSVTNet']:
        from compared_model.MSVTNet import MSVTNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = MSVTNet(nCh=nChan, nTime=nTime, cls=nClass)

    elif model_sel in ['LMDANet']:
        from compared_model.LMDANet import LMDA
        nChan = model_para['channel']
        nTime = model_para['time']
        nBands = model_para['band']
        nClass = model_para['class']
        model = LMDA(chans=nChan, samples=nTime, num_classes=nClass, depth=nBands,)


    elif model_sel in ['EEGDeformer']:
        from compared_model import EEGDeformer
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        fs = model_para['fs']
        model = EEGDeformer.Deformer(num_chan=nChan, num_time=nTime,
                                     temporal_kernel=11, num_kernel=64,
                                     num_classes=nClass, depth=4, heads=16,
                                     mlp_dim=16, dim_head=16, dropout=0.5)


    elif model_sel in ['SST_DPN']:
        from compared_model import SST_DPN
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        fs = model_para['fs']
        model = SST_DPN.SST_DPN(chans=nChan, samples=nTime, num_classes=nClass)
        loss_func = SST_DPN.Loss_func()






    elif model_sel in ['RCSNet']:
        from RCSNet import RCSNet, Loss_func

        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']

        model = RCSNet(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func(beta=model_para['beta'])

    elif model_sel in ['RCSNet_woRVC']:
        from RCSNet_Ablention import RCSNet_woRVC, Loss_func_woRVC
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = RCSNet_woRVC(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_woRVC()

    elif model_sel in ['RCSNet_woDTE']:
        from RCSNet_Ablention import RCSNet_woDTE, Loss_func_woDTE
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = RCSNet_woDTE(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_woDTE(beta=model_para['beta'])

    elif model_sel in ['RCSNet_woLcs']:
        from RCSNet_Ablention import RCSNet_woLcs, Loss_func_woLcs
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = RCSNet_woLcs(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_woLcs()


    elif model_sel in ['RCSNet_woUAF']:
        from RCSNet_Ablention import RCSNet_woUAF, Loss_func_woUAF
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = RCSNet_woUAF(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_woUAF(beta=model_para['beta'])

    elif model_sel in ['RCNet']:
        from RCSNet_Ablention import RCNet, Loss_func_RCNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = RCNet(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_RCNet(beta=model_para['beta'])

    elif model_sel in ['RSNet']:
        from RCSNet_Ablention import RSNet, Loss_func_RSNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = RSNet(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_RSNet(beta=model_para['beta'])


    elif model_sel in ['EEGNet_backbone']:
        from EEGNet_CSL import EEGNet, Loss_func_backbone
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = EEGNet(nChan=nChan, nTime=nTime, nClass=nClass, F1=model_para['rdm_filters'])
        loss_func = Loss_func_backbone(beta=model_para['beta'])

    elif model_sel in ['EEGNet_CSL']:
        from EEGNet_CSL import EEGNet_CSL, Loss_func_CSL
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = EEGNet_CSL(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_CSL(beta=model_para['beta'])

    elif model_sel in ['EEGNet_CL']:
        from EEGNet_CSL import EEGNet_CL, Loss_func_CL
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = EEGNet_CL(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_CL(beta=model_para['beta'])

    elif model_sel in ['EEGNet_SL']:
        from EEGNet_CSL import EEGNet_SL, Loss_func_SL
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = EEGNet_SL(n_chan=nChan, num_classes=nClass, sample_rate=model_para['fs'], para=model_para)
        loss_func = Loss_func_SL(beta=model_para['beta'])


    return model, loss_func





if __name__ == "__main__":
    run_code = 0
