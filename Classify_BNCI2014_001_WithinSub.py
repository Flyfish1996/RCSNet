#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Main Training and Cross-Validation Script for RCSNet on BCI Datasets.
"""

import os
import time
import numpy as np
import torch

# 导入预处理与训练模块
from DataPrepare import PrepareData
from prepocess_in_model import filter_EEG
from train_model_withMean_oneStage import train_in_one_fold, EEGDataset

from RCSNet import RCSNet, RCSNetLoss


# ============================================================
# 1. 结果日志写入工具
# ============================================================

def write_to_txt(model_para: dict, train_para: dict, path_set: dict, results_set: dict):
    """保存模型参数与每位受试者的评估结果到 txt 文件"""
    data_sel = path_set['data_sel']
    model_sel = path_set['model_sel']
    epoch_set = path_set['epoch_set']

    all_acc_test = results_set['acc']
    all_precision = results_set['precision']
    all_f1 = results_set['f1']
    all_kappa = results_set['kappa']

    # 自动创建 results 保存目录
    os.makedirs("results", exist_ok=True)
    file_path = os.path.join("results", f"WithSub_{data_sel}_{model_sel}-epoch{epoch_set}.txt")

    with open(file_path, "w") as file:
        file.write(f"Model: {model_sel} | Dataset: {data_sel}\n")
        file.write("=" * 50 + "\n")
        file.write("Model Parameters:\n")
        for key, value in model_para.items():
            file.write(f"  {key}: {value}\n")

        file.write("\nTraining Parameters:\n")
        for key, value in train_para.items():
            file.write(f"  {key}: {value}\n")
        file.write("=" * 50 + "\n\n")

    # 追加写入各个 Subject 的结果
    with open(file_path, "a") as file:
        file.write("--- Accuracy per Subject ---\n")
        for subi in range(len(all_acc_test)):
            file.write(f"Subject {subi + 1:02d} -------- Accuracy: {all_acc_test[subi]:.4f}\n")
        file.write(f"Mean Accuracy: {np.mean(all_acc_test):.4f} +/- {np.std(all_acc_test):.4f}\n\n")

        file.write("--- Precision per Subject ---\n")
        for subi in range(len(all_precision)):
            file.write(f"Subject {subi + 1:02d} -------- Precision: {all_precision[subi]:.4f}\n")
        file.write(f"Mean Precision: {np.mean(all_precision):.4f} +/- {np.std(all_precision):.4f}\n\n")

        file.write("--- Macro F1-Score per Subject ---\n")
        for subi in range(len(all_f1)):
            file.write(f"Subject {subi + 1:02d} -------- F1-Score: {all_f1[subi]:.4f}\n")
        file.write(f"Mean F1-Score: {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}\n\n")

        file.write("--- Cohen's Kappa per Subject ---\n")
        for subi in range(len(all_kappa)):
            file.write(f"Subject {subi + 1:02d} -------- Kappa: {all_kappa[subi]:.4f}\n")
        file.write(f"Mean Kappa: {np.mean(all_kappa):.4f} +/- {np.std(all_kappa):.4f}\n\n")

    print(f"Results successfully saved to: {file_path}")





# ============================================================
# 2. 交叉验证主流程 (Subject-Specific Cross-Validation)
# ============================================================

def cross_validation_bci(data_sel: str, model_sel: str, model_para: dict, train_para: dict):
    """受试者跨 Session/折 评估函数"""
    data_prepare = PrepareData(data_sel=data_sel, win_classify=train_para['win_classify'])

    all_acc_test, all_model_weights = [], []
    all_precision, all_f1, all_kappa = [], [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for subi in data_prepare.sub_name:
        torch.cuda.empty_cache()

        # 加载受试者数据
        train_test = data_prepare.load_data_sub(target_sub=subi)

        train_data = filter_EEG(
            model_sel, data_sel, train_test['train'], train_test['fs'],
            train_test['win_sel'], train_para['win_classify']
        )
        test_data = filter_EEG(
            model_sel, data_sel, train_test['test'], train_test['fs'],
            train_test['win_sel'], train_para['win_classify']
        )

        fs = train_data['fs']
        n_repeat = train_para.get('n_repeat', 1)

        acc_repeats, prec_repeats, f1_repeats, kappa_repeats = [], [], [], []

        for repeati in range(n_repeat):
            print("\n" + "=" * 80)
            print(f"Subject: {subi} | Model: {model_sel} | Repeat: {repeati + 1}/{n_repeat}")
            print("=" * 80)

            # 计算输入时间点数: T = time_window * fs
            time_samples = int((train_para['win_classify'][1] - train_para['win_classify'][0]) * fs)

            # 1. 实例化 RCSNet 模型
            model = RCSNet(
                n_channels=22,
                num_classes=4,
                sample_rate=fs,
                num_views=model_para.get('K', 3),
                rdm_filters=model_para.get('rdm_filters', 20),
                out_time=model_para.get('time_out', 16),
                seq_model_type=model_para.get('seq_model_type', 'gru')
            )

            # 2. 实例化 RCSNet 联合损失函数 (L_total = L_cls + beta * L_CS)
            beta = model_para.get('beta', 0.5)
            loss_func = RCSNetLoss(beta=beta)

            # 3. 运行单折训练
            acc_test, last_model, precision, f1, kappa = train_in_one_fold(
                train_set_all=train_data,
                test_set=test_data,
                model=model,
                loss_func=loss_func,
                train_para=train_para,
                device=device
            )

            acc_repeats.append(acc_test)
            prec_repeats.append(precision)
            f1_repeats.append(f1)
            kappa_repeats.append(kappa)
            all_model_weights.append(last_model)

        all_acc_test.append(np.mean(acc_repeats))
        all_precision.append(np.mean(prec_repeats))
        all_f1.append(np.mean(f1_repeats))
        all_kappa.append(np.mean(kappa_repeats))

    return all_acc_test, all_model_weights, all_precision, all_f1, all_kappa


# ============================================================
# 3. 主程序入口
# ============================================================

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 数据集与实验配置
    data_sel = 'BNCI2014_001'
    model_sel = 'RCSNet'

    # 1. 训练参数配置
    train_para = {
        'win_classify': [0, 4],  # 截取 0~4 秒 MI 信号
        'lr': 0.001,
        'batch_size': 64,
        'n_repeat': 10,
        'first_epochs': 1000,
        'min_train_epoch': 100,
        'patience': 100,
        'stop_criteria': 'accuracy'
    }





    # 2. RCSNet 模型超参数配置
    model_para = {
        'K': 3,  # 节律视角数量
        'rdm_filters': 20,  # 浅层特征通道数
        'time_out': 16,  # DTE 时间池化维度
        'seq_model_type': 'gru',  # 深度时间序列建模器 ('gru', 'lstm', 'transformer', 'cnn')
        'beta': 0.05  # CSL 损失系数
    }

    print(f"Starting Experiments for {model_sel} on {data_sel}...")

    # 执行交叉验证评估
    all_acc_test, _, all_precision, all_f1, all_kappa = cross_validation_bci(
        data_sel=data_sel,
        model_sel=model_sel,
        model_para=model_para,
        train_para=train_para
    )

    path_set = {
        'data_sel': data_sel,
        'model_sel': model_sel,
        'epoch_set': train_para['first_epochs']
    }

    results_set = {
        'acc': all_acc_test,
        'precision': all_precision,
        'f1': all_f1,
        'kappa': all_kappa
    }

    # 保存实验结果到 txt
    write_to_txt(model_para, train_para, path_set, results_set)