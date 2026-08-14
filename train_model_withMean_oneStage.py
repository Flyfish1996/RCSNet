#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Model Training and Evaluation Script for RCSNet.
Supports cross-validation/single-fold training, early stopping, and metric tracking.
"""

import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)


# ============================================================
# 1. Dataset Wrapper
# ============================================================

class EEGDataset(Dataset):
    """EEG Tensor Dataset Wrapper"""

    def __init__(self, data_tensor: torch.Tensor, target_tensor: torch.Tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self) -> int:
        return self.data_tensor.size(0)

    def __getitem__(self, index: int):
        return self.data_tensor[index], self.target_tensor[index]


# ============================================================
# 2. Single Fold Training Procedure
# ============================================================

def train_in_one_fold(
        train_set_all: dict,
        test_set: dict,
        model: torch.nn.Module,
        loss_func: torch.nn.Module,
        train_para: dict,
        device: torch.device = None
):
    """
    单折/单 Session 训练主流程

    参数:
        train_set_all: 包含 'X' 和 'Y' 训练数据的字典
        test_set: 包含 'X' 和 'Y' 测试数据的字典
        model: RCSNet 模型实例
        loss_func: RCSNetLoss 实例
        train_para: 训练超参数字典
        device: 计算设备 (如 None 则自动检测)
    """
    # 提取训练参数 (自带默认回退值，防止缺键报错)
    batch_size = train_para.get('batch_size', 64)
    first_epochs = train_para.get('first_epochs', 1000)
    min_train_epoch = train_para.get('min_train_epoch', 20)
    patience = train_para.get('patience', 100)
    stop_criteria = train_para.get('stop_criteria', 'accuracy')
    lr = train_para.get('lr', 1e-3)
    weight_decay = train_para.get('weight_decay', 1e-4)

    # 设备选择
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建 DataLoader (测试集不打乱 shuffle=False)
    train_dataset = EEGDataset(train_set_all['X'], train_set_all['Y'])
    test_dataset = EEGDataset(test_set['X'], test_set['Y'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    loss_func = loss_func.to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 指标跟踪变量
    best_acc = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    best_kappa = 0.0
    best_loss = float('inf')
    best_epoch = 0
    best_model_weights = None

    print(f"Start Training on device: {device} | Total Epochs: {first_epochs}")

    for epoch in range(first_epochs):
        epoch_start_time = time.time()

        # ------------------ 训练阶段 ------------------
        model.train()
        loss_func.train()

        pred_train, labels_train = [], []
        loss_train_list = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            results = model(inputs)

            # 对齐 RCSNetLoss 接口: 返回 (loss_total, metrics)
            loss_total, metrics = loss_func(results, targets)

            loss_total.backward()
            optimizer.step()

            # 使用 UAF 融合后的概率进行训练集预测统计
            result_prob = metrics["final_probs"]
            pred_labels_batch = torch.argmax(result_prob, dim=1)

            pred_train.append(pred_labels_batch.cpu().numpy())
            labels_train.append(targets.cpu().numpy())
            loss_train_list.append(loss_total.item())

        pred_train = np.concatenate(pred_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        acc_train = accuracy_score(labels_train, pred_train)
        loss_train = np.mean(loss_train_list)

        # ------------------ 测试/评估阶段 ------------------
        model.eval()
        loss_func.eval()

        pred_test, labels_test = [], []
        loss_test_list = []

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                results = model(inputs)
                loss_total, metrics = loss_func(results, targets)

                result_prob = metrics["final_probs"]
                pred_val_batch = torch.argmax(result_prob, dim=1)

                pred_test.append(pred_val_batch.cpu().numpy())
                labels_test.append(targets.cpu().numpy())
                loss_test_list.append(loss_total.item())

        pred_test = np.concatenate(pred_test, axis=0)
        labels_test = np.concatenate(labels_test, axis=0)
        loss_test = np.mean(loss_test_list)

        acc_test = accuracy_score(labels_test, pred_test)
        precision = precision_score(labels_test, pred_test, average='macro', zero_division=0)
        f1 = f1_score(labels_test, pred_test, average='macro', zero_division=0)
        kappa = cohen_kappa_score(labels_test, pred_test)

        if stop_criteria == 'accuracy':
            if ((acc_test > best_acc) or (acc_test == best_acc and loss_test < best_loss)) and (
                    epoch >= min_train_epoch):
                best_acc = acc_test
                best_precision = precision
                best_f1 = f1
                best_kappa = kappa
                best_loss = loss_test
                best_epoch = epoch + 1
                best_model_weights = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch [{epoch + 1:03d}/{first_epochs:03d}] | "
            f"Train Loss: {loss_train:.4f} | Train Acc: {acc_train:.4f} | "
            f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test:.4f} | "
            f"Best Test Acc: {best_acc:.4f} (Epoch {best_epoch}) | "
            f"Time: {epoch_time:.2f}s"
        )

        # Patience 判据触发早停
        if (epoch + 1 - best_epoch >= patience) or (best_acc >= 0.9999):
            print(f"Early stopping triggered at Epoch {epoch + 1}.")
            break

    print("=" * 80)
    print(f"Training Complete! Best Test Acc: {best_acc:.4f} at Epoch {best_epoch}")
    print("=" * 80)

    return best_acc, best_model_weights, best_precision, best_f1, best_kappa


# ============================================================
# 3. Quick Functional Test
# ============================================================

if __name__ == "__main__":
    print("Testing train_model script syntax...")