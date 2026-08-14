import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union


# ============================================================
# 1. Constraint Layers (权重约束层)
# ============================================================

class LinearWithConstraint(nn.Linear):
    """带 Max-Norm 权重约束的全连接层"""

    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    """带 Max-Norm 权重约束的二维卷积层"""

    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


# ============================================================
# 2. Rhythm View Construction (RVC) & Sinc Filtering
# ============================================================

class SincConv1d(nn.Module):
    """
    可学习的带通 Sinc 卷积层 (支持 Depthwise 分组卷积)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 31,
            sample_rate: float = 250.0,
            stride: int = 1,
            padding: Union[str, int] = 'same',
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ):
        super(SincConv1d, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError("SincConv1d: kernel_size 必须为奇数")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        if padding == 'same':
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding

        # 可学习的截止频率 (Hz)
        self.f_low = nn.Parameter(torch.zeros(out_channels, 1))
        self.f_high = nn.Parameter(torch.zeros(out_channels, 1))

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # 默认初始化
        nn.init.uniform_(self.f_low, 4.0, 12.0)
        nn.init.uniform_(self.f_high, 12.0, 40.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) 或 (B, 1, C, T)
        """
        is_4d = False
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # (B, 1, C, T) -> (B, C, T)
            is_4d = True

        device = x.device
        dtype = x.dtype
        nyquist = self.sample_rate / 2.0

        # 确保频率约束：0 <= f_low < f_high <= Nyquist
        f_low = torch.abs(self.f_low)
        f_high = torch.abs(self.f_high)

        f_low, f_high = torch.min(f_low, f_high), torch.max(f_low, f_high)
        f_low = torch.clamp(f_low, 0.0, nyquist - 1.0)
        f_high = torch.max(f_high, f_low + 1.0)
        f_high = torch.clamp(f_high, max=nyquist)

        # 归一化到 [0, 0.5]
        f_low = f_low / self.sample_rate
        f_high = f_high / self.sample_rate

        t = torch.linspace(
            -(self.kernel_size - 1) / 2.0,
            (self.kernel_size - 1) / 2.0,
            steps=self.kernel_size,
            device=device,
            dtype=dtype
        ).view(1, -1)
        t[t == 0] = 1e-8  # 避免除以零

        # 构造高通与低通滤波器 -> 带通滤波器
        high_pass = 2 * f_high * torch.sin(2 * math.pi * f_high * t) / (2 * math.pi * f_high * t)
        low_pass = 2 * f_low * torch.sin(2 * math.pi * f_low * t) / (2 * math.pi * f_low * t)
        band_pass = high_pass - low_pass

        # 加 Hamming 窗
        window = torch.hamming_window(self.kernel_size, device=device, dtype=dtype)
        band_pass = band_pass * window

        # 归一化滤波器能量
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)
        sinc_filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        y = F.conv1d(
            x, sinc_filters, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups
        )

        if is_4d:
            y = y.unsqueeze(1)  # (B, C, T) -> (B, 1, C, T)

        return y

    def get_freq_bands(self) -> Dict[str, torch.Tensor]:
        """获取当前层学习到的频率边界 (Hz)"""
        nyquist = self.sample_rate / 2.0
        f_low = torch.clamp(torch.abs(self.f_low), 0.0, nyquist - 1.0)
        f_high = torch.clamp(torch.abs(self.f_high), f_low.item() + 1.0 if f_low.numel() == 1 else 1.0, nyquist)
        return {"f_low": f_low.detach().cpu(), "f_high": f_high.detach().cpu()}


class RVC(nn.Module):
    """
    Rhythm View Construction (RVC) 模块
    将原始 EEG 信号划分为 K 个可学习的节律视角视角视图
    """

    def __init__(
            self,
            num_views: int = 3,
            n_channels: int = 22,
            sample_rate: float = 250.0,
            fmin: float = 4.0,
            fmax: float = 40.0,
            kernel_size: int = 31,
    ):
        super().__init__()
        assert num_views >= 1, "num_views (K) 必须 >= 1"
        assert sample_rate > 0, "采样率 sample_rate 必须 > 0"
        assert 0 <= fmin < fmax <= sample_rate / 2, "需满足 0 <= fmin < fmax <= Nyquist"

        self.K = num_views
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.kernel_size = kernel_size

        step_k = (fmax - fmin) / num_views
        self.view_branches = nn.ModuleList()

        for k in range(num_views):
            sub_fmin = fmin + k * step_k
            sub_fmax = fmin + (k + 1) * step_k

            branch_sinc = SincConv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=self.kernel_size,
                sample_rate=self.sample_rate,
                groups=n_channels  # 深度可分离卷积
            )
            branch_sinc.f_low.data = torch.full((n_channels, 1), sub_fmin)
            branch_sinc.f_high.data = torch.full((n_channels, 1), sub_fmax)

            self.view_branches.append(branch_sinc)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        输入: x: [B, 1, C, T]
        输出: List of K * [B, 1, C, T]
        """
        assert x.ndim == 4 and x.shape[1] == 1, f"输入形状应为 [B, 1, C, T]，当前为 {x.shape}"
        return [branch(x) for branch in self.view_branches]

    def get_all_freq_bands(self) -> List[Dict]:
        all_views_info = []
        for k, branch in enumerate(self.view_branches):
            bands = branch.get_freq_bands()
            bands['view_index'] = k
            all_views_info.append(bands)
        return all_views_info


# ============================================================
# 3. Dual-Path Rhythm Encoding (STConvEncoder)
# ============================================================

class STConvEncoder(nn.Module):
    """
    Shallow Spatio-Temporal Convolutional Encoder
    对应论文 Section III-C 中的双路径浅层空时编码器
    包含：Spatial Conv (C,1) -> Temporal Conv (1,24) -> BN -> ELU -> Dropout -> AvgPool (1,64)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32, n_channels: int = 22):
        super(STConvEncoder, self).__init__()
        self.conv = nn.Sequential(
            Conv2dWithConstraint(in_channels, out_channels, (n_channels, 1), stride=(1, 1)),
            Conv2dWithConstraint(out_channels, out_channels, (1, 24), stride=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d((1, 64), stride=(1, 24)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================
# 4. Deep Temporal Encoding (DTE)
# ============================================================

class DeepTemporalEncoder(nn.Module):
    """
    Deep Temporal Encoder (DTE) 模块
    对应论文 Section III-E，包含 1x1 Pointwise 卷积、序列建模器 (GRU/LSTM/Transformer/CNN) 与自适应时间池化
    """

    def __init__(
            self,
            in_channels: int = 32,
            out_channels: int = 32,
            out_time: int = 1,
            seq_model_type: str = 'gru'
    ):
        super(DeepTemporalEncoder, self).__init__()
        self.seq_model_type = seq_model_type.lower()

        # Stage 1: Pointwise Feature Projection
        self.conv_block = nn.Sequential(
            Conv2dWithConstraint(in_channels, out_channels, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        # Stage 3: Adaptive Temporal Pooling
        self.pooling_block = nn.AdaptiveAvgPool2d((1, out_time))

        # Stage 2: Temporal Sequence Learner
        if self.seq_model_type == 'gru':
            self.sequence_learner = nn.GRU(
                input_size=out_channels,
                hidden_size=out_channels // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif self.seq_model_type == 'lstm':
            self.sequence_learner = nn.LSTM(
                input_size=out_channels,
                hidden_size=out_channels // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        elif self.seq_model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=out_channels,
                nhead=1,
                batch_first=True
            )
            self.sequence_learner = nn.TransformerEncoder(encoder_layer, num_layers=1)
        elif self.seq_model_type == 'cnn':
            self.sequence_learner = nn.Sequential(
                LinearWithConstraint(out_channels, out_channels),
                nn.ELU(),
                nn.Dropout(0.3),
                LinearWithConstraint(out_channels, out_channels),
            )
        else:
            raise ValueError(f"不支持的 seq_model_type: {seq_model_type}，可选 ['gru', 'lstm', 'transformer', 'cnn']")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, 1, T]
        x = self.conv_block(x)  # [B, C_out, 1, T]
        x_seq = x.squeeze(2).permute(0, 2, 1)  # [B, T, C_out]

        if self.seq_model_type in ['gru', 'lstm']:
            seq_out, _ = self.sequence_learner(x_seq)
        elif self.seq_model_type in ['transformer', 'cnn']:
            seq_out = self.sequence_learner(x_seq)

        seq_out = seq_out.permute(0, 2, 1).unsqueeze(2)  # [B, C_out, 1, T]
        out = self.pooling_block(seq_out)  # [B, C_out, 1, out_time]
        return out


# ============================================================
# 5. Classifier Head
# ============================================================

class ClassifierModule(nn.Module):
    """
    分类头组件
    返回原始 Logits (配合 CrossEntropyLoss 使用，有利于 UAF 计算不确定性)
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classify_layer = nn.Sequential(
            LinearWithConstraint(feature_dim, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            LinearWithConstraint(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_layer(x)


# ============================================================
# 6. RCSNet Main Model
# ============================================================

class RCSNet(nn.Module):
    """
    Rhythm Commonality-Specificity Network (RCSNet)
    包含 RVC 视图构建、双路径节律编码、深度时间编码与多分支分类头
    """

    def __init__(
            self,
            n_channels: int = 22,
            num_classes: int = 4,
            sample_rate: float = 250.0,
            num_views: int = 3,
            rdm_filters: int = 20,
            out_time: int = 16,
            seq_model_type: str = "gru",

            fmin: float = 4.0,
            fmax: float = 40.0,
            kernel_size: int = 31,
            para: Optional[Dict] = None
    ):
        super(RCSNet, self).__init__()

        # 若传入 para 字典，则从中提取参数覆盖默认设置
        if para is not None:
            num_views = int(para.get("K", num_views))
            rdm_filters = int(para.get("rdm_filters", rdm_filters))
            out_time = int(para.get("time_out", out_time))
            seq_model_type = para.get("seq_model_type", seq_model_type)

        self.K = num_views
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.rdm_filters = rdm_filters
        self.out_time = out_time

        # 1. Rhythm View Construction (RVC)
        self.rhythm_filter = RVC(
            num_views=self.K,
            n_channels=self.n_channels,
            sample_rate=self.sample_rate,
            fmin=fmin,
            fmax=fmax,
            kernel_size=kernel_size
        )

        # 2. Dual-Path Rhythm Encoding (Specific & Common Encoders)
        self.specific_encoders = nn.ModuleList([
            STConvEncoder(1, rdm_filters, n_channels) for _ in range(self.K)
        ])
        self.common_encoders = nn.ModuleList([
            STConvEncoder(1, rdm_filters, n_channels) for _ in range(self.K)
        ])

        # 3. Deep Temporal Encoders (DTE)
        self.specific_dtes = nn.ModuleList([
            DeepTemporalEncoder(rdm_filters, rdm_filters, out_time, seq_model_type)
            for _ in range(self.K)
        ])
        self.common_dte = DeepTemporalEncoder(rdm_filters, rdm_filters, out_time, seq_model_type)

        # 4. Classifiers
        final_dim = rdm_filters * out_time
        self.specific_classifiers = nn.ModuleList([
            ClassifierModule(final_dim, num_classes) for _ in range(self.K)
        ])
        self.common_classifier = ClassifierModule(final_dim, num_classes)

        # 权重初始化
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        rhythm_signals = self.rhythm_filter(x)
        specific_features = []
        common_features = []

        for k in range(self.K):
            x_k = rhythm_signals[k]
            s_k = self.specific_encoders[k](x_k)
            c_k = self.common_encoders[k](x_k)
            specific_features.append(s_k)
            common_features.append(c_k)

        # 共性特征聚合
        common_fused = sum(common_features)

        specific_embeddings = []
        specific_logits = []

        for k in range(self.K):
            z_s_k = self.specific_dtes[k](specific_features[k])
            logit_s_k = self.specific_classifiers[k](z_s_k.flatten(start_dim=1))
            specific_embeddings.append(z_s_k)
            specific_logits.append(logit_s_k)

        common_embedding = self.common_dte(common_fused)
        common_logit = self.common_classifier(common_embedding.flatten(start_dim=1))

        return {
            "specific_features": specific_features,
            "common_features": common_features,
            "common_fused": common_fused,
            "specific_embeddings": specific_embeddings,
            "common_embedding": common_embedding,
            "specific_logits": specific_logits,
            "common_logit": common_logit,
        }


# ============================================================
# 7. Commonality-Specificity Loss & Uncertainty Fusion
# ============================================================

def compute_commonality_specificity_loss(
        specific_features: List[torch.Tensor],
        common_features: List[torch.Tensor],
        eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 Commonality-Specificity Loss (L_CS)
    返回: (loss_cs, loss_con, loss_div)
    """
    K = len(common_features)
    if K < 1:
        raise ValueError("要求 K >= 1")

    # 展平特征: (K, B, D)
    f_s = torch.stack(specific_features, dim=0).flatten(start_dim=2)
    f_c = torch.stack(common_features, dim=0).flatten(start_dim=2)

    # L2 归一化
    f_s_norm = F.normalize(f_s, p=2, dim=-1, eps=eps)
    f_c_norm = F.normalize(f_c, p=2, dim=-1, eps=eps)

    # 1. Common-Specific Diversity Loss (L_div) - 公式 (9)
    # 计算同一 view 内 c_k 和 s_k 的绝对余弦相似度
    div_cos = (f_s_norm * f_c_norm).sum(dim=-1)  # (K, B)
    loss_div = torch.abs(div_cos).mean()

    # 2. Cross-Rhythm Consistency Loss (L_con) - 公式 (8)
    # 计算不同 view 之间 c_i 和 c_j 的余弦相似度
    if K > 1:
        f_c_perm = f_c_norm.permute(1, 0, 2)  # (B, K, D)
        sim_matrix = torch.abs(torch.matmul(f_c_perm, f_c_perm.transpose(-1, -2)))  # (B, K, K)

        row, col = torch.triu_indices(K, K, offset=1, device=f_c.device)
        pair_sims = sim_matrix[:, row, col]  # (B, Pairs)
        loss_con = (1.0 - pair_sims).mean()
    else:
        loss_con = torch.tensor(0.0, device=f_c.device)

    # 3. Overall Commonality-Specificity Loss (L_CS) - 公式 (10)
    loss_cs = 0.5 * (loss_con + loss_div)

    return loss_cs, loss_con, loss_div


def uncertainty_aware_fusion(
        logits_list: List[torch.Tensor],
        temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uncertainty-Aware Fusion (UAF) 推理阶段融合策略
    返回:
        - final_probs: 最终预测概率分布 [B, Q]
        - normalized_weights: 各分支贡献权重 [K+1, B]
    """
    # 1. 计算各分支预测概率与熵 (Predictive Entropy)
    probs_list = [F.softmax(logits / temperature, dim=-1) for logits in logits_list]  # List of [B, Q]

    # Entropy: H_v = - sum(p * log(p))
    entropies = [-torch.sum(p * torch.log(p + 1e-10), dim=-1) for p in probs_list]  # List of [B]

    # 2. 计算置信度得分: exp(-H_v) 并归一化 -
    confidence_scores = torch.exp(-torch.stack(entropies, dim=0))  # [K+1, B]
    normalized_weights = confidence_scores / (torch.sum(confidence_scores, dim=0, keepdim=True) + 1e-10)  # [K+1, B]

    # 3. 加权融合概率 -
    stacked_probs = torch.stack(probs_list, dim=0)  # [K+1, B, Q]
    weighted_probs = stacked_probs * normalized_weights.unsqueeze(-1)  # [K+1, B, Q]
    final_probs = torch.sum(weighted_probs, dim=0)  # [B, Q]

    return final_probs, normalized_weights


# ============================================================
# 8. RCSNet Total Loss Module
# ============================================================

class RCSNetLoss(nn.Module):
    """
    RCSNet 联合训练总损失模块
    L_total = L_cls + beta * L_CS
    """

    def __init__(self, beta: float = 0.5):
        super(RCSNetLoss, self).__init__()
        self.beta = beta

    def forward(
            self,
            results: Dict[str, Union[List[torch.Tensor], torch.Tensor]],
            targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        common_logit = results["common_logit"]  # [B, num_classes]
        specific_logits = results["specific_logits"]  # List of K * [B, num_classes]
        specific_features = results["specific_features"]  # List of K * [B, ...]
        common_features = results["common_features"]  # List of K * [B, ...]

        K = len(specific_logits)

        # 1. 分类损失 (L_cls)
        loss_c_cls = F.cross_entropy(common_logit, targets)
        loss_s_cls = sum([F.cross_entropy(l_s, targets) for l_s in specific_logits])
        loss_cls = (loss_c_cls + loss_s_cls) / (K + 1)

        # 2. 解耦损失 (L_CS)
        loss_cs, loss_con, loss_div = compute_commonality_specificity_loss(
            specific_features=specific_features,
            common_features=common_features
        )

        # 3. 总损失 (L_total)
        loss_total = loss_cls + self.beta * loss_cs

        # UAF 融合决策概率 (用于评估/验证)
        all_logits = specific_logits + [common_logit]
        final_probs, fusion_weights = uncertainty_aware_fusion(all_logits)

        metrics = {
            "loss_total": loss_total.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_cs": loss_cs.detach(),
            "loss_con": loss_con.detach(),
            "loss_div": loss_div.detach(),
            "final_probs": final_probs,
            "fusion_weights": fusion_weights
        }

        return loss_total, metrics


# ============================================================
# 9. Main Function Test Block
# ============================================================

if __name__ == "__main__":
    print("================== RCSNet 网络测试 ==================")

    # 模拟输入参数: Batch_size=4, 1通道, 22电极, 1000时间点 (4秒, 250Hz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_x = torch.randn(4, 1, 22, 1000).to(device)
    dummy_y = torch.randint(0, 4, (4,)).to(device)

    # 1. 使用默认参数初始化
    model = RCSNet(n_channels=22, num_classes=4, sample_rate=250.0).to(device)
    criterion = RCSNetLoss(beta=0.5)

    # 2. 前向传播
    outputs = model(dummy_x)
    loss, metrics = criterion(outputs, dummy_y)

    print("输入数据维度:", dummy_x.shape)
    print("公共分支 Logits 维度:", outputs["common_logit"].shape)
    print("特定分支 Logits 数量:", len(outputs["specific_logits"]))
    print("总损失 L_total:", loss.item())
    print("分类损失 L_cls:", metrics["loss_cls"].item())
    print("解耦损失 L_CS:", metrics["loss_cs"].item())
    print("UAF 融合后的预测概率维度:", metrics["final_probs"].shape)
    print("================== 测试成功 ==================")