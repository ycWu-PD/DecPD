import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    def __init__(self, alpha=None, par=None, a=None, reduction='mean'):
        """
        alpha: 类别权重 (list or Tensor)，用于平衡不平衡数据。可以为 None。
        gamma: Focal Loss 中的聚焦因子
        reduction: 'mean', 'sum' 或 'none'，指定如何聚合损失。
        """
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = par + a
        self.reduction = reduction

        if alpha is not None:
            self.alpha = alpha.clone().detach().float()

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes], 预测的 logits
        # targets: [batch_size], 真实标签

        # 将 logits 转换为概率分布
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # 取正确类别的概率
        pt = (probs * targets_one_hot).sum(dim=1)

        # 计算 Focal Loss 的基本公式部分
        loss = - (1 - pt) ** self.gamma * torch.log(pt)

        # 如果指定 alpha，应用类别权重
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # 获取对应类别的权重
            loss = alpha_t * loss

        # 根据 reduction 参数聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
