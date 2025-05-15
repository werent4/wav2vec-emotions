import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
        if alpha is not None and isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.FloatTensor(alpha)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
            print("Warning: NaN or Inf in ce_loss")
            
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor) and len(self.alpha) > 1:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss *= self.alpha

        return focal_loss.mean()