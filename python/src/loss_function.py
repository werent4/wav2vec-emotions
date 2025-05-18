import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha_tensor = torch.FloatTensor(alpha)
                self.register_buffer('alpha_weights', alpha_tensor)
            elif isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha_weights', alpha)
            else:
                self.alpha_scalar = alpha 

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
            print("Warning: NaN or Inf in ce_loss")
            
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if hasattr(self, 'alpha_weights'):
            alpha_t = self.alpha_weights[targets]
            focal_loss = alpha_t * focal_loss
        elif hasattr(self, 'alpha_scalar'):
            focal_loss *= self.alpha_scalar

        return focal_loss.mean()