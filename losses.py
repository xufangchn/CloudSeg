import torch
import torch.nn as nn
import torch.nn.functional as F

class ECCharbonnierLoss(nn.Module):
    def __init__(self):
        super(ECCharbonnierLoss, self).__init__()
        self._alpha = 0.45
        self._epsilon = 1e-3
        self.EC_weight = 5.
    
    def forward(self, pred_cloudfree, cloudfree, cloudmask):
        cloudmask = cloudmask.unsqueeze(1).expand(-1, pred_cloudfree.shape[1], -1, -1)
        weight = torch.ones_like(cloudmask) + self.EC_weight*cloudmask
        loss = torch.mean( weight*torch.pow( ( (pred_cloudfree-cloudfree) ** 2 + self._epsilon ** 2 ), self._alpha ) )
        return loss


class MaskHint(nn.Module):
	'''
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	'''
	def __init__(self):
		super(MaskHint, self).__init__()

	def forward(self, fm_s, fm_t, mask):
		loss = F.mse_loss(fm_s, fm_t, reduction='none')
		mask = torch.ones_like(mask) - mask
		mask = mask.unsqueeze(1)
		loss = torch.mean(loss * mask)
		return loss
      

    
if __name__ == '__main__':
    fn = ECCharbonnierLoss()
    a = torch.randn([2,2,3,3])
    b = torch.randn([2,2,3,3])
    cloudmask = torch.ones([2,3,3])
    print(fn(a,b,cloudmask))