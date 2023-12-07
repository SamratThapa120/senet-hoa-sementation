import torch

class LogCoshDiceLoss:
    def __init__(self,eps=1):
        self.eps=eps
    def __call__(self,gt,pred):
        """
            accepts tensor of shape [batch_size,channel, width , height]
        """
        sigm_p = torch.sigmoid(pred)
        loss_whole = 1- ((2*gt*sigm_p).sum([1,2,3])+self.eps)/(gt.sum([1,2,3])+sigm_p.sum([1,2,3])+self.eps)
        return torch.log(torch.cosh(loss_whole)).mean()

class DiceLoss:
    def __init__(self,eps=1):
        self.eps=eps
    def __call__(self,gt,pred):
        """
            accepts tensor of shape [batch_size,channel, width , height]
        """
        sigm_p = torch.sigmoid(pred)
        loss_whole = 1- ((2*gt*sigm_p).sum([1,2,3])+self.eps)/(gt.sum([1,2,3])+sigm_p.sum([1,2,3])+self.eps)
        return loss_whole.mean()