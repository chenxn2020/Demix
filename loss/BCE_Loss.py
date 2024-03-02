import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class BCE_Loss(nn.Module):
    def __init__(self, args, model):
        super(BCE_Loss, self).__init__()
        self.args = args
        self.loss = nn.BCEWithLogitsLoss(reduction='none') 
        self.model = model

    def forward(self, score, label, subsampling_weight=None):
        pos_score, neg_score = score
        pos_label, neg_label = label
        pos_loss = self.loss(pos_score, pos_label)
        neg_loss = self.loss(neg_score, neg_label)
        pos_loss = pos_loss.squeeze(dim=1) #shape [bs]
        if self.args.adv_temp != 0:
            #这里我们用self-adv sampling
            neg_loss = F.softmax(neg_score * self.args.adv_temp, dim=1).detach() * neg_loss
            neg_loss = neg_loss.sum(dim=1) #shape [bs]
            if self.args.use_weight:
                pos_loss = (subsampling_weight * pos_loss).sum()/subsampling_weight.sum() # shape:[] 数值
                neg_loss = (subsampling_weight * neg_loss).sum()/subsampling_weight.sum()
                loss = (pos_loss + neg_loss) / 2
            else:
                loss = (pos_loss + neg_loss).mean() 
        else:
            #这里用uniform sampling
            if self.args.use_weight:
                neg_loss = neg_loss.sum(dim=1) #shape [bs]
                pos_loss = (subsampling_weight * pos_loss).sum()/subsampling_weight.sum() # shape:[] 数值
                neg_loss = (subsampling_weight * neg_loss).sum()/subsampling_weight.sum()
                loss = (pos_loss + neg_loss) / 2
            else:
                loss = (pos_loss + neg_loss.sum(dim = 1)).mean()   #mixkg里的loss
        if self.args.model_name == 'DistMult' or self.args.model_name == 'ComplEx':
            assert(self.args.regularization != 0)
            loss = loss + self.args.regularization * self.model.reg(self.args.regu_norm)
        return loss