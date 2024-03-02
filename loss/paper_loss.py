import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Paper_Loss(nn.Module):
    def __init__(self, args, model):
        super(Paper_Loss, self).__init__()
        self.args = args
        self.model = model
    def forward(self, pos_score, neg_score, subsampling_weight=None, neg_mode='mean'):
        

        pos_loss = F.logsigmoid(pos_score).view(neg_score.shape[0]) #shape:[bs]
        if neg_mode == 'mean':
            neg_loss = F.logsigmoid(-neg_score).mean(dim=-1) #shape:[bs]
        else:
            neg_loss = F.logsigmoid(-neg_score).sum(dim=-1) #shape:[bs]
        loss = -(pos_loss +  neg_loss * self.args.negloss_weight)
        if self.args.use_weight:
            #----使用softmax计算weight          
            # weight = F.softmax(subsampling_weight, dim=0)
            #----使用比例计算weight
            weight = subsampling_weight / subsampling_weight.sum()
            loss = (weight * loss).sum()
        else:
            #TODO:考虑还要不要除以 2
            loss = loss.mean()

        # loss = (positive_sample_loss + negative_sample_loss) / 2

        if self.args.model_name == 'ComplEx' or self.args.model_name == 'DistMult' or self.args.model_name == 'BoxE' or self.args.model_name == 'SimplE':
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
            # embed();exit()
            loss = loss + regularization
        return loss
    
    def normalize(self):
        regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
        return regularization