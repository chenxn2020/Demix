import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Cluster_Adv_Loss(nn.Module):
    def __init__(self, args, model):
        super(Cluster_Adv_Loss, self).__init__()
        self.args = args
        self.model = model
    def forward(self, pos_score, neg_score, cache_sample_cluster=None, subsampling_weight=None):
        
        if cache_sample_cluster != None:
        #------根据cache_sample_cluster中记录的信息, True为和正样本同簇的样本， 更改neg_quality
            neg_quality = torch.where(cache_sample_cluster == True, -torch.ones_like(neg_score) * 100000, neg_score)
        else:
            neg_quality = neg_score

        neg_score = (F.softmax(neg_quality * self.args.adv_temp, dim=1).detach()
                    * F.logsigmoid(-neg_score)).sum(dim=1)  #shape:[bs]
        pos_score = F.logsigmoid(pos_score).squeeze(1) #shape:[bs]
        if self.args.use_weight:
            positive_sample_loss = - (subsampling_weight * pos_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * neg_score).sum()/subsampling_weight.sum()
        else:
            positive_sample_loss = - pos_score.mean()
            negative_sample_loss = - neg_score.mean()
        
        loss = (positive_sample_loss + negative_sample_loss) / 2
    
        if self.args.model_name == 'ComplEx' or self.args.model_name == 'DistMult':
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
            loss = loss + regularization
        return loss