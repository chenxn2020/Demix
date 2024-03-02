import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Adv_Bce_Loss(nn.Module):
    def __init__(self, args, model):
        super(Adv_Bce_Loss, self).__init__()
        self.args = args
        self.model = model
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, pos_score, neg_score, neg_label, subsampling_weight=None):
        neg_loss = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                    * self.bce(neg_score, neg_label)).sum(dim=1)  #shape:[bs]
        # neg_score = F.logsigmoid(-neg_score).mean(dim = 1)
        pos_label = torch.ones_like(pos_score)
        pos_loss = self.bce(pos_score, pos_label).squeeze(1) #shape:[bs]
        # from IPython import embed;embed();exit()

        if self.args.use_weight:
            positive_sample_loss = (subsampling_weight * pos_loss).sum()/subsampling_weight.sum()
            negative_sample_loss = (subsampling_weight * neg_loss).sum()/subsampling_weight.sum()
        else:
            positive_sample_loss = pos_loss.mean()
            negative_sample_loss = neg_loss.mean()

        loss = (positive_sample_loss + negative_sample_loss) / 2

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