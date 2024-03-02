import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Margin_Loss(nn.Module):
    def __init__(self, args, model):
        super(Margin_Loss, self).__init__()
        self.args = args
        self.model = model
        self.margin_loss = nn.MarginRankingLoss(self.args.margin)

    def forward(self, pos_score, neg_score):
        y = torch.ones(self.args.train_bs, self.args.num_neg).cuda()
        loss = self.margin_loss(pos_score, neg_score, y) #正样本分数越大越好
        return loss