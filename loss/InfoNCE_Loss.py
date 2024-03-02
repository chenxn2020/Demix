import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class InfoNCE_Loss(nn.Module):
    def __init__(self, args, model):
        super(InfoNCE_Loss, self).__init__()
        self.args = args
        self.model = model
        # self.temp = args.temperature
        self.temp = 0.1
    def forward(self, anchor_emb, pos_emb, neg_emb, reduction='mean'):
        #mix up neg_emb
        # anchor_emb=anchor_emb.expand_as(neg_emb)
        seed = torch.rand(anchor_emb.shape[0], 1, 1).cuda()/2  #(0, 1)均匀分布
        neg_emb = seed * anchor_emb + (1 - seed) * neg_emb
        #归一化
        anchor_emb, pos_emb, neg_emb = self.normalize(anchor_emb, pos_emb, neg_emb)
        #这里算向量之间的相似度
        pos_score = torch.sum(anchor_emb * pos_emb, dim=-1)
        neg_score = torch.sum(anchor_emb * neg_emb, dim=-1)
        logits = torch.cat([pos_score, neg_score], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logits/self.temp, labels, reduction=reduction)
        return loss
   
    def normalize(self, *emb):
        return [None if x is None else F.normalize(x, dim=-1) for x in emb]