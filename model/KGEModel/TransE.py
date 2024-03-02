import torch.nn as nn
import torch
from .model import Model
import torch.nn.functional as F
from IPython import embed


class TransE(Model):
    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None

        self.init_emb()

    def init_emb(self):
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )

        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        if self.args.loss_name == "Margin_Loss":
            nn.init.xavier_uniform_(self.ent_emb.weight.data)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)
        else:
            nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        

    def score_func(self, head_emb, relation_emb, tail_emb, mode=None):
        if self.args.loss_name == "Margin_Loss":
            head_emb = F.normalize(head_emb, 2, -1)
            relation_emb = F.normalize(relation_emb, 2, -1)
            tail_emb = F.normalize(tail_emb, 2, -1)
            '''正则化影响超大，对于margin loss来说'''
        score = (head_emb + relation_emb) - tail_emb
        # if self.args.loss_name == "Margin_Loss":
        #     score = torch.norm(score, p=1, dim=-1)
        # else: 
        score = self.margin.item() - torch.norm(score, p=1, dim=2)
        return score

    def forward(self, triples, negs=None, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
    
    def get_score(self, batch, mode=None, calc_score=False):
        triples = batch["positive_sample"]
        # score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        if calc_score:
            head_emb, relation_emb, tail_emb = self.tri2emb(triples)
        else:
            head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        #---查看分数和emb相似度的一致性:大体上是一致的
        # head_top10 = torch.argsort(score, dim=-1, descending=True)[0][:10]
        # neg_head = head_emb[0][head_top10]
        # gold_head = (tail_emb[0] - relation_emb[0])
        # pos_head = head_emb[0][triples[0][0]].unsqueeze(0)
        # score_top10 = score[0][head_top10]
        # pos_score = score[0][triples[0][0]]
        # embed();exit()
        return score

    # def get_emb(self, triples, negs=None, mode='single'):
    #     #pos emb
    #     pos_head, pos_rel, pos_tail = self.tri2emb(triples)
    #     #neg emb
    #     head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
    #     if mode == 'head-batch':
    #         anchor_head = pos_tail - pos_rel
    #         return anchor_head, pos_head, head_emb
    #     elif mode == 'tail-batch':
    #         anchor_tail = pos_head + pos_rel
    #         return anchor_tail, pos_tail, tail_emb
        