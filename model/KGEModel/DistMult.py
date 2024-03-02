import torch.nn as nn
import torch
from .model import Model
from IPython import embed


class DistMult(Model):
    def __init__(self, args):
        super(DistMult, self).__init__(args)
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
        if self.args.init_mode == 'xavier':
            # NSCaching和PUDA 都是此初始化方式
            nn.init.xavier_uniform_(self.ent_emb.weight.data)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)
        else:
            # self.ent_emb.weight.data = torch.zeros(self.args.num_ent, self.args.emb_dim)
            # self.rel_emb.weight.data = torch.zeros(self.args.num_rel, self.args.emb_dim)
            nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        if mode == 'head-batch':
            score = head_emb * (relation_emb * tail_emb)
        else:
            score = (head_emb * relation_emb) * tail_emb

        score = score.sum(dim = -1)
        return score

    def forward(self, triples, negs=None, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score
    
    def get_score(self, batch, mode):
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
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

        return score
    
    def reg(self, regu_norm):
        reg = (self.ent_emb.weight.norm(p = regu_norm) ** regu_norm + \
            self.rel_emb.weight.norm(p = regu_norm) ** regu_norm)
        return reg