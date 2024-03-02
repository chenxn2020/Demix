import torch.nn as nn
import torch
from .model import Model
from IPython import embed


class ComplEx(Model):
    def __init__(self, args):
        super(ComplEx, self).__init__(args)
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
        
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 2)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

        return torch.sum(
            re_head * re_tail * re_relation
            + im_head * im_tail * re_relation
            + re_head * im_tail * im_relation
            - im_head * re_tail * im_relation,
            -1
        )
        

    def forward(self, triples, negs=None, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def get_score(self, batch, mode):
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
    def reg(self, regu_norm):
        reg = (self.ent_emb.weight.norm(p = regu_norm) ** regu_norm + \
            self.rel_emb.weight.norm(p = regu_norm) ** regu_norm)
        return reg