import torch.nn as nn
import torch
from .model import Model
from IPython import embed


class RotatE(Model):
    def __init__(self, args):
        super(RotatE, self).__init__(args)
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
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation_emb/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.margin.item() - score.sum(dim = -1)
        return score

    def forward(self, triples, negs=None, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    # def get_score(self, batch, mode):
    #     triples = batch["positive_sample"]
    #     head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
    #     score = self.score_func(head_emb, relation_emb, tail_emb, mode)
    #     return score
    def get_score(self, batch, mode=None, calc_score=False):
        triples = batch["positive_sample"]
        # score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        if calc_score:
            head_emb, relation_emb, tail_emb = self.tri2emb(triples)
        else:
            head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score