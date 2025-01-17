import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from .model import Model
from IPython import embed


class CrossE(Model):
    def __init__(self, args):
        super(CrossE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.init_emb()

    def init_emb(self):

        self.dropout = nn.Dropout(self.args.dropout)
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)  #关系的rel emb
        self.rel_reverse_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)  #reverse关系的rel emb 
        self.h_weighted_vector = nn.Embedding(self.args.num_rel, self.args.emb_dim)   #interaction mactrix
        self.t_weighted_vector = nn.Embedding(self.args.num_rel, self.args.emb_dim)   #interaction mactrix
        # self.bias = nn.Embedding(2, self.args.emb_dim)
        self.hr_bias = nn.Parameter(torch.zeros([self.args.emb_dim]))
        self.tr_bias = nn.Parameter(torch.zeros([self.args.emb_dim]))
        sqrt_size = 6.0 / math.sqrt(self.args.emb_dim)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-sqrt_size, b=sqrt_size)
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-sqrt_size, b=sqrt_size)
        nn.init.uniform_(tensor=self.rel_reverse_emb.weight.data, a=-sqrt_size, b=sqrt_size)
        nn.init.uniform_(tensor=self.h_weighted_vector.weight.data, a=-sqrt_size, b=sqrt_size)
        nn.init.uniform_(tensor=self.t_weighted_vector.weight.data, a=-sqrt_size, b=sqrt_size)

    def score_func(self, ent_emb, rel_emb, weighted_vector, mode):
        x = ent_emb * weighted_vector + rel_emb * ent_emb * weighted_vector
        if mode == "tail_predict":
            x = torch.tanh(x + self.hr_bias)
        else:
            x = torch.tanh(x + self.tr_bias)
        x = self.dropout(x)
        x = torch.mm(x, self.ent_emb.weight.data.t())
        x = torch.sigmoid(x)
        return x

    def forward(self, triples, mode="single"):
        head_emb = self.ent_emb(triples[:, 0])
        tail_emb = self.ent_emb(triples[:, 2])
        rel_emb = self.rel_emb(triples[:, 1])
        rel_reverse_emb = self.rel_reverse_emb(triples[:, 1])
        h_weighted_vector = self.h_weighted_vector(triples[:, 1])
        t_weighted_vector = self.t_weighted_vector(triples[:, 1])
        hr_score = self.score_func(head_emb, rel_emb, h_weighted_vector, "tail_predict")
        tr_score = self.score_func(tail_emb, rel_reverse_emb, t_weighted_vector, "head_predict")
        # bias = self.bias(triples_id)
        return hr_score, tr_score
    def get_score(self, batch, mode):
        triples = batch["positive_sample"]
        if mode == "tail_predict":
            head_emb = self.ent_emb(triples[:, 0])
            rel_emb = self.rel_emb(triples[:, 1])
            h_weighted_vector = self.h_weighted_vector(triples[:, 1])
            return self.score_func(head_emb, rel_emb, h_weighted_vector, "tail_predict")
        else:
            tail_emb = self.ent_emb(triples[:, 2])
            rel_reverse_emb = self.rel_reverse_emb(triples[:, 1])
            t_weighted_vector = self.t_weighted_vector(triples[:, 1])
            return self.score_func(tail_emb, rel_reverse_emb, t_weighted_vector, "head_predict")

    def regularize_loss(self, norm=2):
        return (self.ent_emb.weight.norm(p = norm) ** norm + \
            self.rel_emb.weight.norm(p = norm) ** norm + \
            self.rel_reverse_emb.weight.norm(p = norm) ** norm + \
            self.h_weighted_vector.weight.norm(p = norm) ** norm + \
            self.t_weighted_vector.weight.norm(p = norm) ** norm + \
            self.hr_bias.norm(p = norm) ** norm + \
            self.tr_bias.norm(p=norm) ** norm)

    
   

