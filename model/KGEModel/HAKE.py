import torch.nn as nn
import torch
from .model import Model
import torch.nn.functional as F
from IPython import embed

class HAKE(Model):
    def __init__(self, args):
        super(HAKE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.init_emb()
    

    def init_emb(self):
        self.epsilon = 2.0
        self.pi = 3.14159265358979323846
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )
        
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 3)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.ones_(
            tensor=self.rel_emb.weight.data[:, self.args.emb_dim : 2 * self.args.emb_dim]
        )

        nn.init.zeros_(
            tensor=self.rel_emb.weight.data[:, 2 *self.args.emb_dim : 3 * self.args.emb_dim]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[self.args.phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[self.args.modulus_weight]]))

    def score_func(self, head_emb, rel_emb, tail_emb, mode):
        phase_head, mod_head = torch.chunk(head_emb, 2, dim=-1)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel_emb, 3, dim=-1)
        phase_tail, mod_tail = torch.chunk(tail_emb, 2, dim=-1)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if mode == "head-batch":
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=-1) * self.phase_weight
        r_score = torch.norm(r_score, dim=-1) * self.modulus_weight

        return self.margin.item() - (phase_score + r_score)
    
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
        return score