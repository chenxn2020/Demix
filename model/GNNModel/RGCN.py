# import dgl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn.pytorch import RelGraphConv
# # TODO: 这里能不能直接用之前写好的DistMult去完成score的计算？


# class RGCN(nn.Module):
#     def __init__(self, args):
#         super(RGCN, self).__init__()
#         self.args = args
#         self.ent_emb = None
#         self.rel_emb = None 
#         self.RGCN_1 = None
#         self.RGCN_2 = None   
#         self.Loss_emb   = None 
#         # create rgcn layers
#         self.build_model()

#     def build_model(self):

#         self.ent_emb = nn.Embedding(self.args.num_ent,self.args.emb_dim)

#         self.rel_emb = nn.Parameter(torch.Tensor(self.args.num_rel, self.args.emb_dim))

#         nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

#         self.RGCN_1 = RelGraphConv(self.args.emb_dim, self.args.emb_dim, 
#                                        self.args.num_rel, "bdd", num_bases=100, 
#                                        activation=F.relu, self_loop=True,
#                                        dropout=0.2)

#         self.RGCN_2 = RelGraphConv(self.args.emb_dim, self.args.emb_dim,
#                                        self.args.num_rel, "bdd", num_bases=100,
#                                        activation=None, self_loop=True,
#                                        dropout=0.2)

#     def forward(self, graph, ent, rel, norm, triples, mode='single'):
#         embedding = self.ent_emb(ent.squeeze())
#         embedding = self.RGCN_1(graph, embedding, rel, norm)
#         embedding = self.RGCN_2(graph, embedding, rel, norm)
#         self.Loss_emb = embedding
#         head_emb, rela_emb, tail_emb = self.tri2emb(embedding, triples, mode)
#         score = self.score_func(head_emb, rela_emb, tail_emb, mode)
#         return score

#     def score_func(self, head_emb, rela_emb, tail_emb, mode):
#         if mode == 'head-batch':
#             score = head_emb * (rela_emb * tail_emb)
#         else:
#             score = (head_emb * rela_emb) * tail_emb

#         score = score.sum(dim = -1)
#         return score

#     def tri2emb(self, embedding, triples, mode="single"):

#         rela_emb = self.rel_emb[triples[:, 1]].unsqueeze(1)  # [bs, 1, dim]
#         head_emb = embedding[triples[:, 0]].unsqueeze(1)  # [bs, 1, dim] 
#         tail_emb = embedding[triples[:, 2]].unsqueeze(1)  # [bs, 1, dim]

#         if mode == "head-batch":
#             head_emb = embedding.unsqueeze(0)  # [1, num_ent, dim]

#         elif mode == "tail-batch":
#             tail_emb = embedding.unsqueeze(0)  # [1, num_ent, dim]

#         return head_emb, rela_emb, tail_emb
