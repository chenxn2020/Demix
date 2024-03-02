import argparse
import pytorch_lightning as pl
import torch
from collections import defaultdict as ddict

import loss
import numpy as np
from IPython import embed
import pandas as pd
import os
# from sklearn.manifold import TSNE 
# tsne = TSNE(n_components=2) 

class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


class MixupBaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None, train_sampler=None):
        super().__init__()
        self.model = model
        self.args = args
        optim_name = args.optim_name
        self.optimizer_class = getattr(torch.optim, optim_name)
        loss_name = args.loss_name
        self.loss_class = getattr(loss, loss_name)
        self.loss = self.loss_class(args, model)
        self.train_sampler = train_sampler
        # self.tsne = TSNE(n_components=2) 


    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError
    
    def get_progress_bar_dict(self):
        # 不显示v_num
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def collect_results(self, results, mode):
        """Summarize the results of each batch and calculate the final result of the epoch
        Args:
            results ([type]): The results of each batch
            mode ([type]): Eval or Test
        Returns:
            dict: The final result of the epoch
        """
        outputs = ddict(float)
        count = np.array([o["count"] for o in results]).sum()
        for metric in list(results[0].keys())[1:]:
            final_metric = "|".join([mode, metric])
            outputs[final_metric] = np.around(np.array([o[metric] for o in results]).sum() / count, decimals=6).item()
        return outputs
    
    # def down_dim(self, pair, csv_name):
    #     ent2name = dict()
    #     id2ent = self.train_sampler.id2ent
    #     with open('./dataset/FB15K237/FB15k_mid2name.txt') as f:
    #         for line in f.readlines():
    #             ent, name = line.strip().split()
    #             ent2name[ent] = name
        
    #     hr2t_train = self.train_sampler.hr2t_train
    #     rt2h_train = self.train_sampler.rt2h_train
    #     hr2t_false = self.train_sampler.hr2t_valid_test
    #     rt2h_false = self.train_sampler.rt2h_valid_test
    #     # ss = self.filter_num(hr2t_train)
    #     head, rel = pair
    #     pos = torch.tensor([head, rel, 0]).unsqueeze(0)
    #     pos_emb = self.filter_emb(pos, pair, hr2t_train)
    #     # pos_ent = rt2h_train[nationality]
    #     # pos_emb = self.model.get_emb(pos_ent, "entity")
    #     # pos_emb = tsne.fit_transform(pos_emb.detach().numpy())
    #     pos_label = np.array([['pos' for i in range(pos_emb.shape[0])]])
    #     pos_emb = np.vstack((pos_emb.T, pos_label)).T
    #     #----false
    #     false_ent = hr2t_false[pair]
    #     false_score = self.model(pos, false_ent, 'tail-batch').detach()
    #     b = torch.argsort(false_score, dim=1, descending=False)
    #     fasle_ent = false_ent[b[0]]
    #     false_emb = self.model.get_emb(false_ent, "entity")
    #     false_emb = self.tsne.fit_transform(false_emb.detach().numpy())
    #     false_label = np.array([['false' for i in range(false_emb.shape[0])]])
    #     false_emb = np.vstack((false_emb.T, false_label)).T
    #     emb = np.vstack((pos_emb, false_emb))
    #     tsne = pd.DataFrame(emb, columns=['x', 'y', 'ent'])
    #     tsne.to_csv(csv_name)
    #     # embed();exit()
    #     # pos_score = self.model(pos, rt2h_train[nationality], 'head-batch').detach()
    #     # candidate_pool = pos_t[0][torch.nonzero(score<=mean_score)[:,1]]
        
    # def get_topk_neg(self, pair):
    #     pair2ent = self.train_sampler.hr2t_all
    #     label = torch.zeros(len(self.train_sampler.ent2id))
    #     pos = pair2ent[pair]
    #     label[pos] = 1.0
    #     head, rel = pair
    #     pos_sample = torch.tensor([head, rel, 0]).unsqueeze(0)
    #     score = self.model(pos_sample, mode='tail-batch').detach()
    #     pred_score = torch.where(label.bool(), -torch.ones_like(score) * 10000000, score)
    #     _, topk_neg = torch.topk(pred_score, dim=1, k=30)
    #     topk_neg_emb = self.model.get_emb(topk_neg, "entity").squeeze(0)
    #     ss = self.tsne.fit_transform(topk_neg_emb.detach().numpy())
    #     tsne = pd.DataFrame(ss, columns=['x', 'y'])
    #     tsne.to_csv('./true-neg-hsg.csv')
    #     embed();exit()

    # def filter_num(self, pair2ent):
    #     ss = dict()
    #     for pair in pair2ent.keys():
    #         if pair2ent[pair].shape[0] > 10 and pair2ent[pair].shape[0] < 25:
    #             ss[pair] = pair2ent[pair]
    #     return ss
    # def get_rank(self, pred_score):
    #     rank = 1 + torch.argsort(
    #         torch.argsort(pred_score, dim=1, descending=True), dim=1, descending=False)
    #     return rank
    
    # def filter_emb(self, pos, pair, pair2ent, mode='tail-batch'):
    #     # tsne = TSNE(n_components=2) 
    #     ent = pair2ent[pair]
    #     score = self.model(pos, ent, mode).detach()
    #     mean_score = score.mean().item()
    #     mean_ent = ent[torch.nonzero(score<=mean_score)[:,1]]
    #     score = self.model(pos, mean_ent, mode).detach()
    #     b = torch.argsort(score, dim=1, descending=False)
    #     filter_ent = mean_ent[b[0]]
    #     embed();exit()
    #     filter_emb = self.model.get_emb(filter_ent, "entity")
    #     emb = self.tsne.fit_transform(filter_emb.detach().numpy())
    #     return emb
    
    # def get_pairs(self, num=20):
    #     tmp = []
    #     for i in list(self.train_sampler.hr2t_valid_test.keys()):
    #         if self.train_sampler.hr2t_valid_test[i].shape[0] > num:
    #             tmp.append(i)
    #     return tmp
    
    # def save_dim(self):
    #     pairs = self.get_pairs(30)
    #     # for idx in range(len(pairs)):
    #     pair = pairs[8]
    #     self.get_topk_neg(pair)
    #     # embed();exit()
    #     # csv_name = 'tsne-nomarl' + str(idx) + '.csv'
    #     csv_name = '8new.csv'
    #     path = os.path.join("./tsne-hsg", csv_name)
    #     self.down_dim(pair, path)
            
             