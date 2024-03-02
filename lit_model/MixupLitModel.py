from ast import Pass
from cProfile import label
from logging import debug
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from collections import defaultdict as ddict
from .MixupBaseLitModel import MixupBaseLitModel
from IPython import embed
from eval_task import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import time
from functools import partial


class MixupLitModel(MixupBaseLitModel):
    """Processing of training, evaluation and testing.
    """

    def __init__(self, model, args, train_sampler):
        super().__init__(model, args, train_sampler)
        # Candidate mixup pool.
        self.hr2id = train_sampler.hr2id
        self.rt2id = train_sampler.rt2id
        self.hr_cnd = torch.ones(len(self.hr2id), args.cnd_size).int() 
        self.rt_cnd = torch.ones(len(self.rt2id), args.cnd_size).int()
        self.hr_score = torch.ones(len(self.hr2id)+1, 2) * 10000 
        self.rt_score = torch.ones(len(self.rt2id)+1, 2) * 10000
        self.start_time = time.time()
        self.valid_analysis_dataloader = self.build_dataloader()
        self.args = args

    def forward(self, x):
        return self.model(x)
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def training_step(self, batch, batch_idx):
        """Getting samples and training in KG model.
        
        Args:
            batch: The training data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            loss: The training loss for back propagation.
        """
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        pos_label = torch.ones(self.args.train_bs, 1).cuda()
        neg_label = torch.zeros_like(neg_sample).float()
        if self.current_epoch > self.args.mix_epoch:
            neg_score, neg_label = self.neg_generate(batch, neg_label)
        else:
            neg_score = self.model(pos_sample, neg_sample, mode)
        pos_score = self.model(pos_sample)
        score = [pos_score, neg_score]
        label = [pos_label, neg_label]
        if self.args.use_weight:
            subsampling_weight = batch["subsampling_weight"]
            loss = self.loss(score, label, subsampling_weight)
        else:
            loss = self.loss(score, label)
        self.log("Train|loss", loss,  on_step=False, on_epoch=True)
        return loss

    
    def calc_mixup_score(self, pos_sample, neg_emb, mode):
        head, rel, tail = self.model.tri2emb(pos_sample)
        if mode == "tail-batch":
            score = self.model.score_func(head, rel, neg_emb, mode)
        else:
            score = self.model.score_func(neg_emb, rel, tail, mode)
        return score

    def neg_generate(self, batch, neg_label):
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        pair_id = batch["pair_id"]
        neg_score = self.model(pos_sample, neg_sample, mode).detach()
        if neg_score.shape[1] > self.args.mix_neg:
            value, idx = torch.topk(neg_score, k=self.args.mix_neg, dim=-1)
            tmp_idx = torch.arange(0, self.args.train_bs).type(torch.LongTensor).unsqueeze(1).expand(-1, self.args.mix_neg)
            neg_sample = neg_sample[tmp_idx, idx]
            neg_score = value
            neg_label = neg_label[:, :self.args.mix_neg]
            
        
        neg_emb = self.model.get_emb(neg_sample, "entity") #shape [bs, num_neg, dim]
        neg_emb = self.model.get_emb(neg_sample, "entity") #shape [bs, num_neg, dim]
        if mode == "tail-batch":
            score_scope = self.hr_score[pair_id].cuda()
            cnd_pool = self.hr_cnd[pair_id]
        else:
            score_scope = self.rt_score[pair_id].cuda()
            cnd_pool = self.rt_cnd[pair_id]
        beta = torch.distributions.beta.Beta(self.args.alpha, self.args.alpha)
        min_score = score_scope[:, 0].unsqueeze(1) + (self.args.delta * min(self.args.beta, (self.current_epoch - self.args.mix_epoch)/self.args.time))
        mean_score = score_scope[:, 1].unsqueeze(1)
        if self.args.correct_neg:
            neg_label = self.correct_neg(neg_score, mean_score, neg_label)
            return neg_score, neg_label
        tmp_score = neg_score - min_score
        true_neg_idx = (tmp_score < 0) 
        '''mixup harder neg'''
        if not self.args.no_harder:
            neg_emb, neg_label = self.harder_mix(neg_emb, true_neg_idx, neg_sample, neg_label)


        neg_score[true_neg_idx] = 1000000 
        marginal_neg_idx = ((neg_score - mean_score) <= 0) 
        marginal_neg_emb = neg_emb[marginal_neg_idx]
        if marginal_neg_emb.shape[0] == 0 or self.args.no_cnd:
            mix_neg_score = self.calc_mixup_score(pos_sample, neg_emb, mode)
        else:
            pos_cnd_lam = beta.sample([marginal_neg_emb.shape[0]]).unsqueeze(1).cuda()
            pos_cnd_lam = torch.max(pos_cnd_lam, 1. - pos_cnd_lam)
            mix_cnd_idx = torch.randint(low=0, high=cnd_pool.shape[1], size=(self.args.train_bs, cnd_pool.shape[1]))
            tmp_cnd_idx = torch.arange(0, self.args.train_bs).type(torch.LongTensor).unsqueeze(1).expand(-1, cnd_pool.shape[1])
            upset_cnd_pool = cnd_pool[tmp_cnd_idx, mix_cnd_idx]
            x_idx = (marginal_neg_idx == True).nonzero()[:, 0]
            y_idx = (marginal_neg_idx == True).nonzero()[:, 1]


            mix_cnd = upset_cnd_pool[x_idx, y_idx].cuda()
            
            mix_cnd_emb = self.model.get_emb(mix_cnd, "entity")
            mix_pos_emb = pos_cnd_lam * marginal_neg_emb + (1. - pos_cnd_lam) * mix_cnd_emb
            neg_emb[marginal_neg_idx] = mix_pos_emb
            neg_label[marginal_neg_idx] = (1. - pos_cnd_lam).squeeze(1)
            mix_neg_score = self.calc_mixup_score(pos_sample, neg_emb, mode)
        return mix_neg_score, neg_label
    
    def correct_neg(self, neg_score, mean_score, neg_label):
        tmp_idx = (neg_score - mean_score) > 0
        neg_label[tmp_idx] = 1.0
        return neg_label

    def harder_mix(self, neg_emb, true_neg_idx, neg_sample, neg_label):
        beta = torch.distributions.beta.Beta(self.args.alpha, self.args.alpha)
        true_neg_emb = neg_emb[true_neg_idx]  
        harder_lam = beta.sample([true_neg_emb.shape[0]]).unsqueeze(1).cuda()
        harder_lam = torch.max(harder_lam, 1. - harder_lam)
        mix_neg_idx = torch.randint(low=0, high=neg_sample.shape[1], size=(self.args.train_bs, neg_sample.shape[1]))
        tmp_idx = torch.arange(0, self.args.train_bs).type(torch.LongTensor).unsqueeze(1).expand(-1, neg_sample.shape[1])
        mix_emb = neg_emb[tmp_idx, mix_neg_idx][true_neg_idx]
        harder_mix_emb = harder_lam * true_neg_emb + (1. - harder_lam) * mix_emb
        neg_emb[true_neg_idx] = harder_mix_emb
        neg_label[true_neg_idx] = (1. - harder_lam.squeeze(1)) * neg_label[tmp_idx, mix_neg_idx][true_neg_idx]

        return neg_emb, neg_label
    
    def training_epoch_end(self, results) -> None:
        if self.current_epoch < self.args.mix_epoch:
            return

        for pair in self.train_sampler.hr2t_train.keys():
            if self.train_sampler.hr2t_train[pair].shape[0] <= self.args.pos_threshold:
                continue
            pos = torch.tensor([pair[0], pair[1], 0]).unsqueeze(0).cuda()
            pos_t = self.train_sampler.hr2t_train[pair].unsqueeze(0).cuda()
            score = self.model(pos, pos_t, "tail-batch")
            pair_id = self.hr2id[pair]
            mean_score = score.mean().item()
            min_score = score.min().item()
            self.hr_score[pair_id][0] = min_score
            self.hr_score[pair_id][1] = mean_score
            candidate_pool = pos_t[0][torch.nonzero(score<=mean_score)[:,1]]

            idx = torch.randint(low=0, high=candidate_pool.shape[0], size=(self.args.cnd_size,))
            self.hr_cnd[pair_id] = candidate_pool[idx]

        for pair in self.train_sampler.rt2h_train.keys():
            if self.train_sampler.rt2h_train[pair].shape[0] <= self.args.pos_threshold:
                continue
            pos = torch.tensor([0, pair[0], pair[1]]).unsqueeze(0).cuda()
            pos_h = self.train_sampler.rt2h_train[pair].contiguous().unsqueeze(0).cuda()
            score = self.model(pos, pos_h, "head-batch")
            pair_id = self.rt2id[pair]
            mean_score = score.mean().item()
            min_score = score.min().item()
            self.rt_score[pair_id][0] = min_score
            self.rt_score[pair_id][1] = mean_score
            candidate_pool = pos_h[0][torch.nonzero(score<=mean_score)[:,1]]
            idx = torch.randint(low=0, high=candidate_pool.shape[0], size=(self.args.cnd_size,))
            self.rt_cnd[pair_id] = candidate_pool[idx]
        





    def build_dataloader(self):
        valid_analysis_dataloader = DataLoader(
            self.train_sampler.valid_triples + self.train_sampler.test_triples,
            batch_size=self.args.eval_bs,
            num_workers=0,
            collate_fn=self.collate_data,
        )
        return valid_analysis_dataloader

    def collate_data(self, data):
        batch_data = {}
        pair2id_ls = []
        for h, r, t in data:
            hr2id = self.train_sampler.hr2id.get((h, r), -1)
            rt2id = self.train_sampler.rt2id.get((r, t), -1)
            pair2id_ls.append([hr2id, rt2id])
        batch_data["false_negative"] = torch.LongTensor(data)
        batch_data['pair_id'] = torch.tensor(np.array(pair2id_ls))
        return batch_data

        


    def validation_step(self, batch, batch_idx):
        """Getting samples and validation in KG model.
        
        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,3,10.
        """
        results = dict()
        ranks = link_predict(batch, self.model, prediction='all')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    
    def validation_epoch_end(self, results) -> None:


        outputs = self.collect_results(results, "Eval")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Getting samples and test in KG model.
        
        Args:
            batch: The evaluation data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,3,10.
        """
        results = dict()
        ranks = link_predict(batch, self.model, prediction='all')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        results["mr"] = torch.sum(ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    
    def test_epoch_end(self, results) -> None:
        outputs = self.collect_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)
    
    

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.

        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.   
        """
        begin = self.args.lr_begin
        step = self.args.lr_step
        ls = [begin, begin + step, begin + 2*step]
        milestone_list = ls
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_list, gamma=self.args.lr_change)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

        