from logging import debug
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from collections import defaultdict as ddict
from IPython import embed
import loss
from .BaseLitModel import BaseLitModel
from eval_task import *
from IPython import embed
from torch.utils.data import DataLoader

from functools import partial

class KGEBCELitModel(BaseLitModel):
    def __init__(self, model, args, train_sampler):
        super().__init__(model, args)
        self.train_sampler = train_sampler
        self.hr2t_train_ls = train_sampler.hr2t_train_ls
        self.rt2h_train_ls = train_sampler.rt2h_train_ls
        # self.train_loader = self.build_dataloader()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        ##----------全量样本做bce loss
        if self.args.loss_name == "Cross_Entropy_Loss":
            # sample = batch["sample"]
            # label  = batch["label"]
            # mode = batch["mode"]
            # sample_score = self.model(sample, mode=mode)
            # logits = torch.sigmoid(sample_score)
            # loss = self.loss(logits, label)
            sample = batch["positive_sample"]
            pos_sample = sample[:, :3]
            neg_sample = batch["negative_sample"]
            mode = batch["mode"]
            pos_score = self.model(pos_sample)
            neg_score = self.model(pos_sample,neg_sample, mode)
            pos_label = torch.ones_like(pos_score)
            neg_label = batch["neg_label"]
            score = torch.cat((pos_score, neg_score), dim=-1)
            label = torch.cat((pos_label, neg_label), dim=-1)
            pred = torch.sigmoid(score)
            loss = self.loss(pred, label)
        #-----------随机采样后做bce loss
        else :
            sample = batch["positive_sample"]
            pos_sample = sample[:, :3]
            neg_sample = batch["negative_sample"]
            mode = batch["mode"]
            subsampling_weight = batch["subsampling_weight"]
            if self.args.use_weight:
                subsampling_weight = batch["subsampling_weight"]
            pos_score = self.model(pos_sample)
            neg_score = self.model(pos_sample,neg_sample, mode)
            # pos_label = torch.ones_like(pos_score)
            neg_label = batch["neg_label"]
            neg_label = self.correct_label(neg_score, neg_label)
            loss = self.loss(pos_score, neg_score, neg_label, subsampling_weight)
            return loss
            # self.search_mixup_partner(batch)
            # score = torch.cat((pos_score, neg_score), dim=-1)
            # label = torch.cat((pos_label, neg_label), dim=-1)
            score, label = self.mixup(batch)
            loss = self.loss(score, label)
            #TODO:分开计算正负loss再加起来的方式效果很差
            # pos_loss = self.loss(pos_score, pos_label)
            # neg_loss = self.loss(neg_score, neg_label)
            # loss = pos_loss + self.args.beta * neg_loss   #这种loss的计算方式，模型效果特别差 母鸡why
        
        self.log("Train|loss", loss,  on_step=False, on_epoch=True)
        return loss
    def correct_label(self, neg_score, neg_label):
        neg_logits = torch.sigmoid(neg_score).detach()
        pos_label = torch.ones_like(neg_label)
        #对预测概率大于一定值的负样本标签设为1
        correct_label = torch.where(neg_logits>self.args.corr_logits, pos_label, neg_label)
        return correct_label

    def build_dataloader(self):
        return DataLoader(
                self.train_sampler.train_triples,
                shuffle=False,
                batch_size=self.args.train_bs,
                pin_memory=True,
                collate_fn=self.collate_data,
            )
    def collate_data(self, data):
        batch = torch.stack([_[0] for _ in data], dim=0)
        return batch.cuda()

    def search_mixup_partner(self, batch):
        #TODO:想想怎么mixup
        #TODO：可以分三种情况mixup，ent, head-tail, head-rel-tail
        sample = batch["positive_sample"]
        pos_sample = sample[:, :3]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        neg_label = batch["neg_label"]
        # for idx, neg in enumerate(neg_sample):
        #     if mode == "tail-batch":
        #         cache_idx = sample[idx][-1].item()
        #         all_pos = torch.tensor(self.hr2t_train_ls[cache_idx]).cuda().view(1, -1)
        #         all_pos_score = self.model(pos_sample[idx].unsqueeze(0), all_pos, mode) #shape[1, n]
        #         embed();exit()
    def adv_mixup(self, batch):
        #---将mixup和adv loss结合
        sample = batch["positive_sample"]
        pos_sample = sample[:, :3]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        neg_label = batch["neg_label"]
        t = 0
        idx=0
        for i in pos_sample:
            aa = (i[0].item(), i[1].item())
            if self.train_sampler.hr2t_test[aa] != set():
                if self.train_sampler.hr2t_test[aa].shape[0] > t:
                    t = self.train_sampler.hr2t_test[aa].shape[0]
                    idx = aa


        embed();exit()


    def mixup(self, batch):
        sample = batch["positive_sample"]
        pos_sample = sample[:, :3]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        neg_label = batch["neg_label"]
        pos_score = self.model(pos_sample)
        neg_score = self.model(pos_sample, neg_sample, mode)
        pos_label = torch.ones_like(pos_score)
        seed = torch.rand_like(neg_score).unsqueeze(-1)
        seed = torch.where(seed-0.5>0, seed, 1-seed)
        #---取正样本的emb
        pos_head, pos_rel, pos_tail = self.model.tri2emb(pos_sample) #shape:[bs, 1, dim]
        if mode == "tail-batch":
            neg_t = self.model.ent_emb(neg_sample)
            mixup_t = seed * neg_t + (1-seed)*pos_tail  #shape:[bs, num_neg, dim]
            mixup_label = seed.squeeze(-1) * neg_label + (1 - seed.squeeze(-1)) * pos_label #shape:[bs, num_neg]
            mixup_score = self.model.score_func(pos_head, pos_rel, mixup_t, mode)
        else:
            neg_h = self.model.ent_emb(neg_sample)
            mixup_h = seed * neg_h + (1-seed)*pos_head  #shape:[bs, num_neg, dim]
            mixup_label = seed.squeeze(-1) * neg_label + (1 - seed.squeeze(-1)) * pos_label #shape:[bs, num_neg]
            mixup_score = self.model.score_func(mixup_h, pos_rel, pos_tail, mode)
        #---汇总所有label
        label = torch.cat([pos_label, mixup_label], dim=-1)
        #---汇总所有分数
        score = torch.cat([pos_score, mixup_score], dim=-1)

        return score, label



    def validation_step(self, batch, batch_idx):
        # pos_triple, tail_label, head_label = batch
        results = dict()
        ranks = link_predict(batch, self.model, calc_filter=self.args.calc_filter)
        results["count"] = torch.numel(ranks)
        results["Eval|mrr"] = torch.sum(1.0 / ranks).item()
        for k in [1, 3, 10]:
            results["Eval|hits@{}".format(k)] = torch.numel(ranks[ranks <= k])
        return results

    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval|")
        # self.log("Eval|mrr", outputs["Eval|mrr"], on_epoch=True)
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        
        results = dict()
        ranks = link_predict(
            batch,
            self.model,
            calc_filter=self.args.calc_filter
        )
        results["count"] = torch.numel(ranks)
        results["Test|mrr"] = torch.sum(1.0 / ranks).item()
        for k in [1, 3, 10]:
            results["Test|hits@{}".format(k)] = torch.numel(ranks[ranks <= k])
        return results

    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Test|")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def get_results(self, results, mode):
        outputs = ddict(float)
        count = np.array([o["count"] for o in results]).sum().item()
        metrics = ["mrr", "hits@1", "hits@3", "hits@10"]
        metrics = [mode + metric for metric in metrics]
        for metric in metrics:
            number = np.array([o[metric] for o in results]).sum().item() / count
            outputs[metric] = round(number, 4)
        return outputs