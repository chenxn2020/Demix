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
from torch.utils.data import DataLoader
from IPython import embed

from functools import partial

class KGELitModel(BaseLitModel):
    def __init__(self, model, args, sampler):
        super().__init__(model, args, sampler)
        self.dis2cluster, self.cluster_id = 0,0
    
    def read_test32(self):
        test_pos = torch.zeros(1, 3).int()
        test_neg = torch.zeros(1, 10).int()
        with open("./dataset/FB15K237/test32.json", "r") as f:
            aa = json.load(f)
        for key in aa.keys():
            pos_sample = key.split("-")
            pos_sample = torch.tensor([int(_) for _ in pos_sample]).unsqueeze(0)
            test_pos = torch.cat((test_pos, pos_sample), dim=0)
            neg_sample = torch.tensor(aa[key]).unsqueeze(0)
            test_neg = torch.cat((test_neg, neg_sample), dim=0)
        return test_pos[1:].cuda(), test_neg[1:].cuda()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        pos_sample = sample[:,:3]
        pos_score = self.model(pos_sample)
        neg_score = self.model(pos_sample, neg_sample, mode)
        #-------------对负样本进行mixup
        # seed = torch.rand(neg_sample.shape[0], neg_sample.shape[1],1).cuda()
        # seed = seed * min(1, (self.current_epoch+1)/20)
        # if mode == 'tail-batch':
        #     pos_head, pos_rel, pos_tail = self.model.tri2emb(pos_sample)
        #     neg_head, neg_rel, neg_tail = self.model.tri2emb(pos_sample, neg_sample, mode)
        #     pos_tail = pos_tail.expand_as(neg_tail).detach()
        #     mixup_tail = seed*pos_tail + (1-seed)*neg_tail
        #     neg_score = self.model.score_func(neg_head, neg_rel, mixup_tail, mode)
        # else:
        #     pos_head, pos_rel, pos_tail = self.model.tri2emb(pos_sample)
        #     neg_head, neg_rel, neg_tail = self.model.tri2emb(pos_sample, neg_sample, mode)
        #     pos_head = pos_head.expand_as(neg_head).detach()
        #     mixup_head = seed*pos_head + (1-seed)*neg_head
        #     neg_score = self.model.score_func(mixup_head, neg_rel, neg_tail, mode)
        # if self.current_epoch == 20:
        #     embed();exit()
        #----聚类后计算loss
        #---------对负样本进行筛选，去掉对应pair下相同cluster的样本
        mask_ls = torch.zeros(1, neg_sample.shape[1]).bool()
        # if self.current_epoch < 15:
        if self.current_epoch == 0:
            neg_score = self.model(pos_sample, neg_sample, mode)
        else:
            # neg_filter = torch.zeros(1, 50).long().cuda()
            if mode == 'head-batch':
                pair_ls = sample[:, -2]
                for idx in range(pair_ls.shape[0]):
                    pair = pair_ls[idx].item()
                    '''注意这里使用rt2h_cl_train'''
                    train_cl = self.rt2h_cl_train[pair].flatten()   #这个pair在训练集中的包含的所有cluster
                    neg_cl = self.cluster_id[neg_sample[idx].cpu()].flatten()  #每个负样本对应的cluster
                    mask = np.in1d(neg_cl.numpy(), train_cl.numpy(), invert=True)
                    mask = torch.tensor(mask).view(1,-1)
                    mask_ls = torch.cat((mask_ls, mask), dim=0)
                    #---删掉属于pair包含的cluster的负样本
                #     neg = neg_sample[idx][mask]
                #     neg_score = self.model(pos_sample[idx].unsqueeze(0), neg, mode).detach()
                #     _, top_k_idx = torch.topk(neg_score, k=50, dim=-1)
                #     #----取得分前100的负样本
                #     neg = neg[top_k_idx[0]].unsqueeze(0)
                #     neg_filter = torch.cat((neg_filter, neg), dim=0)
                # neg_sample = neg_filter[1:]
            elif mode == 'tail-batch':
                pair_ls = sample[:, -1]
                for idx in range(pair_ls.shape[0]):
                    pair = pair_ls[idx].item()
                    '''注意这里使用hr2t_cl_train'''
                    train_cl = self.hr2t_cl_train[pair].flatten()   #这个pair在训练集中的包含的所有cluster
                    neg_cl = self.cluster_id[neg_sample[idx].cpu()].flatten()  #每个负样本对应的cluster
                    mask = np.in1d(neg_cl.numpy(), train_cl.numpy(), invert=True)
                    mask = torch.tensor(mask).view(1,-1)
                    mask_ls = torch.cat((mask_ls, mask), dim=0)
        #             #---删掉属于pair包含的cluster的负样本
        #             # neg = neg_sample[idx][mask]
        #             # neg_score = self.model(pos_sample[idx].unsqueeze(0), neg, mode).detach()
        #             # _, top_k_idx = torch.topk(neg_score, k=50, dim=-1)
        #             # #----取得分前topk的负样本
        #             # neg = neg[top_k_idx[0]].unsqueeze(0)
        #             # neg_filter = torch.cat((neg_filter, neg), dim=0)
        #         # neg_sample = neg_filter[1:]
        neg_score = self.model(pos_sample, neg_sample, mode)
        #----adv_loss
        
        if self.current_epoch == 0:
            loss = self.loss(pos_score, neg_score, epoch=self.current_epoch)
        if self.current_epoch != 0:
            loss = self.loss(pos_score, neg_score, mask_ls[1:], self.current_epoch)
            # if mode == 'tail-batch':
            #     pos = pos_sample[:,2]
            # else:
            #     pos = pos_sample[:, 0]
            # pos_cluster = self.cluster_id[pos]
            # neg_cluster = self.cluster_id[neg_sample.view(-1)].view(pos.shape[0], -1)
            # loss = self.loss(pos_score, neg_score, epoch=self.current_epoch,pos_cluster=pos_cluster, neg_cluster=neg_cluster)

        self.log("Train|loss", loss,  on_step=False, on_epoch=True)
        return loss
    
    # def 
    
    def training_epoch_end(self, results):
        epoch = self.current_epoch
        # if epoch and (not epoch % 5):
        if not epoch % 5:
            self.dis2cluster, self.cluster_id = self.run_cluster()



            # update_triples = self.trainer.datamodule.data_train
            # self.trainer.datamodule.data_train = update_triples
            # loader_a = len(self.trainer.datamodule.train_dataloader())
            # self.trainer.train_dataloader = self.trainer.datamodule.train_dataloader()

    # def train_dataloader(self):
    #     loader_a = DataLoader(range(8), batch_size=4)
    #     return loader_a

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
        #-----测试下nolk和lk对最多t的h，r pair的平均得分情况
        # aa = self.sampler.hr2t_train[list(self.sampler.hr2t_train.keys())[16]]
        # aa = torch.tensor(aa)
        # pos = torch.zeros(843,3).long()
        # pos[:,0]=4994
        # pos[:,1]=31
        # pos[:,2]=aa
        # pos_sample = pos.cuda()
        # pos_score = self.model(pos_sample)
        # neg_tail = self.sampler.hr2t_test[list(self.sampler.hr2t_train.keys())[16]]
        # neg_sample = torch.tensor(list(neg_tail)).cuda()
        # neg_score = self.model(pos_sample, neg_sample, "tail-batch")
        # embed();exit()
        #--------------
        # self.get_pair_score()

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
            outputs[metric] = round(number, 2)
        return outputs

    """这里设置优化器和lr_scheduler"""

    def configure_optimizers(self):
        # milestones = int(self.args.max_epochs / 2)
        # optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[milestones], gamma=0.1
        # )
        # optim_dict = {"optimizer": optimizer, "lr_scheduler": StepLR}
        # return optim_dict
        milestione_list = [30, 70, 120] 
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestione_list, gamma=0.5
        )
        optim_dict = {"optimizer": optimizer, "lr_scheduler": StepLR}
        return optim_dict

