import argparse
from logging import NOTSET
import pytorch_lightning as pl
import torch
import loss
from collections import defaultdict as ddict
import numpy as np
from IPython import embed
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import time
import torch.nn.functional as F
# import faiss 

class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None, sampler=None):
        super().__init__()
        self.model = model
        self.args = args
        optim_name = args.optim_name
        self.optimizer_class = getattr(torch.optim, optim_name)
        loss_name = args.loss_name
        self.loss_class = getattr(loss, loss_name)
        self.loss = self.loss_class(args, model)
        self.sampler = sampler
        
        if self.args.litmodel_name == "KGEStdLitModel":
            self._init_cache_std()
        if self.args.litmodel_name == "KGECacheLitModel":
            self._init_cache()
    
    def _init_cache_std(self):
        #------cache use------------------------------------
        self.expand_head_cache = torch.zeros(self.args.headcache_num, self.args.num_neg + 3).long().cuda() #算方差用
        self.expand_tail_cache = torch.zeros(self.args.tailcache_num, self.args.num_neg + 3).long().cuda()
        self.expand_head_cache[:, self.args.num_neg:] = torch.tensor(self.sampler.head_cache_pos).cuda()
        self.expand_tail_cache[:, self.args.num_neg:] = torch.tensor(self.sampler.tail_cache_pos).cuda()
        #用来缓存近几轮候选样本的得分, 算方差用
        self.expand_head_cache_score_list = torch.zeros(self.args.update_cache_epoch, self.args.headcache_num, self.args.num_neg+1).cuda()
        self.expand_tail_cache_score_list = torch.zeros(self.args.update_cache_epoch, self.args.tailcache_num, self.args.num_neg+1).cuda()
        #--用来监督exapnd_cache是否被更新过,防止二次更新.只有在要更换负样本候选池时使用
        self.head_cache_update = torch.zeros(self.args.headcache_num).int().cuda()
        self.tail_cache_update = torch.zeros(self.args.tailcache_num).int().cuda()
        # self.test32_headcache()

    def _init_cache(self):
        self.head_cache = torch.zeros(self.args.headcache_num, self.args.cache_size).long().cuda()
        self.tail_cache = torch.zeros(self.args.tailcache_num, self.args.cache_size).long().cuda()
        self.head_cache_all = self.sampler.head_cache_all
        self.tail_cache_all = self.sampler.tail_cache_all

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser

    

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
    
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
            outputs[final_metric] = np.around(np.array([o[metric] for o in results]).sum() / count, decimals=3).item()
        return outputs
    
    def rollback_update(self):
        #使用后回滚，方便下次使用
        self.head_cache_update = torch.zeros(self.args.headcache_num).int().cuda()
        self.tail_cache_update = torch.zeros(self.args.tailcache_num).int().cuda()

    def build_dataloader(self):
        expand_head_dataset = TensorDataset(self.expand_head_cache.cpu())
        expand_head_cache_dataloader = DataLoader(
            expand_head_dataset,
            batch_size=self.args.train_bs,
            # batch_size=16,
            num_workers=0,
            collate_fn=self.collate_data,
        )
        
        expand_tail_dataset = TensorDataset(self.expand_tail_cache.cpu())
        expand_tail_cache_dataloader = DataLoader(
            expand_tail_dataset,
            batch_size=self.args.train_bs,
            # batch_size=16,
            num_workers=0,
            collate_fn=self.collate_data,
        )
        return expand_head_cache_dataloader, expand_tail_cache_dataloader
    
    def collate_data(self, data):
        batch = torch.stack([_[0] for _ in data], dim=0)
        return batch.cuda()

    '''cache + std'''
    def update_cache_std(self, **batch):
        '''这个函数在每个step都会运行'''
        pos_sample = batch["positive_sample"]
        neg_head_set = batch['neg_head_set']
        neg_tail_set = batch['neg_tail_set']
        #每个正样本对应的cache, 计算loss用
        head_cache_idx_list = batch["head_cache_idx_list"] #shape: [batch_size]
        tail_cache_idx_list = batch["tail_cache_idx_list"]
        #一个batch下的正样本，去重后的cache idx， 更新cache用
        head_cache_set = batch["head_cache_idx_set"] #shape: [batch_size]
        tail_cache_set = batch["tail_cache_idx_set"]

        if self.current_epoch == 0:
            '''第一轮epoch替换cache，后续epoch更新cache'''
            #-----替换tail_cache,并更新expand_cache-----------------
            #查看cache是否被替换过，防止二次替换
            tail_cache_update = torch.nonzero(self.tail_cache_update[tail_cache_set] == 0).flatten()
            if tail_cache_update.shape[0] != 0:
                update_cache = tail_cache_set[tail_cache_update]
                # if torch.nonzero(update_cache == 16).shape[0] != 0:
                # embed();exit()
                # self.tail_cache[update_cache] = neg_tail_set[tail_cache_update,:self.args.cache_size]
                self.expand_tail_cache[update_cache, :self.args.num_neg] = neg_tail_set[tail_cache_update,:]
            self.tail_cache_update[tail_cache_set] = 1
            #----替换head_cache,并更新expand_cache------------------
            head_cache_update = torch.nonzero(self.head_cache_update[head_cache_set] == 0).flatten()
            if head_cache_update.shape[0] != 0:
                update_cache = head_cache_set[head_cache_update]
                # self.head_cache[update_cache] = neg_head_set[head_cache_update,:self.args.cache_size]
                self.expand_head_cache[update_cache, :self.args.num_neg] = neg_head_set[head_cache_update, :]
            self.head_cache_update[head_cache_set] = 1
            #---------test32使用
            # self.expand_head_cache[self.test_headcache_idx, :11] = self.test_head

        elif self.current_epoch % self.args.update_cache_epoch == 0:
            '''更新expand_cache'''
            '''这里要注意的是更新expand_cache，cache_size--numn_neg的样本是随机负采样的，即第50--第127位置的样本
            前50个是cache中的负样本'''
            tail_cache_update = torch.nonzero(self.tail_cache_update[tail_cache_set] == 0).flatten()
            head_cache_update = torch.nonzero(self.head_cache_update[head_cache_set] == 0).flatten()
            #--------更新exppand_tail_cache
            if tail_cache_update.shape[0] != 0:
                update_cache = tail_cache_set[tail_cache_update]
                self.expand_tail_cache[update_cache, self.args.head_cache_size:self.args.num_neg] = neg_tail_set[tail_cache_update,\
                    self.args.head_cache_size:self.args.num_neg]
                # self.expand_tail_cache[update_cache, :self.args.cache_size] = self.tail_cache[update_cache]

            #--------更新exppand_head_cache
            if head_cache_update.shape[0] != 0:
                update_cache = head_cache_set[head_cache_update]
                self.expand_head_cache[update_cache, self.args.tail_cache_size:self.args.num_neg] = neg_head_set[head_cache_update,\
                    self.args.tail_cache_size:self.args.num_neg]
                # self.expand_head_cache[update_cache, :self.args.cache_size] = self.head_cache[update_cache]
        '''返回cache中的负样本和其他候选样本'''
        head_cache_sample = self.expand_head_cache[head_cache_idx_list, :self.args.num_neg]
        tail_cache_sample = self.expand_tail_cache[tail_cache_idx_list, :self.args.num_neg]
        return head_cache_sample, tail_cache_sample


    def update_cache_sample_std(self):
        #计算替换头实体的负样本最后一轮的得分，和近几轮的方差
        head_sample_score = self.expand_head_cache_score_list[-1]
        head_sampe_std = torch.std(self.expand_head_cache_score_list, dim=0)

        head_sample_quality = self.calc_sample_quality(head_sample_score, head_sampe_std)
        _, top_head_idx = torch.topk(head_sample_quality[:, :self.args.num_neg], k=self.args.head_cache_size, dim=-1)
        # _, top_head_idx = torch.topk(head_sample_quality[:, :self.args.num_neg], k=self.args.head_cache_size*2, dim=-1)
        head_cache_idx = torch.arange(0, self.expand_head_cache.shape[0]).type(torch.LongTensor).unsqueeze(1).expand(-1, self.args.head_cache_size)
        # self.head_cache = self.expand_head_cache[head_cache_idx, top_head_idx]
        self.expand_head_cache[:, :self.args.head_cache_size] = self.expand_head_cache[head_cache_idx, top_head_idx]
        self.attention(head_sample_score)
        # self.expand_head_cache[:, :self.args.head_cache_size] = self.expand_head_cache[head_cache_idx, top_head_idx[:, 10:10+self.args.head_cache_size]]
        
        #把要观察的负样本放到cache中
        self.expand_head_cache[self.test_headcache_idx, :11] = self.test_head
        self.test_std = head_sampe_std[self.test_headcache_idx, :11].cpu().tolist()


        #计算替换尾实体的负样本最后一轮的得分，和近几轮的方差
        tail_sample_score = self.expand_tail_cache_score_list[-1]
        tail_sampe_std = torch.std(self.expand_tail_cache_score_list, dim=0)
        tail_sample_quality = self.calc_sample_quality(tail_sample_score, tail_sampe_std)
        _, top_tail_idx = torch.topk(tail_sample_quality[:,:self.args.num_neg], k=self.args.tail_cache_size, dim=-1)
        # _, top_tail_idx = torch.topk(tail_sample_quality[:,:self.args.num_neg], k=self.args.tail_cache_size*2, dim=-1)
        tail_cache_idx = torch.arange(0, self.expand_tail_cache.shape[0]).type(torch.LongTensor).unsqueeze(1).expand(-1, self.args.tail_cache_size)
        # self.tail_cache = self.expand_tail_cache[tail_cache_idx, top_tail_idx]
        # self.expand_tail_cache[:, :self.args.tail_cache_size] = self.expand_tail_cache[tail_cache_idx, top_tail_idx[:,5:5+self.args.tail_cache_size]]
        self.expand_tail_cache[:, :self.args.tail_cache_size] = self.expand_tail_cache[tail_cache_idx, top_tail_idx]

    '''后续着重优化这个计算部分'''
    def calc_sample_quality(self, sample_score, sample_std):
        '''根据样本的得分方差和最后一轮的得分，来计算样本的质量'''
        #--先按照分数大小过滤一半
        # pos_score = sample_score[:, -1].unsqueeze(1)
        sample_score = sample_score[:, :self.args.num_neg]
        sample_std = sample_std[:, :self.args.num_neg]
        # _, top_idx = torch.topk(sample_score, k=int(self.args.num_neg/2), dim=-1)
        # tmp_idx = torch.arange(0, sample_score.shape[0]).type(torch.LongTensor).unsqueeze(1).expand(-1, int(self.args.num_neg/2))
        # sample_score = sample_score[tmp_idx, top_idx]
        # sample_std = sample_std[tmp_idx, top_idx]
        sample_quality = torch.sigmoid(sample_score) + \
            self.args.alpha * min(1, (self.current_epoch+1)/self.args.warmup) * sample_std
        # if self.current_epoch + 1 == self.args.warmup:
        #     embed();exit()
        return sample_quality

    def collect_expand_sample_score(self, expand_head_score, expand_tail_score, idx):
        '''收集样本的分数，为后续算方差做准备'''
        self.expand_head_cache_score_list[idx] = expand_head_score
        self.expand_tail_cache_score_list[idx] = expand_tail_score
    
    
    def get_pair_score(self):
        #---------对特定的pair取出相应的triples，来检查它们的分数
        pair = (4994, 31) #这个hr的pair是在训练集和测试集中组成样本数最多的pair
        train_pair_t = torch.tensor(list(self.sampler.hr2t_train[pair]))
        test_pair_t = torch.tensor(list(self.sampler.hr2t_test[pair]))
        pos = torch.zeros(train_pair_t.shape[0], 3).long()
        pos[:,0]=4994
        pos[:,1]=31
        pos[:,2]=train_pair_t
        pos_sample = pos.cuda()
        neg_sample = torch.tensor(list(test_pair_t)).cuda()
        pos_score = self.model(pos_sample)
        neg_score = self.model(pos_sample, neg_sample, "tail-batch")
        return pos_score, neg_score

        
    def configure_optimizers(self):
        milestione_list = [30, 70, 120] 
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestione_list, gamma=0.5
        )
        optim_dict = {"optimizer": optimizer, "lr_scheduler": StepLR}
        return optim_dict


    def run_cluster(self):
        #-----使用kmeans算法对实体的emb进行聚类
        ent_emb = F.normalize(self.model.ent_emb.weight.data, p=2, dim=-1).detach().cpu().numpy()
        d = ent_emb.shape[1]
        #----聚类的数目
        k = self.args.cluster_num
        clus = faiss.Clustering(d, k)
        # clus.verbose = True
        #----迭代次数
        clus.niter = self.args.cluster_iter_num
        clus.nredo = 5
        clus.seed = self.args.seed
        clus.max_points_per_centroid = 200
        clus.min_points_per_centroid = 1
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0  
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(ent_emb, index)
        dis2cluster, cluster_id = index.search(ent_emb, 1)
        #---质心的emb
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        #----对质心的emb进行L2正则化
        centroids = F.normalize(torch.tensor(centroids), p=2, dim=-1)
        # dis2cluster = torch.tensor(np.array([n for n in dis2cluster]))
        cluster_id = torch.tensor(np.array([n for n in cluster_id]))

        #----查看聚类的效果
        if self.args.use_multi_cluster != 1:
            hr2t_train_ls = self.sampler.hr2t_train_ls
            rt2h_train_ls = self.sampler.rt2h_train_ls
            #-----储存每个pair包含哪些cluster
            self.hr2t_cl_train, self.rt2h_cl_train = [], []
            for pair2e in hr2t_train_ls:
                self.hr2t_cl_train.append(torch.unique(cluster_id[pair2e]))
            for pair2e in rt2h_train_ls:
                self.rt2h_cl_train.append(torch.unique(cluster_id[pair2e]))
        #----每个实体对应的cluster
        # self.dis2cluster = dis2cluster
        self.cluster_id = cluster_id.cuda()
        # self.centro_emb = centroids.cuda()
    
    def parse_batch(self, **batch):
        sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        pos_sample = sample[:,:3]
        mode = batch["mode"]
        head_cache_idx = sample[:, -2]
        tail_cache_idx = sample[:, -1]
        cache_idx = [head_cache_idx, tail_cache_idx]
        return pos_sample, neg_sample, mode, cache_idx

    '''only cache'''
    def update_cache(self, **batch):
        pos_sample, neg_sample, mode, cache_idx = self.parse_batch(**batch)
        head_cache_idx, tail_cache_idx = cache_idx
        if self.current_epoch == 0:
            '''第一轮epoch替换cache，后续epoch更新cache'''
            if mode == 'tail-batch':
                self.tail_cache[tail_cache_idx] = neg_sample[:,:self.args.cache_size]
            else:
                self.head_cache[head_cache_idx] = neg_sample[:,:self.args.cache_size]
        else:
            # if self.current_epoch % self.args.update_cache_epoch == 0:
                #每隔几轮更新一次cache
                self.update_cache_sample(**batch)
        
        
        if mode == 'tail-batch':
            cache_sample = self.tail_cache[tail_cache_idx]
            #TODO:在cache中检索false negative
        else:
            cache_sample = self.head_cache[head_cache_idx]

        return cache_sample
    
    #TODO:在update的时候，筛选出fs，在loss中减少weight

    def search_false_negative(self, cache_sample, **batch):
        #-----检索cache中的sample是否是fs， 后续可以在这里设计无监督设置下的操作
        pos_sample, neg_sample, mode, cache_idx = self.parse_batch(**batch)
        head_cache_idx, tail_cache_idx = cache_idx
        cache_sample_fs = [] #这里是个bool型的tensor，用来说明样本是否是false negative。True为fs
        if mode == "head-batch":
            for idx, cache_idx in enumerate(head_cache_idx.cpu().tolist()):
                fs_list = torch.tensor(self.head_cache_all[cache_idx])
                sample_fs_label = np.in1d(cache_sample[idx].cpu(), fs_list)
                cache_sample_fs.append(sample_fs_label)
        else:
            for idx, cache_idx in enumerate(tail_cache_idx.cpu().tolist()):
                fs_list = torch.tensor(self.tail_cache_all[cache_idx])
                sample_fs_label = np.in1d(cache_sample[idx].cpu(), fs_list)
                cache_sample_fs.append(sample_fs_label)
        #-----True为false negative
        cache_sample_fs = torch.tensor(np.array(cache_sample_fs))
        return cache_sample_fs

    



    def update_cache_sample(self, **batch):
        pos_sample, neg_sample, mode, cache_idx = self.parse_batch(**batch)
        head_cache_idx, tail_cache_idx = cache_idx
        tmp_idx = torch.arange(0, head_cache_idx.shape[0]).type(torch.LongTensor).unsqueeze(1).expand(-1, self.args.cache_size)
        '''后续可能要考虑relation 的伯努利值'''
        if mode == 'tail-batch':
            expand_cache = torch.cat((self.tail_cache[tail_cache_idx], neg_sample), dim=-1)
            cache_score = self.update_cache_score(expand_cache, **batch)
            #---选择分数top k的样本进入cache
            _, top_idx = torch.topk(cache_score, k=self.args.cache_size, dim=-1)
            self.tail_cache[tail_cache_idx] = expand_cache[tmp_idx, top_idx]
        else:
            expand_cache = torch.cat((self.head_cache[head_cache_idx], neg_sample), dim=-1)
            cache_score = self.update_cache_score(expand_cache, **batch)
            _, top_idx = torch.topk(cache_score, k=self.args.cache_size, dim=-1)
            self.head_cache[head_cache_idx] = expand_cache[tmp_idx, top_idx]



    def update_cache_score(self, expand_cache, **batch):
        pos_sample, neg_sample, mode, cache_idx = self.parse_batch(**batch)
        cache_score = self.model(pos_sample, expand_cache, mode).detach()
        #------利用聚类信息，扔掉与正样本同簇的样本
        if self.args.use_multi_cluster == 1:
            pos_cluster, cache_cluster = self.search_cluster_id(expand_cache, pos_sample, mode)
            cluster_bool = (pos_cluster == cache_cluster).detach()  #同簇为True
        else:
            #----统计该pair下所有正样本的簇类，并以此排除负样本
            cluster_bool = self.multi_cluster_neg(expand_cache, pos_sample, cache_idx, mode).detach()
        #---同簇的样本分数减小，从而尽可能的在cache中排除
        cache_score = torch.where(cluster_bool == True, -torch.ones_like(cache_score) * 1000, cache_score).detach()
        # cache_score = torch.where(cluster_bool == True, cache_score/(self.current_epoch*10), cache_score).detach()

        #------选用分数和正样本有一定差距的负样本
        # pos_score = self.model(pos_sample).detach()
        # cache_score = torch

        '''把cache中高于正样本分数的值抹去'''
        # tmp = torch.tensor([-10.0]).expand_as(cache_score).cuda()
        # cache_score = torch.where(cache_score < pos_score, cache_score, tmp)
        
        return cache_score 
    
    def multi_cluster_neg(self, expand_cache, pos_sample, cache_idx, mode):
        start = time.time()
        head_cache_idx, tail_cache_idx = cache_idx
        mask_ls = torch.zeros(1, expand_cache.shape[1]).bool().cuda()
        if mode == 'head-batch':
                pair_ls = head_cache_idx
                for idx in range(pair_ls.shape[0]):
                    pair = pair_ls[idx].item()
                    '''注意这里使用rt2h_cl_train'''
                    train_cl = self.rt2h_cl_train[pair].flatten().cuda()   #这个pair在训练集中的包含的所有cluster
                    cache_cl = self.cluster_id[expand_cache[idx]].flatten()  #每个负样本对应的cluster
                    mask = torch.isin(cache_cl, train_cl).unsqueeze(0)   #在multi类簇中的样本为True
                    mask_ls = torch.cat((mask_ls, mask), dim=0)
        elif mode == 'tail-batch':
            pair_ls = tail_cache_idx
            for idx in range(pair_ls.shape[0]):
                pair = pair_ls[idx].item()
                '''注意这里使用hr2t_cl_train'''
                train_cl = self.hr2t_cl_train[pair].flatten().cuda()   #这个pair在训练集中的包含的所有cluster
                cache_cl = self.cluster_id[expand_cache[idx]].flatten()  #每个负样本对应的cluster
                mask = torch.isin(cache_cl, train_cl).unsqueeze(0)   #在multi类簇中的样本为True
                mask_ls = torch.cat((mask_ls, mask), dim=0)
        end = time.time()
        return mask_ls[1:]

    def search_cluster_id(self, cache_sample, pos_sample, mode):
        if mode == "tail-batch":
            pos_t = pos_sample[:, 2]
            pos_cluster = self.cluster_id[pos_t]
            cache_cluster = self.cluster_id[cache_sample].squeeze(-1)
        else:
            pos_h = pos_sample[:, 0]
            pos_cluster = self.cluster_id[pos_h]
            cache_cluster = self.cluster_id[cache_sample].squeeze(-1)
        return pos_cluster, cache_cluster
