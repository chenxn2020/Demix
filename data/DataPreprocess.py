import numpy as np
from torch.utils.data import Dataset
import torch
import os
from collections import defaultdict as ddict
from IPython import embed
import json
from scipy import sparse

class KGData(object):
    """Data preprocessing of kg data.
    Attributes:
        args: Some pre-set parameters, such as dataset path, etc.
    """

    # TODO:把里面的函数再分一分，最基础的部分再初始化的使用调用，其他函数具体情况再调用
    def __init__(self, args):
        self.args = args

        #  基础部分
        self.ent2id = {}
        self.rel2id = {}
        # predictor需要
        self.id2ent = {}
        self.id2rel = {}
        # 存放三元组的id
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.head_cache_idx = []
        self.tail_cache_idx = []
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.all_true_triples = set()
        self.head_cache_pos = [] #这个列表存储一个cache对应的正样本，为了后续都所有候选负样本计算分数使用
        self.tail_cache_pos = []
        self.get_id()
        self.get_triples_id()
        if args.use_weight:
            self.count = self.count_frequency()
        
        

    def get_id(self):
        """Get entity/relation id, and entity/relation number.
        Update:
            self.ent2id: Entity to id.
            self.rel2id: Relation to id.
            self.id2ent: id to Entity.
            self.id2rel: id to Relation.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """
        with open(os.path.join(self.args.data_path, "entities.dict")) as fin:
            for line in fin:
                eid, entity = line.strip().split("\t")
                self.ent2id[entity] = int(eid)
                self.id2ent[int(eid)] = entity

        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation] = int(rid)
                self.id2rel[int(rid)] = relation

        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)

    def get_triples_id(self):
        """Get triples id, save in the format of (h, r, t).
        Update:
            self.train_triples: Train dataset triples id.
            self.valid_triples: Valid dataset triples id.
            self.test_triples: Test dataset triples id.
        """
        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        # with open(os.path.join(self.args.data_path, "valid.txt")) as f:
        with open(os.path.join(self.args.data_path, "test.txt")) as f:
        # with open(os.path.join(self.args.data_path, self.args.valid_replace)) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )


    def get_hr2t_rt2h_from_train(self):
        """Get the set of hr2t and rt2h from train dataset, the data type is numpy.

        Update:
            self.hr2t_train: The set of hr2t.
            self.rt2h_train: The set of rt2h.
        """
        
        if self.args.leakage:
            # triples = self.all_true_triples
            triples = set(self.train_triples + self.valid_triples)
        else:
            triples = self.train_triples
        
        for h, r, t in triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)

        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = torch.tensor((list(self.hr2t_train[(h, r)])))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = torch.tensor(list(self.rt2h_train[(r, t)]))
        
    def get_train_caches(self):
        train_caches = []
        # assert len(self.train_triples) == len(self.head_cache_idx)
        for idx in range(len(self.train_triples)):
            triples_caches = [self.head_cache_idx[idx], self.tail_cache_idx[idx]]
            triples = list(self.train_triples[idx])
            train_caches.append(triples + triples_caches)
        train_caches = np.array(train_caches)
        return train_caches

    def get_cache_inf(self):
        #------用list存储pair对应的实体，以及每个训练样本对应第几个pair
        #------之后在训练的过程中去找到false negative
        hr2t_all_tmp = ddict(set)
        rt2h_all_tmp = ddict(set)
        hr_idx = dict()
        rt_idx = dict()
        hr_count, rt_count = 0, 0
        self.hr2t_train_ls = []
        self.rt2h_train_ls = []
        #--------这里的head_cache_all是记录对于每个cache，所有的正样本
        #-------比如head_cache_all表示对于第i个cache，有多少个头实体。以列表的形式存储
        self.head_cache_all, self.tail_cache_all = [], []
        for h, r, t in self.all_true_triples:
            hr2t_all_tmp[(h, r)].add(t)
            rt2h_all_tmp[(r, t)].add(h)
        for h, r, t in self.train_triples:
            '''保存三元组对应的cache的索引'''
            if (h, r) not in hr_idx:
                hr_idx[(h, r)] = hr_count
                hr_count += 1
                self.tail_cache_all.append(list(hr2t_all_tmp[(h, r)]))
                self.hr2t_train_ls.append(list(self.hr2t_train[(h,r)]))
            self.tail_cache_idx.append(hr_idx[(h, r)])
            if (r, t) not in rt_idx:
                rt_idx[(r, t)] = rt_count
                rt_count += 1
                self.head_cache_all.append(list(rt2h_all_tmp[(r, t)]))
                self.rt2h_train_ls.append(list(self.rt2h_train[(r,t)]))
            self.head_cache_idx.append(rt_idx[(r, t)])
    
        self.args.headcache_num = len(rt_idx)
        self.args.tailcache_num = len(hr_idx)
        self.rt_idx = rt_idx
        self.hr_idx = hr_idx
        # ---之前检查每个pair的情况和cluster时写的
        # self.hr2t_train_ls = []
        # self.rt2h_train_ls = []
        # self.hr2t_test_ls = []
        # self.rt2h_test_ls = []
        # for h, r, t in self.train_triples:
        #     '''保存三元组对应的cache的索引'''
        #     if (h, r) not in hr_idx:
        #         hr_idx[(h, r)] = hr_count
        #         hr_count += 1
        #         self.tail_cache_pos.append((h, r, t))
        #         #--按照索引添加
        #         self.hr2t_train_ls.append(list(self.hr2t_train[(h,r)]))
        #         self.hr2t_test_ls.append(list(self.hr2t_test[(h,r)]))
        #     self.tail_cache_idx.append(hr_idx[(h, r)])
        #     if (r, t) not in rt_idx:
        #         rt_idx[(r, t)] = rt_count
        #         rt_count += 1
        #         self.head_cache_pos.append((h, r, t))
        #         self.rt2h_train_ls.append(list(self.rt2h_train[(r,t)]))
        #         self.rt2h_test_ls.append(list(self.rt2h_test[(r,t)]))
        #     self.head_cache_idx.append(rt_idx[(r, t)])

        
        
        # self.args.headcache_num = len(rt_idx)
        # self.args.tailcache_num = len(hr_idx)
        # self.rt_idx = rt_idx
        # self.hr_idx = hr_idx
    
    def count_frequency(self):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        start = self.args.freq_start
        count = {}
        for head, relation, tail in self.train_triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    def get_pair2id(self):
        self.hr2id = {}
        self.rt2id = {}
        #--建立pair--idx的索引表
        with open(os.path.join(self.args.data_path, "hr2id.txt"), "r") as f:
            for line in f.readlines():
                pair, idx = line.strip().split("\t")
                pair1, pair2  = pair.split("_")
                pair = (int(pair1), int(pair2))
                idx = int(idx)
                self.hr2id[pair] = idx
        with open(os.path.join(self.args.data_path, "rt2id.txt"), "r") as f:
            for line in f.readlines():
                pair, idx = line.strip().split("\t")
                pair1, pair2  = pair.split("_")
                pair = (int(pair1), int(pair2))
                idx = int(idx)
                self.rt2id[pair] = idx
        


class BaseSampler(KGData):
    def __init__(self, args):
        super().__init__(args)
        self.get_hr2t_rt2h_from_train()
        # self.get_cache_inf()
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.hr2t_valid_test = ddict(set)
        self.rt2h_valid_test = ddict(set)
        self.hr2t_test = ddict(set)
        self.rt2h_test = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.get_hr2t_rt2h_from_valid_test()
        if self.args.use_sim:
            self.get_sim_ent()
        if self.args.rw_sans:
            self.k_neighbors = self.get_neighbor()

    def get_neighbor(self):
        k_mat = sparse.load_npz('matrix_FB15k-237_k2_nrw4000.npz')
        return k_mat
    def get_sim_ent(self):
        with open("hr2t_top20_all.json", "r") as f:
            self.hr2t_sim = json.load(f)
        with open("rt2h_top20_all.json", "r") as f:
            self.rt2h_sim = json.load(f)


    def get_hr2t_rt2h_from_valid_test(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        """与之前从训练集得到numpy字典不一样，这次得到tensor"""
        valid_test = set(self.valid_triples + self.test_triples)
        # for h, r, t in self.test_triples:
        for h, r, t in valid_test:
            self.hr2t_valid_test[(h, r)].add(t)
            self.rt2h_valid_test[(r, t)].add(h)
        for h, r in self.hr2t_valid_test:
            self.hr2t_valid_test[(h, r)] = torch.tensor(list(self.hr2t_valid_test[(h, r)]))
        for r, t in self.rt2h_valid_test:
            self.rt2h_valid_test[(r, t)] = torch.tensor(list(self.rt2h_valid_test[(r, t)]))
        
        for h, r, t in self.test_triples:
            self.hr2t_test[(h, r)].add(t)
            self.rt2h_test[(r, t)].add(h)
        for h, r in self.hr2t_test:
            self.hr2t_test[(h, r)] = torch.tensor(list(self.hr2t_test[(h, r)]))
        for r, t in self.rt2h_test:
            self.rt2h_test[(r, t)] = torch.tensor(list(self.rt2h_test[(r, t)]))

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        """与之前从训练集得到numpy字典不一样,这次得到tensor"""
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    # 突然发现numpy这种corrupt的方式挺好的，时间复杂度和空间复杂度都挺小的。本来还打算在实体池里先filter再rand sample
    # def corrupt_head(self, t, r, num_max=1):
    #     tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,))
    #     if not self.args.filter_flag:
    #         return tmp
    #     mask = torch.isin(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
    #     neg = tmp[mask]
    #     return neg

    # def corrupt_tail(self, h, r, num_max=1):
    #     tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,))
    #     if not self.args.filter_flag:
    #         return tmp
    #     mask = torch.isin(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
    #     neg = tmp[mask]
    #     return neg

    def corrupt_head(self, t, r, num_max=1):
        """Negative sampling of head entities.
        Args:
            t: Tail entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 
        Returns:
            neg: The negative sample of head entity filtering out the positive head entity.
        """
        if self.args.rw_sans:
            khop = self.k_neighbors[t].indices
            tmp = khop[np.random.randint(len(khop), size=num_max)].astype(
                    np.int64)
        else:
            tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):
        """Negative sampling of tail entities.
        Args:
            h: Head entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 
        Returns:
            neg: The negative sample of tail entity filtering out the positive tail entity.
        """
        if self.args.rw_sans:
            khop = self.k_neighbors[h].indices
            tmp = khop[np.random.randint(len(khop), size=num_max)].astype(
                    np.int64)
        else:
            tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):
        # return torch.randint(low = 0, high = self.ent_total, size = (neg_size, )).numpy()
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            # if self.args.use_sim:
            #     if "-".join([str(r), str(t)]) in self.rt2h_sim:
            #         mask = torch.isin(neg_tmp, torch.tensor(self.rt2h_sim["-".join([str(r), str(t)])]), assume_unique=True, invert=True)
            #         neg_tmp = neg_tmp[mask]
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        # return torch.randint(low = 0, high = self.ent_total, size = (neg_size, )).numpy()
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            # if self.args.use_sim:
            # #----是否使用同一relation下sim ent的信息
            #     if "-".join([str(h), str(r)]) in self.hr2t_sim:
            #         mask = torch.isin(neg_tmp, torch.tensor(self.hr2t_sim["-".join([str(h), str(r)])]), assume_unique=True, invert=True)
            #         neg_tmp = neg_tmp[mask]
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_all_true_triples(self):
        return self.all_true_triples


class RevSampler(KGData):
    def __init__(self, args):
        super().__init__(args)
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.add_reverse_relation()
        self.add_reverse_triples()
        self.get_hr2t_rt2h_from_train()

    def add_reverse_relation(self):
        """Get entity/relation/reverse relation id, and entity/relation number.
        Update:
            self.ent2id: Entity id.
            self.rel2id: Relation id.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """
        
        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            len_rel2id = len(self.rel2id)
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation + "_reverse"] = int(rid) + len_rel2id
                self.id2rel[int(rid) + len_rel2id] = relation + "_reverse"
        self.args.num_rel = len(self.rel2id)

    def add_reverse_triples(self):

        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        with open(os.path.join(self.args.data_path, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )


    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_all_true_triples(self):
        return self.all_true_triples    