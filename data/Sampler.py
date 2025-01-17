from numpy.random.mtrand import normal
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import random
from .DataPreprocess import *
from IPython import embed
# import dgl 
import torch.nn.functional as F
import time
import queue
from scipy import sparse

'''TransE提出的随机负采样'''
class UniSampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        self.cross_sampling_flag = 0
        
    def sampling(self, data):
        batch_data = {}
        neg_ent_sample = []
        subsampling_weight = []
        pair2id_ls = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t in data:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_head)
                pair2id_ls.append(self.rt2id[(r, t)])
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t in data:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_tail)
                pair2id_ls.append(self.hr2id[(h, r)])
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)

        batch_data["positive_sample"] = torch.LongTensor(data)
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        batch_data['pair_id'] = torch.tensor(np.array(pair2id_ls))
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(np.array(subsampling_weight)))
        return batch_data

    def sampling2(self, data):
        batch_data = {}
        neg_ent_head = []
        neg_ent_tail = []
        subsampling_weight = []
        for h, r, t in data:
            neg_head = self.head_batch(h, r, t, self.args.num_neg)
            neg_ent_head.append(neg_head)
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_ent_tail.append(neg_tail)
            if self.args.use_weight:
                weight = self.count[(h, r)] + self.count[(t, -r-1)]
                subsampling_weight.append(weight)

        batch_data["positive_sample"] = torch.LongTensor(data)
        batch_data['negative_head'] = torch.LongTensor(np.array(neg_ent_head))
        batch_data['negative_tail'] = torch.LongTensor(np.array(neg_ent_tail))
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(np.array(subsampling_weight)))
        return batch_data
    
    def batch_sampling(self, data):
        batch_data = {}
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        pos_sample = torch.tensor(data)
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            neg_sample = pos_sample[:, 0]
        else:
            batch_data['mode'] = "tail-batch"
            neg_sample = pos_sample[:, -1]
        if self.args.use_weight:
            subsampling_weight = []
            for h, r, t in data:
                weight = self.count[(h, r)] + self.count[(t, -r-1)]
                subsampling_weight.append(weight)

        batch_data["positive_sample"] = pos_sample
        batch_data['negative_sample'] = neg_sample
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(subsampling_weight))
        return batch_data

    def sampling_cache(self, data):
        batch_data = {}
        neg_ent_sample = []
        subsampling_weight = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t, _, _ in data:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_head)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t, _, _ in data:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_tail)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(subsampling_weight))
        return batch_data
    
    def sampling_all(self, data):
        #进行全量数据的bce loss时使用
        batch_data = {}
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        table = torch.zeros(len(data), self.args.num_ent)
        for id, (h, r, t) in enumerate(data):
            if self.cross_sampling_flag == 0:
                batch_data["mode"] = "head-batch"
                rt_sample = self.rt2h_train[(r, t)]
                table[id][rt_sample] = 1

            else:
                batch_data["mode"] = "tail-batch"
                hr_sample = self.hr2t_train[(h, r)]
                table[id][hr_sample] = 1
        batch_data["sample"] = torch.LongTensor(data)
        batch_data["label"] = table.float()
        return batch_data

    def sampling_with_label(self, data):
        #只对正样本和随机采样的负样本进行bce loss
        batch_data = {}
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        # self.cross_sampling_flag = 1
        neg_label = torch.zeros(len(data), self.args.num_neg) #只记录负样本的标签，leakage的时候做处理。否则就是全0
        # neg_label = torch.zeros(len(data), self.args.num_ent) #只记录负样本的标签，leakage的时候做处理。否则就是全0
        neg_ent_sample = []
        subsampling_weight = []
        train_label = torch.zeros(len(data), self.args.num_ent)
        
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for idx, cache_inf in enumerate(data):
                h, r, t, _, _ = cache_inf
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                # neg_head = [i for i in range(self.args.num_ent)]
                neg_ent_sample.append(neg_head)
                # if self.args.label_leakage and r == self.args.lk_rel:
                if self.args.label_leakage :
                    #此处做标签的泄漏
                    label_lk = self.search_real_pos(h, r, t, neg_head, "head-batch") ##真实的标签
                    neg_label[idx] = label_lk
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)
                train_label[idx][self.rt2h_train[(r, t)]] = 1.0
        else:
            batch_data['mode'] = "tail-batch"
            for idx, cache_inf in enumerate(data):
                h, r, t, _, _ = cache_inf
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                # neg_tail = [i for i in range(self.args.num_ent)]
                neg_ent_sample.append(neg_tail)
                # if self.args.label_leakage and r == self.args.lk_rel:
                if self.args.label_leakage :
                    #此处做标签的泄漏
                    label_lk = self.search_real_pos(h, r, t, neg_tail, "tail-batch") ##真实的标签
                    neg_label[idx] = label_lk
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)
                train_label[idx][self.hr2t_train[(h, r)]] = 1.0

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        batch_data["neg_label"] = neg_label
        batch_data["train_label"] = train_label
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(subsampling_weight))
        return batch_data


    def search_real_pos(self, h, r, t, neg_sample, mode):
        #对随机采样的无标签样本查看哪些时真正的正样本
        if mode == "head-batch":
            all_pos_sample = self.rt2h_all[(r, t)]
            neg_sample = torch.tensor(neg_sample)
            label = torch.isin(neg_sample, all_pos_sample).float() ##存在label为1，否则为0
        else:
            all_pos_sample = self.hr2t_all[(h, r)]
            neg_sample = torch.tensor(neg_sample)
            label = torch.isin(neg_sample, all_pos_sample).float() ##存在label为1，否则为0
        return label


        
    

    def sampling_caches_std(self, data):
        batch_data = {}
        neg_head_set = []
        neg_tail_set = []
        head_cache_idx_list = []
        tail_cache_idx_list = []
        head_cache_idx_set = []
        tail_cache_idx_set = []

        for h, r, t, head_cache_idx, tail_cache_idx in data:
            '''找head的candidate'''
            head_cache_idx_list.append(head_cache_idx)
            '''这样做是保证随机采样的candidate对应的cache都是unique的'''
            if head_cache_idx not in head_cache_idx_set:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_head_set.append(neg_head)
                head_cache_idx_set.append(head_cache_idx)
            '''找tail的candidate'''
            tail_cache_idx_list.append(tail_cache_idx)
            if tail_cache_idx not in tail_cache_idx_set:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_tail_set.append(neg_tail)
                tail_cache_idx_set.append(tail_cache_idx)


        batch_data["positive_sample"] = torch.LongTensor(data)
        batch_data['neg_head_set'] = torch.LongTensor(neg_head_set)
        batch_data['neg_tail_set'] = torch.LongTensor(neg_tail_set)
        batch_data["head_cache_idx_list"] = torch.LongTensor(head_cache_idx_list)
        batch_data["tail_cache_idx_list"] = torch.LongTensor(tail_cache_idx_list)
        #----对cache_idx去重，保证每次step不会对某些相同的cache更新多次
        batch_data["head_cache_idx_set"] = torch.LongTensor(head_cache_idx_set)
        batch_data["tail_cache_idx_set"] = torch.LongTensor(tail_cache_idx_set)

        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'mode']
    
    


"""SRNS负采样"""
class SrnsSampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        self.train_set = self.hr2t_train

    def sampling(self, data):
        batch_data = {}
        batch_data["positive_sample"] = torch.LongTensor(data)
        batch_data["train_set"] = self.train_set

        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample','train_set']

'''TransH提出的随机负采样'''
class BernSampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        self.lef_mean, self.rig_mean = self.calc_bern()
        self.save_replace_head_prob()
    
    def save_replace_head_prob(self):
        '''这里记录对于一个relation来说，需要替换尾实体的概率有多大'''
        replace_head_prob = ddict(float)
        for r in self.lef_mean:
            replace_head_prob[r] = self.rig_mean[r] / (self.lef_mean[r] + self.rig_mean[r])
        with open(os.path.join(self.args.data_path, "replace_tail_prob.txt"), "w") as f:
            for r in range(self.args.num_rel):
                f.write(str(r) + '\t' + str(replace_head_prob[r]) + '\n')

        
    def __normal_batch(self, h, r, t, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        '''使用伯努利分布来选择是替换头实体还是尾实体
		原理是prob表示的是关系r组成的三元组中不同尾实体个数的占比,
		prob越大表示关系是many-to-one的可能性越高，越应该替换尾实体'''
        prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.args.bern_flag else 0.5
        for i in range(neg_size):
            if random.random() >= prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.corrupt_head(t, r, num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)

        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.corrupt_tail(h, r, num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        return np.hstack((neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]))
    def sampling(self, data):
        batch_data = {}
        neg_ent_sample = []
        batch_data['mode'] = 'bern'
        for h, r, t in data:
            neg_ent_sample = self.__normal_batch(h, r, t, self.args.num_neg)
        batch_data["positive_sample"] = torch.LongTensor(data)
        batch_data['negative_sample'] = torch.LongTensor(neg_ent_sample)
        return batch_data
    
    def calc_bern(self):
        h_of_r = ddict(set)
        t_of_r = ddict(set)
        freqRel = ddict(float)
        lef_mean = ddict(float)
        rig_mean = ddict(float)
        for h, r, t in self.train_triples:
            freqRel[r] += 1.0
            h_of_r[r].add(h)
            t_of_r[r].add(t)
        for r in h_of_r:
            lef_mean[r] = freqRel[r] / len(h_of_r[r])
            rig_mean[r] = freqRel[r] / len(t_of_r[r])
        return lef_mean, rig_mean

    @staticmethod
    def sampling_keys():
        return ['positive_sample', 'negative_sample', 'mode']

'''
RotatE提出的自对抗负采样,
这个类似于一个插件，可以结合uni/bern一起使用
'''
class AdvSampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        self.freq_hr, self.freq_tr = self.calc_freq()
    def sampling(self, pos_sample):
        data = pos_sample.numpy().tolist()
        adv_sampling = []
        for h, r, t in data:
            weight = self.freq_hr[(h, r)] + self.freq_tr[(t, r)]
            adv_sampling.append(weight)
        adv_sampling = torch.tensor(adv_sampling, dtype=torch.float32).cuda()
        adv_sampling = torch.sqrt(1 / adv_sampling)
        return adv_sampling
    def calc_freq(self):
        freq_hr, freq_tr = {}, {}
        for h, r, t in self.train_triples:
            if (h, r) not in freq_hr:
                freq_hr[(h, r)] = self.args.freq_init
            else:
                freq_hr[(h, r)] += 1
            if (t, r) not in freq_tr:
                freq_tr[(t, r)] = self.args.freq_init
            else:
                freq_tr[(t, r)] += 1
        return freq_hr, freq_tr

"""ConvE提出的1vsall负采样"""

class AllSampler(RevSampler):
    # TODO:类名还需要商榷下
    def __init__(self, args):
        super().__init__(args)
        # self.num_rel_without_rev = self.args.num_rel // 2
        
    def sampling(self, data):
        '''全部做tail_prediction'''
        # sample_id = [] #确定triple里的relation是否是reverse的。reverse为1，不是为0
        batch_data = {}
        table = torch.zeros(len(data), self.args.num_ent)
        for id, (h, r, _) in enumerate(data):
            hr_sample = self.hr2t_train[(h, r)]
            table[id][hr_sample] = 1
            # if r > self.num_rel_without_rev:
            #     sample_id.append(1)
            # else:
            #     sample_id.append(0)
        batch_data["sample"] = torch.LongTensor(data)
        batch_data["label"] = table.float()
        # batch_data["sample_id"] = torch.LongTensor(sample_id)
        return batch_data

    def sampling_keys(self):
        return ["sample", "label"]
    
class CrossESampler(BaseSampler):
    # TODO:类名还需要商榷下
    def __init__(self, args):
        super().__init__(args)
        self.neg_weight = float(self.args.neg_weight / self.args.num_ent)
    def sampling(self, data):
        '''一个样本同时做head/tail prediction'''
        batch_data = {}
        hr_label = self.init_label(len(data))
        tr_label = self.init_label(len(data))
        for id, (h, r, t) in enumerate(data):
            hr_sample = self.hr2t_train[(h, r)]
            hr_label[id][hr_sample] = 1.0
            tr_sample = self.rt2h_train[(r, t)]
            tr_label[id][tr_sample] = 1.0
        batch_data["sample"] = torch.LongTensor(data)
        batch_data["hr_label"] = hr_label.float()
        batch_data["tr_label"] = tr_label.float()
        return batch_data

    def init_label(self, row):
        label = torch.rand(row, self.args.num_ent)
        label = (label > self.neg_weight).float()
        label -= 1.0
        return label

    def sampling_keys(self):
        return ["sample", "label"]

'''图神经网络中以图为单位采样'''

class GraphSampler(RevSampler):
    def __init__(self, args):
        super().__init__(args)
        self.entity   = None
        self.relation = None
        self.triples  = None
        self.graph    = None
        self.norm     = None
        self.label    = None

    def sampling(self, pos_triples):
        batch_data = {}
        
        pos_triples = np.array(pos_triples)
        pos_triples, self.entity = self.sampling_postive(pos_triples)
        head_triples = self.sampling_negative('head', pos_triples, self.args.num_neg)
        tail_triples = self.sampling_negative('tail', pos_triples, self.args.num_neg)
        self.triples = np.concatenate((pos_triples,head_triples,tail_triples))
        batch_data['entity']  = self.entity
        batch_data['triples'] = self.triples
        
        self.label = torch.zeros((len(self.triples),1))
        self.label[0 : self.args.train_bs] = 1
        batch_data['label'] = self.label
        
        split_size = int(self.args.train_bs * 0.5) 
        graph_split_ids = np.random.choice(
            self.args.train_bs,
            size=split_size, 
            replace=False
        )
        head,rela,tail = pos_triples.transpose()
        head = torch.tensor(head[graph_split_ids], dtype=torch.long).contiguous()
        rela = torch.tensor(rela[graph_split_ids], dtype=torch.long).contiguous()
        tail = torch.tensor(tail[graph_split_ids], dtype=torch.long).contiguous()
        self.graph, self.relation, self.norm = self.build_graph(len(self.entity), (head,rela,tail))
        batch_data['graph']    = self.graph
        batch_data['relation'] = self.relation
        batch_data['norm']     = self.norm

        return batch_data

    def get_sampling_keys(self):
        return ['graph','triples','label','entity','relation','norm']

    def sampling_negative(self, mode, pos_triples, num_neg):
        
        neg_random = np.random.choice(
            len(self.entity), 
            size = num_neg * len(pos_triples)
        )
        neg_samples = np.tile(pos_triples, (num_neg, 1))
        if mode == 'head':
            neg_samples[:,0] = neg_random
        elif mode == 'tail':
            neg_samples[:,2] = neg_random
        return neg_samples

    def build_graph(self, num_ent, triples):
        if self.args.model_name == 'RGCN':
            power = -1
        elif self.args.model_name == 'CompGCN':
            power = -0.5
        triples = np.array(triples).transpose()
        head, rela, tail = triples[:,0], triples[:,1], triples[:,2]
        #graph = dgl.graph(([], []))
        graph = dgl.DGLGraph()
        graph.add_nodes(num_ent)
        graph.add_edges(head, tail)
        node_norm = self.comp_deg_norm(graph, power)
        #edge_norm = self.node_norm_to_edge_norm(graph, node_norm.view(-1,1))
        edge_norm = self.node_norm_to_edge_norm(graph,node_norm)
        rela = torch.tensor(rela)
        return graph, rela, edge_norm

    def comp_deg_norm(self, graph, power=-1):
        graph = graph.local_var()
        in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().numpy()
        norm = in_deg.__pow__(power)
        norm[np.isinf(norm)] = 0
        #return torch.from_numpy(norm.astype('int64'))
        return torch.from_numpy(norm)

    def node_norm_to_edge_norm(slef, graph, node_norm):
        graph = graph.local_var()
        # convert to edge norm
        graph.ndata['norm'] = node_norm
        graph.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
        return graph.edata['norm']

    def sampling_postive(self,postive_triples):

        edges = np.random.choice(
            np.arange(len(postive_triples)),
            size = self.args.train_bs,
            replace=False
        )
        edges = postive_triples[edges]
        head, rela, tail = np.array(edges).transpose()
        entity, index = np.unique((head, tail), return_inverse=True) 
        head, tail = np.reshape(index, (2, -1))

        return np.stack((head,rela,tail)).transpose(), \
                torch.from_numpy(entity).view(-1,1).long()

class KBATSampler(GraphSampler):  #note: 这里原论文用的单向 此处用双向 
    def __init__(self, args): 
        super().__init__(args)
        self.n_hop           = None
        self.graph           = None
        self.neighbours      = None
        self.adj_matrix      = None
        self.entity          = None
        self.triples_GAT_pos = None
        self.triples_GAT_neg = None
        self.triples_Con     = None
        self.label           = None

        self.get_neighbors()

    def sampling(self, pos_triples):
        batch_data = {}
        #--------------------KBAT-Sampler------------------------------------------
        self.entity = self.get_unique_entity(pos_triples)
        head_triples = self.sampling_negative('head', pos_triples, 1)
        tail_triples = self.sampling_negative('tail', pos_triples, 1)
        self.triples_GAT_pos = torch.tensor(pos_triples).repeat(self.args.num_neg*2, 1)
        self.triples_GAT_neg = torch.tensor(np.concatenate((head_triples, tail_triples)))
        batch_data['triples_GAT_pos'] = self.triples_GAT_pos
        batch_data['triples_GAT_neg'] = self.triples_GAT_neg

        head, rela, tail = torch.tensor(self.train_triples).t()
        self.adj_matrix  = (torch.stack((tail, head)), rela) # note: 源代码就是先尾后头
        batch_data['adj_matrix'] = self.adj_matrix

        self.n_hop = self.get_batch_nhop_neighbors_all()
        batch_data['n_hop'] = self.n_hop
        #--------------------ConvKB-Sampler------------------------------------------

        pos_triples = pos_triples[:128]
        head_triples = self.sampling_negative('head', pos_triples, 20)
        tail_triples = self.sampling_negative('tail', pos_triples, 20)
        self.triples_Con = np.concatenate((pos_triples, head_triples, tail_triples))
        self.label = -torch.ones((len(self.triples_Con),1))
        self.label[0 : 128] = 1
        batch_data['triples_Con'] = self.triples_Con
        batch_data['label'] = self.label

        return batch_data

    def get_sampling_keys(self):
        return [ 'adj_matrix', 'n_hop', 'triples_GAT_pos', 
        'triples_GAT_neg', 'triples_Con' , 'label']

    def bfs(self, graph, source, nbd_size=2): # todo: 这个完全运行需要20分钟，能否优化？
        """
        返回neigbors 记录source的前两跳关系和实体
        """
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:
                            continue
                        parent[target] = (top[0], graph[top[0]][target]) # 记录父亲节点id和关系id

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1]))) #删除已知的source 记录前两跳实体及关系
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors

    def get_neighbors(self, nbd_size=2):
        """返回neighors记录source在两跳邻域内的关系和实体"""
        self.graph = {}

        for triple in self.train_triples:
            head = triple[0]
            rela = triple[1]
            tail = triple[2]

            if(head not in self.graph.keys()):
                self.graph[head] = {}
                self.graph[head][tail] = rela
            else:
                self.graph[head][tail] = rela

        neighbors = {}
        import pickle
        print("Opening node_neighbors pickle object")
        file = self.args.data_path + "/2hop.pickle"
        with open(file, 'rb') as handle:
            self.neighbours = pickle.load(handle)  
        return

        start_time = time.time()
        print("Start Graph BFS")
        for head in self.graph.keys():
            temp_neighbors = self.bfs(self.graph, head, nbd_size)
            for distance in temp_neighbors.keys():
                if(head in neighbors.keys()):
                    if(distance in neighbors[head].keys()):
                        neighbors[head][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[head][distance] = temp_neighbors[distance]
                else:
                    neighbors[head] = {}
                    neighbors[head][distance] = temp_neighbors[distance]

        print("Finish BFS, time taken ", time.time() - start_time)
        self.neighbours = neighbors

    def get_unique_entity(self, triples):
        train_triples = np.array(triples)
        train_entities = np.concatenate((train_triples[:,0], train_triples[:,2]))
        return np.unique(train_entities)

    def get_batch_nhop_neighbors_all(self, nbd_size=2):
        
        batch_source_triples = []
        
        for source in self.entity:
            if source in self.neighbours.keys():
                nhop_list = self.neighbours[source][nbd_size]
                for i, tup in enumerate(nhop_list):
                    if(i >= 2):  # note: 这里如果是FB则仅使用一个2跳邻居 如果是WN则使用全部
                        break
                    batch_source_triples.append([source, 
                                                tup[0][-1], 
                                                tup[0][0],
                                                tup[1][0]])

        n_hop =  np.array(batch_source_triples).astype(np.int32)
        
        return torch.autograd.Variable(torch.LongTensor(n_hop))

class CompGCNSampler(AllSampler, GraphSampler):
    def __init__(self, args):
        super().__init__(args)
        self.relation = None
        self.triples  = None
        self.graph    = None
        self.norm     = None
        self.label    = None

        self.graph, self.relation, self.norm = \
            self.build_graph(self.args.num_ent, np.array(self.train_triples).transpose())

    def sampling(self, pos_triples):
        batch_data = {}
        
        batch_data = AllSampler.sampling(self, pos_triples)

        batch_data['graph']    = self.graph
        batch_data['relation'] = self.relation
        batch_data['norm']     = self.norm

        return batch_data

    def get_sampling_keys(self):
        return ['sample','label','graph','relation','norm']

    def node_norm_to_edge_norm(self, graph, node_norm):
        graph.ndata['norm'] = node_norm
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        norm = graph.edata.pop('norm').squeeze()
        return norm  #返回边的id（包括负边） 和 边的norm值


'''测试/验证的时候使用'''
class TestSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        """与之前从训练集得到numpy字典不一样，这次得到tensor"""
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        # sample_id = []
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail = triple            
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label"]

class GraphTestSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent
        self.triples = sampler.train_triples

    def get_hr2t_rt2h_from_all(self):
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            # from IPython import embed;embed();exit()
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        
        head, rela, tail = np.array(self.triples).transpose()
        graph, rela, norm = self.sampler.build_graph(self.num_ent, (head, rela, tail))
        batch_data["graph"]  = graph
        batch_data["rela"]   = rela
        batch_data["norm"]   = norm
        batch_data["entity"] = torch.arange(0, self.num_ent, dtype=torch.long).view(-1,1)
        
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label",\
             "graph", "rela", "norm", "entity"]
             
'''继承torch.Dataset'''
class KGDataset(Dataset):

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]