import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Adv_Loss_1(nn.Module):
    def __init__(self, args, model):
        super(Adv_Loss_1, self).__init__()
        self.args = args
        self.model = model
    def forward(self, pos_score, neg_score, mask_ls=None, epoch=None, pos_cluster=None, neg_cluster=None):
        
        
        # neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
        #             * F.logsigmoid(-neg_score)).sum(dim=1)  #shape:[bs]
        #-----rotate的计算方式,只关注样本的得分
        neg_quality = neg_score
        #------------------
        #------只关注低于正样本分数的负样本
        # neg_quality = torch.where(neg_score<pos_score-2.0, neg_score, -torch.ones_like(neg_score) * 1000)
        #-------根据聚类的结果，删去同一簇的ent
        # pos_sample, neg_sample, mode, dis2cluster, cluster_id, epoch = aa
        # #TODO: 还需要试下incremental的设定，以及多个簇的选用
        if epoch != 0:
        # if epoch >= 15:
        #--------考虑多个cluster
            neg_quality = torch.where(mask_ls.cuda() == True, neg_score, -torch.ones_like(neg_score) * 1000)
            # neg_quality = torch.where(mask_ls.cuda() != True, neg_score, -torch.ones_like(neg_score) * 1000)
        #     if mode == 'tail-batch':
        #         pos = pos_sample[:,2]
        #     else:
        #         pos = pos_sample[:, 0]
        #     pos_cluster = cluster_id[pos]
        #     neg_cluster = cluster_id[neg_sample.view(-1)].view(pos.shape[0], -1)
        #------------只看单个cluster
            # neg_quality = torch.where(pos_cluster.cuda() == neg_cluster.cuda(), -torch.ones_like(neg_score) * 1000, neg_score)
        #-------------
        neg_score = (F.softmax(neg_quality * self.args.adv_temp, dim=1).detach()
                    * F.logsigmoid(-neg_score)).sum(dim=1)  #shape:[bs]
        pos_score = F.logsigmoid(pos_score) #shape:[bs, 1]
        # from IPython import embed;embed();exit()
        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2
    
        if self.args.model_name == 'ComplEx' or self.args.model_name == 'DistMult':
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
            loss = loss + regularization
        return loss