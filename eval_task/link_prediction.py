from numpy import DataSource
import torch
import os
import json
from IPython import embed


def link_predict(batch, model=None, prediction="all", calc_filter=True, id2ent=None, id2rel=None):
    if prediction == "all":
        head_ranks = head_predict(batch, model, calc_filter)
        tail_ranks = tail_predict(batch, model, calc_filter)
        # head_ranks, head_score = head_predict(batch, model, calc_filter)
        # tail_ranks, tail_score = tail_predict(batch, model, calc_filter)
        # valid_head_path = "/home/chenxn/NSGenerating/dataset/FB15K237/valid_head.txt"
        # valid_tail_path = "/home/chenxn/NSGenerating/dataset/FB15K237/valid_tail.txt"
        # sample = batch["positive_sample"]
        # save_valid_replace(valid_head_path, head_ranks, sample, id2ent, id2rel, 'head')
        # save_valid_replace(valid_tail_path, tail_ranks, sample, id2ent, id2rel, 'tail')
        # embed();exit()
        ranks = torch.cat([head_ranks, tail_ranks])
    elif prediction == "head":
        ranks = head_predict(batch, model, calc_filter)
    elif prediction == "tail":
        ranks = tail_predict(batch, model, calc_filter)

    return ranks.float()

def save_valid_replace(path, ranks, sample, id2ent, id2rel, label):
    idx = 0
    # file = open(path, "w")
    with open(path, "w") as file:
        while idx <= 15:
            if label == 'head':
                tmp = 0
                while ranks[idx][tmp].item() == sample[idx][0].item():
                    tmp += 1
                head = ranks[idx][tmp].item()
                relation = sample[idx][1].item()
                tail = sample[idx][2].item()
                file.write(id2ent[head]+'\t'+id2rel[relation]+'\t'+id2ent[tail]+'\n')
            else:
                tmp = 0
                while ranks[idx][tmp].item() == sample[idx][2].item():
                    tmp += 1
                head = sample[idx][0].item()
                relation = sample[idx][1].item()
                tail = ranks[idx][tmp].item()
                file.write(id2ent[head]+'\t'+id2rel[relation]+'\t'+id2ent[tail]+'\n')
            idx += 1


def head_predict(batch, model, calc_filter):
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 0]
    label = batch["head_label"]
    pred_score = model.get_score(batch, "head_predict")
    return calc_ranks(idx, label, pred_score, calc_filter)


def tail_predict(batch, model, calc_filter):
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 2]
    label = batch["tail_label"]
    pred_score = model.get_score(batch, "tail_predict")
    return calc_ranks(idx, label, pred_score, calc_filter)


def calc_ranks(idx, label, pred_score, calc_filter):
    """Calculating triples score ranks

    Args:
        idx ([type]): The id of the entity to be predicted
        label ([type]): The id of existing triples, to calc filtered results
        pred_score ([type]): The score of the triple predicted by the model

    Returns:
        tensor: The rank of the triple to be predicted, dim [batch_size]
    """

    b_range = torch.arange(pred_score.size()[0])
    if calc_filter:
        target_pred = pred_score[b_range, idx]
        pred_score = torch.where(label.bool(), -torch.ones_like(pred_score) * 10000000, pred_score)
        pred_score[b_range, idx] = target_pred
    # -------构建test32.json
    # aa = torch.argsort(pred_score, dim=1, descending=True)
    # ss = dict()
    # top10_head = aa[:,:10].cpu().tolist()
    # pos_triple = pos_triple.cpu().tolist()
    # for idx, triple in enumerate(pos_triple):
    #     ss['-'.join([str(_) for _ in triple])] = top10_head[idx]
    # with open("./dataset/FB15K237/test32.json", "w") as f:
    #     json.dump(ss, f)
    # embed();exit()
    # return aa, pred_score
    ranks = (
        1
        + torch.argsort(
            torch.argsort(pred_score, dim=1, descending=True), dim=1, descending=False
        )[b_range, idx]
    )
    return ranks