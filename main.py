# -*- coding: utf-8 -*-
# from torch._C import T
# from train import Trainer
import pytorch_lightning as pl
from utils.setup_parser import setup_parser
from pytorch_lightning import seed_everything
from IPython import embed
import wandb
from utils.tools import *
from data.Sampler import *
import os
os.environ["WANDB_DISABLED"] = "true"

def calc_pair(sampler, data_path):
    #====统计训练集种每个pair下的样本数和验证集和测试集的样本数对应
    hr2t_train = sampler.hr2t_train
    rt2h_train = sampler.rt2h_train 
    hr2id_path  = "/".join([data_path, "hr2id.txt"])
    rt2id_path  = "/".join([data_path, "rt2id.txt"])
    with open(hr2id_path, "w") as f:
        for idx, key in enumerate(hr2t_train.keys()):
            f.write("_".join([str(key[0]), str(key[1])])+'\t'+str(idx)+'\n')
    with open(rt2id_path, "w") as f:
        for idx, key in enumerate(rt2h_train.keys()):
            f.write("_".join([str(key[0]), str(key[1])])+'\t'+str(idx)+'\n')

    # sampler.get_hr2t_rt2h_from_valid_test()
    # hr2t_valid_test = sampler.hr2t_valid_test
    # hr2t_test = sampler.hr2t_test
    # rt2h_valid_test = sampler.rt2h_valid_test 
    # rt2h_test = sampler.rt2h_test
    # with open("")
    # for key in hr2t_train.keys():
        


def main():
    args = setup_parser() #设置参数
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)
    if args.load_config:
        config_path = ""
        args = load_config(args, config_path)
    seed_everything(args.seed) 
    """set up sampler to datapreprocess""" #设置数据处理的采样过程
    train_sampler_class = import_class(f"data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # 这个sampler是可选择的
    "写入pair2id文件"
    if not os.path.exists("/".join([args.data_path, "rt2id.txt"])):
        calc_pair(train_sampler, args.data_path)
    train_sampler.get_pair2id()

    test_sampler_class = import_class(f"data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  # test_sampler是一定要的
    """set up datamodule""" #设置数据模块
    data_class = import_class(f"data.{args.data_class}") #定义数据类 DataClass
    kgdata = data_class(args, train_sampler, test_sampler)
    """set up model"""
    model_class = import_class(f"model.{args.model_name}")
    if args.model_name == "ComplEx_NNE_AER":
        model = model_class(args, train_sampler.rel2id)
    else:
        model = model_class(args)

    # '''初始化cache'''
    # pair_cache_list = train_sampler.get_train_caches()

    """set up lit_model"""
    litmodel_class = import_class(f"lit_model.{args.litmodel_name}")
    lit_model = litmodel_class(model, args, train_sampler)
    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    '''标记实验是否是数据泄露情况'''
    is_leakage = 'sample_lk' if args.leakage else 'sample_nk'
    is_filter = 'filter' if args.calc_filter else 'raw'
    if args.use_wandb:
        # if args.loss_name == "Adv_Loss":
        #     if args.litmodel_name == 'KGECacheLitmodel':
        #         log_name = "_".join([args.model_name, args.dataset_name, str(args.lr), is_leakage, \
        #             str(args.adv_temp), str(args.cache_size), str(args.alpha)])
        #     else:
        #         log_name = "_".join([args.model_name, args.dataset_name, str(args.lr), is_leakage, \
        #             str(args.adv_temp)])
        # else:
        if args.label_leakage:
            log_name = "_".join([args.model_name, args.dataset_name, args.litmodel_name, str(args.lr), "label_lk"])
        else:
            lr_number = "lr" + str(args.lr)
            lr_begin = "lrb" + str(args.lr_begin)
            lr_step = "lrs" + str(args.lr_step)
            lr_change = "lrc" + str(args.lr_change)
            mix_epoch = 'mix' + str(args.mix_epoch)
            pos = "pos" + str(args.pos_threshold)
            num_neg = "NSG" + str(args.num_neg)
            if args.adv_temp:
                num_neg = num_neg + "adv"

            # log_name = "_".join([args.model_name, args.dataset_name, args.litmodel_name, str(args.lr), is_leakage])
            log_name = "_".join([args.model_name, args.dataset_name, num_neg, mix_epoch, pos, lr_number, lr_begin, lr_step, lr_change])
        # logger = pl.loggers.WandbLogger(name=log_name, project="NeuroKR", offline=args.wandb_offline)
        logger = pl.loggers.WandbLogger(name=log_name, project="NSGenerating", offline=args.wandb_offline)
        # logger = pl.loggers.WandbLogger(name=log_name, project="NeuroKR", offline=True)
        logger.log_hyperparams(vars(args))
    """early stopping"""
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval|hits@10",
        mode="max",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )
    """set up model save method"""
    # 目前是保存在验证集上mrr结果最好的模型
    # 模型保存的路径
    # dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name, is_leakage])
    if args.adap_mixup:
        aa = "AdaptiveMixup"
    else:
        aa = "normal"
    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name, aa])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval|hits@10",
        mode="max",
        filename="{epoch}-{Eval|hits@10:.4f}",
        dirpath=dirpath,
        save_weights_only=True,
        save_top_k=1,
    )
    callbacks = [early_callback, model_checkpoint]
    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus="0,",
        check_val_every_n_epoch=args.check_per_epoch,
        # reload_dataloaders_every_n_epochs=5
    )
    '''保存参数到config'''
    if args.save_config:
        save_config(args)

    # args.checkpoint_dir="output/link_prediction/FB15K237/TransE/no_lk/epoch=4-Eval|mrr=0.330.ckpt"
    # path = "output/link_prediction/FB15K237/TransE/no_lk/epoch=29-Eval|mrr=0.330.ckpt"
    # lit_model.load_state_dict(torch.load(path)["state_dict"])
    if not args.test_only:
        # train&valid
        #----先nolk训练，后lk训练，看有没有提升
        if args.use_pretrain == 1:
            # path = 'output/link_prediction/FB15K237/TransE/no_lk/epoch=34-Eval|mrr=0.320.ckpt'
            # path = 'output/link_prediction/FB15K237/TransE/sample_nk/epoch=189-Eval|hits@10=0.5121.ckpt'
            # path = 'output/link_prediction/FB15K237/TransE/leakage/epoch=264-Eval|mrr=0.560.ckpt'
            lit_model.load_state_dict(torch.load(path)["state_dict"])
            # lit_model.evpdata
        lit_model.train()
        #--------------
        trainer.fit(lit_model, datamodule=kgdata)
        # 加载本次实验中dev上表现最好的模型，进行test
        path = model_checkpoint.best_model_path
    else:
        # path = args.checkpoint_dir
        # path = "output/link_prediction/FB15K237/TransE/normal/epoch=114-Eval|hits@10=0.5200.ckpt"
        # path = 'output/link_prediction/FB15K237/RotatE/normal/epoch=69-Eval|hits@10=0.5120.ckpt'
        # path = 'output/link_prediction/FB15K237/HAKE/normal/epoch=169-Eval|hits@10=0.5383.ckpt'

        path = 'output/link_prediction/FB15K237/TransE/normal/epoch=84-Eval|hits@10=0.5152.ckpt'
        # path = '/data/chenxn/NeuralKG/output/link_prediction/FB15K237/TransE/epoch=29-Eval|mrr=0.300.ckpt'

        # path = 'output/link_prediction/WN18RR/TransE/no_lk/epoch=164-Eval|mrr=0.220.ckpt'
        # path = 'output/link_prediction/FB15K237/TransE/no_lk/epoch=34-Eval|mrr=0.330.ckpt'
        #---正常训练方式
        # path = 'output/link_prediction/FB15K237/TransE/no_lk/epoch=34-Eval|mrr=0.320.ckpt'
        #--------用了cache，进行泄漏的结果
        # path = 'output/link_prediction/FB15K237/TransE/leakage/epoch=264-Eval|mrr=0.560.ckpt'
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    # lit_model.save_dim()

    trainer.test(lit_model, datamodule=kgdata)

# def collect_candidate_entity(lit_model, kgdata):
#     trainer.test(lit_model, datamodule=kgdata)
if __name__ == "__main__":
    main()
