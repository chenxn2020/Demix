# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import pytorch_lightning as pl
import lit_model
import data
def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--use_wandb", default=True, action='store_true')
    parser.add_argument('--norm_flag', default=False, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--filter_flag', default=True, action='store_false')
    parser.add_argument('--save_config', default=False, action='store_true')
    parser.add_argument('--load_config', default=False, action='store_true')
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--litmodel_name", default="MixupLitModel", type=str)
    parser.add_argument('--model_name', default="TransE", type=str)
    parser.add_argument('--loss_name', default="BCE_Loss", type=str)
    parser.add_argument('--dataset_name', default="FB15K237", type=str)
    parser.add_argument('--optim_name', default="Adam", type=str)
    parser.add_argument('--margin', default=12.0, type=float)
    parser.add_argument('--checkpoint_dir', default="", type=str)
    parser.add_argument('--regularization', '-r', default=0.0, type=float) 
    parser.add_argument('--mu', default=10, type=float, help='penalty coefficient for ComplEx_NNE')
    parser.add_argument('--emb_shape', default=20, type=int, help='Only on ConvE,The first dimension of the reshaped 2D embedding')
    parser.add_argument('--inp_drop', default=0.2, type=float, help='only on ConvE,Dropout for the input embeddings')
    parser.add_argument('--hid_drop', default=0.3, type=float, help='only on ConvE,Dropout for the hidden layer')
    parser.add_argument('--fet_drop', default=0.2, type=float, help='only on ConvE,Dropout for the convolutional features')
    parser.add_argument('--hid_size', default=9728, type=int, help='only on ConvE,The side of the hidden layer. The required size changes with the size of the embeddings.')
    parser.add_argument('--smoothing', default=0.1, type=float, help='only on ConvE,Make the label smooth')
    parser.add_argument('--dropout', default=0.5, type=float, help='only on CrossE,for Dropout')
    parser.add_argument('--neg_weight', default=50, type=int, help='only on CrossE, make up label')
    parser.add_argument('--slackness_penalty', default=0.01, type=float)
    parser.add_argument('--check_per_epoch', default=5, type=int)
    parser.add_argument('--early_stop_patience', default=10, type=int)
    parser.add_argument('--emb_dim', default=1000, type=int)
    parser.add_argument('--out_dim', default=200, type=int)
    parser.add_argument('--num_neg', default=64, type=int, help='负采样的个数')
    parser.add_argument('--num_ent', default=None, type=int)
    parser.add_argument('--num_rel', default=None, type=int)
    parser.add_argument('--freq_init', default=4, type=int)
    parser.add_argument('--eval_task', default="link_prediction", type=str)
    parser.add_argument('--adv_temp', default=0, type=float)
    parser.add_argument('--data_class', default="KGDataModule", type=str)
    parser.add_argument("--train_sampler_class",default="UniSampler",type=str)
    parser.add_argument("--test_sampler_class",default="TestSampler",type=str)
    parser.add_argument("--decoder_model", default=None, type=str)
    parser.add_argument("--opn", default='corr',type=str, help="only on CompGCN, choose Composition Operation")
    parser.add_argument("--calc_hits", default=[1,3,10], type=lambda s: [int(item) for item in s.split(',')], help='calc hits list')

    parser.add_argument("--leakage", default=False, action='store_true', help="训练构建字典是否发生数据泄漏")
    parser.add_argument("--calc_filter", default=True, action='store_true', help="计算指标是raw/filter")
    parser.add_argument("--wandb_offline", default=False, action='store_true', help="确认是否wandb同步")
    parser.add_argument("--valid_replace", default=None, type=str)
    parser.add_argument('--mix_neg', default=16, type=int) 
    

    parser.add_argument('--head_cache_size', default=50, type=int)
    parser.add_argument('--tail_cache_size', default=50, type=int)
    parser.add_argument('--headcache_num', default=0, type=int)
    parser.add_argument('--tailcache_num', default=0, type=int)
    parser.add_argument('--update_cache_epoch', default=1, type=int, help="每隔多少epoch更新一次cache")
    parser.add_argument('--cache_size', default=50, type=int)

    parser.add_argument("--warmup", default=50, type=int)
    parser.add_argument('--cluster_num', default=1000, type=int)
    parser.add_argument('--cluster_iter_num', default=5, type=int)

    #-----
    parser.add_argument('--search_sample_fs', default=False, action="store_true")
    parser.add_argument('--label_leakage', default=False, action="store_true")
    parser.add_argument('--use_multi_cluster', default=1, type=int)
    parser.add_argument('--use_weight', default=1, type=int)
    parser.add_argument('--freq_start', default=3, type=int)
    parser.add_argument('--use_pretrain', default=0, type=int)
    parser.add_argument('--lk_rel', default=0, type=int)
    parser.add_argument('--corr_logits', default=0.9, type=float, help="把预测概率大于一定值的负样本当作正样本")
    parser.add_argument('--use_sim', default=False, action="store_true")
    parser.add_argument('--negative_adversarial_sampling', default=False, action="store_true")

    parser.add_argument('--modulus_weight', default=3.5, type=float)
    parser.add_argument('--phase_weight', default=1.0, type=float)

    parser.add_argument('--adap_mixup', default=False, action="store_true")
    parser.add_argument('--self_neg', default=0, type=int)
    parser.add_argument('--pos_threshold', default=5, type=int)
    parser.add_argument('--negloss_weight', default=1.0, type=float)
    parser.add_argument('--topk', default=30, type=int)
    parser.add_argument('--neg_mode', default="normal", type=str)
    parser.add_argument('--neg_correct', default=1.0, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--mix_epoch', default=8, type=int)
    parser.add_argument('--no_harder', default=False, action='store_true')
    parser.add_argument('--cnd_size', default=30, type=int)
    parser.add_argument('--lr_begin', default=20, type=int) #学习率调节器的最初轮次
    parser.add_argument('--lr_step', default=30, type=int) #学习率调节器的步长
    parser.add_argument('--lr_change', default=0.1, type=float) #学习率调节器的变化率、
    parser.add_argument('--beta', default=5, type=float)
    parser.add_argument('--delta', default=1, type=float)
    parser.add_argument('--time', default=4, type=int)
    parser.add_argument('--regu_norm', default=3, type=int)
    parser.add_argument('--init_mode', default='uniform', type=str)
    parser.add_argument('--use_adv', default=False, action='store_true', help='结合self-adv')
    parser.add_argument('--no_cnd', default=False, action='store_true', help='去掉adamix模块')
    parser.add_argument('--correct_neg', default=False, action='store_true', help='去掉mpne模块')
    parser.add_argument('--rw_sans', default=False, action='store_true', help='结合rw-sans')

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model.BaseLitModel.add_to_argparse(lit_model_group)

    data_group = parser.add_argument_group("Data Args")
    data.BaseDataModule.add_to_argparse(data_group)


    parser.add_argument("--help", "-h", action="help")
    args = parser.parse_args()
    
    return args