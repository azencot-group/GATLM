import argparse

import torch
from FEDformer.exp.exp_id_curv import Exp_id_curv
import random
import numpy as np
from utils import get_settings
from config import dataset_defaults

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=0, help='status')
parser.add_argument('--task_id', type=str, default='ETTh2', help='task id')
parser.add_argument('--model', type=str, default='FEDformer',
                    help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')
parser.add_argument('--seed', type=int, default=1, help='status')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# data loader
parser.add_argument('--data', type=str, default='ETTh2', help='dataset type')
parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
parser.add_argument('--save_path', type=str, default='./', help='save path for intrinsic dimensions and curvatures')
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--detail_freq', type=str, default='h', help='like freq, but use in predict')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
# parser.add_argument('--cross_activation', type=str, default='tanh'

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

# intrinsic dimension
parser.add_argument('--id_samples', type=int, default=500000, help='latent samples for intrinsic dimension estimation')
parser.add_argument('--curv_samples', type=int, default=100000, help='latent samples for curvature estimation')

args = parser.parse_args()
args_dict = args.__dict__
dict_name = args.task_id
args_dict.update(dataset_defaults[dict_name])

args = argparse.Namespace(**args_dict)

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_id_curv
settings = get_settings(args)

exp = Exp(args)  # set experiments
exp.save_ids(args,settings)
exp.save_curvs(args,settings)

