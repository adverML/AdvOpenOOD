import collections
import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, RegNet_Y_16GF_Weights
from torchvision import transforms as trn
from torch.hub import load_state_dict_from_url

from openood.evaluation_api import AttackDataset

from openood.networks import ResNet50, Swin_T, ViT_B_16, RegNet_Y_16GF
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.cider_net import CIDERNet

from openood.attacks.misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    create_dir,
    str2bool,
    create_log_file,
    save_log
)

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

parser = argparse.ArgumentParser()
parser.add_argument('--arch',
                    default='resnet50',
                    choices=['resnet50', 'swin-t', 'vit-b-16', 'regnet'])
parser.add_argument('--tvs-version', default=1, choices=[1, 2])
parser.add_argument('--ckpt-path', default=None)
parser.add_argument('--tvs-pretrained', action='store_true')
parser.add_argument('--batch-size', default=50, type=int)

parser.add_argument("--att",  default="pgd", choices=[None, 'fgsm', 'bim', 'pgd', 'df', 'cw', 'mpgd'], help="")
parser.add_argument("--bpda",  default=False, type=str2bool, help="")
parser.add_argument("--eps",  default="4/255", help="")
parser.add_argument("--norm",  default="Linf", choices=['Linf', 'L2', 'L1'], help="")
parser.add_argument('--masked-patch-size', default=60, type=int)

parser.add_argument("--debug",  default=False, type=str2bool, help="")
args = parser.parse_args()

if args.eps and "." in args.eps:
    eps = float(args.eps)
else:
    eps = args.eps.replace("/", "") if args.eps and "/" in args.eps else ""; print("eps", eps)
args.eps = convert_to_float(args.eps)

if not args.tvs_pretrained:
    assert args.ckpt_path is not None
    root = '/'.join(args.ckpt_path.split('/')[:-1])
else:
    root = os.path.join(
        ROOT_DIR, 'results',
        f'imagenet_{args.arch}_tvsv{args.tvs_version}_base_default')
    if not os.path.exists(root):
        os.makedirs(root)

# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
# postprocessor_name = args.postprocessor
# load pre-setup postprocessor if exists
# if os.path.isfile(
#         os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl')):
#     with open(
#             os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl'),
#             'rb') as f:
#         postprocessor = pickle.load(f)
# else:
#     postprocessor = None

# assuming the model is either
# 1) torchvision pre-trained; or
# 2) a specified checkpoint
if args.tvs_pretrained:
    if args.arch == 'resnet50':
        net = ResNet50()
        weights = eval(f'ResNet50_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'swin-t':
        net = Swin_T()
        weights = eval(f'Swin_T_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'vit-b-16':
        net = ViT_B_16()
        weights = eval(f'ViT_B_16_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'regnet':
        net = RegNet_Y_16GF()
        weights = eval(
            f'RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    else:
        raise NotImplementedError
else:
    raise ValueError('Only pretrained models!')

net.cuda()
net.eval()

attackdataset = AttackDataset(
    net,
    id_name='imagenet',  # the target ID dataset
    data_root=os.path.join(ROOT_DIR, 'data'),
    config_root=os.path.join(ROOT_DIR, 'configs'),
    preprocessor=preprocessor,  # default preprocessing
    batch_size=args.
    batch_size,  # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=8
    )

attackdataset.run_attack(args)