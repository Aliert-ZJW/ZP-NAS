'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np


import torch.nn.functional as F
# def cross_entropy(logit, target):
#     # target must be one-hot format!!
#     prob_logit = F.log_softmax(logit, dim=1)
#     loss = -(target * prob_logit).sum(dim=1).mean()
#     return loss

def compute_nas_score( model, input, target):

    _, output, _ = model(input)
    loss = nn.CrossEntropyLoss().cuda()
    # loss = nn.cross_entropy(output, target)
    loss(output, target).backward()
    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))

    return grad_norm


from nas_201_api import NASBench201API as API
import argparse
import os
import torch
import numpy as np
import time
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import kendalltau
from datasets import get_datasets
from models import get_cell_based_tiny_net, DownSample_Jacobs
from config_utils import draw_scatter, prepare_seed, get_model_infos
from pandas import DataFrame
parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--api_loc', default='/data/zjw/code/TE-NAS/NAS-Bench-201/NAS-Bench-201-v1_0-e61699.pth', type=str, help='path to API')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--dataset', default='ImageNet16-120', help='ImageNet16-120, cifar100, cifar10')
parser.add_argument('--data_loc', default='/data/zjw/data/ImageNet16/')
# parser.add_argument('--data_loc', default='/data/zjw/data/cifar-10-batches-py/')
parser.add_argument('--batch_size', default=100)
parser.add_argument('--seed', default=0, type=int, help='manual seed')
parser.add_argument('--samples', default=1000, type=int, help='sample networks')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api = API(args.api_loc)

train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'

else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'



def test_best(index):
    '''
      none 类型的数据多表现为求特征值的时候，基本全是i零，所以开题进行排除操作
    '''
    scores = []
    accs = []
    times = []
    flops = []
    params = []

    prepare_seed(args.seed)
    accuracy = []
    indices = np.random.choice(15625, args.samples, replace=False)

    index_tqdm = tqdm(indices, desc='')
    for idx in index_tqdm:
        start = time.time()

        data_iter = iter(train_loader)
        x_, target = next(data_iter)
        x, target = x_.to(device), target.to(device)

        config = api.get_net_config(idx, 'ImageNet16-120')
        print(config, idx)

        info_ = api.query_by_index(idx)
        acc_test = info_.get_metrics(args.dataset, acc_type)['accuracy']
        print('acc', acc_test)

        network = get_cell_based_tiny_net(config)
        network = network.to(device)
        flop, param = get_model_infos(network, xshape)
        flops.append(flop)
        params.append(param)
        network.train()
        s = compute_nas_score(network, x, target)
        accs.append(acc_test)
        scores.append(s)
        # print(flop)
        # print(param)
        times.append(time.time() - start)
    draw_scatter(accs, scores)
    best_net = indices[np.nanargmax(accs)]
    print(accuracy)
    data = DataFrame({'accuracy': accs, 'scores': scores})
    lendall = data.corr(method='kendall')
    print(lendall)

    config = api.get_net_config(best_net, 'ImageNet16-120')
    print(api.query_by_arch(config['arch_str']))

    info = api.query_by_index(best_net)
    acc_val = info.get_metrics(args.dataset, 'x-valid')['accuracy']
    acc_test = info.get_metrics(args.dataset, 'x-test')['accuracy']
    print('acc_val', acc_val)
    print('acc_test', acc_test)
    print(sorted(accs, reverse=True)[:10])
    print(f'mean time: {np.mean(times)}')


if __name__ == '__main__':
    test_best(False)




