# import matplotlib
# from jedi.api.refactoring import inline

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
import matplotlib.pyplot as plt
import seaborn as sns
parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--api_loc', default='/data/zjw/code/EPE-NAS/nas_201_api/NAS-Bench-201-v1_0-e61699.pth', type=str, help='path to API')
parser.add_argument('--GPU', default='1', type=str)
parser.add_argument('--dataset', default='ImageNet16-120', help='ImageNet16-120, cifar100, cifar10')
parser.add_argument('--data_loc', default='/data/zjw/data/data/imagenet16/ImageNet16/')
parser.add_argument('--batch_size', default=128)
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

def get_batch_jacobian(net, x, target):
    net.zero_grad()
    x.requires_grad_(True)
    _, y, _ = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()

def count(labels, item):
    num = 0
    index = []
    for i, lable in enumerate(labels):
        if item == lable:
            num += 1
            index.append(i)
    return num, index

def eval_score_perclass(jacobs_batch, jacob, idx, labels=None):
    k = 1e-5
    corrs = np.corrcoef(jacob)
    s = 0
    for i in range(len(corrs)):
        for j in range(len(corrs)):
            if corrs[i][j] < 0.1:
                s += np.log(abs(corrs[i][j])+k)
    score = np.absolute(s)
    return score

def jacob_corr_sum(jacob):
    k = 1e-5
    corr = np.corrcoef(jacob)
    score = np.sum(np.log(abs(corr)+k))

def not_same_kind(output, target):

    bool = [True] * len(target)
    output_mv = []
    for i, lable in enumerate(target):
        num, index = count(target, lable)
        if num > 1:
            if bool[i] == True:
                output_mv.append((output[i]))
                for inx in index:
                    bool[inx] = False
        else:
            output_mv.append(output[i])
    return output_mv

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
    indices = np.random.choice(15625, args.samples, replace=False)

    index_tqdm = tqdm(indices, desc='')
    for idx in index_tqdm:
        jacobs = []
        targets = []
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
        network.train()
        jacobs_batch, target = get_batch_jacobian(network, x, target)
        jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
        targets.append(target.cpu().numpy())
        jacobs = np.concatenate(jacobs, axis=0)
        s = eval_score_perclass(jacobs_batch, jacobs, idx, targets[0])
        accs.append(acc_test)
        # sc += s
        scores.append(s)
        flops.append(flop)
        params.append(param)
        # print(flop)
        # print(param)
        times.append(time.time() - start)
    # print(sc/20)
    draw_scatter(accs, scores)
    best_net = indices[np.nanargmax(scores)]
    data = DataFrame({'accuracy': accs, 'params': params})
    lendall = data.corr(method='kendall')
    print(lendall)

    config = api.get_net_config(best_net, 'ImageNet16-120')
    print(api.query_by_arch(config['arch_str']))


if __name__ == '__main__':
    test_best(False)


