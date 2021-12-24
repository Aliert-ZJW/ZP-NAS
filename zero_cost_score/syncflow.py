'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from
https://github.com/SamsungLabs/zero-cost-nas
'''


# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np



import torch

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array



def compute_synflow_per_weight(net, inputs, mode, batch_size):
    device = inputs.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([batch_size] + input_dim).double().to(device)
    _, output, _ = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs

def do_compute_nas_score(model, input, batch_size):

    grads_abs_list = compute_synflow_per_weight(net=model, inputs=input, mode='', batch_size=batch_size)
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('!!!')


    return -1 * score


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
        flops.append(flop)
        params.append(param)
        network.train()
        s = do_compute_nas_score(network, x, args.batch_size)
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




