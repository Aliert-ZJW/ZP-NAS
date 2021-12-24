'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

This file is modified from:
https://github.com/VITA-Group/TENAS
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time
from datasets import get_datasets
from models import get_cell_based_tiny_net, DownSample_Jacobs
from config_utils import draw_scatter, prepare_seed
from pandas import DataFrame
from nas_201_api import NASBench201API as API
from tqdm import tqdm
import math
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from pdb import set_trace as bp
from datasets import CUTOUT, Dataset2Class, ImageNet16
from operator import mul
from functools import reduce


class RandChannel(object):
    # randomly pick channels from input
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __repr__(self):
        return ('{name}(num_channel={num_channel})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        channel = img.size(0)
        channel_choice = sorted(np.random.choice(list(range(channel)), size=self.num_channel, replace=False))
        return torch.index_select(img, 0, torch.Tensor(channel_choice).long())


def get_datasets_(name, root, input_size, cutout=-1):
    assert len(input_size) in [3, 4]
    if len(input_size) == 4:
        input_size = input_size[1:]
    assert input_size[1] == input_size[2]

    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std  = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std  = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('imagenet-1k'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('ImageNet16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name.startswith('ImageNet16'):
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name.startswith('imagenet-1k'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if name == 'imagenet-1k':
            xlists    = []
            xlists.append(transforms.Resize((32, 32), interpolation=2))
            xlists.append(transforms.RandomCrop(input_size[1], padding=0))
        elif name == 'imagenet-1k-s':
            xlists = [transforms.RandomResizedCrop(32, scale=(0.2, 1.0))]
            xlists = []
        else: raise ValueError('invalid name : {:}'.format(name))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        xlists.append(RandChannel(input_size[0]))
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose([transforms.Resize(40), transforms.CenterCrop(32), transforms.ToTensor(), normalize])
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k'):
        train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    elif name == 'ImageNet16':
        train_data = ImageNet16(root, True , train_transform)
        test_data  = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True , train_transform, 120)
        test_data  = ImageNet16(root, False, test_transform , 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        train_data = ImageNet16(root, True , train_transform, 150)
        test_data  = ImageNet16(root, False, test_transform , 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        train_data = ImageNet16(root, True , train_transform, 200)
        test_data  = ImageNet16(root, False, test_transform , 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else: raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, class_num


class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron).cuda()
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations.half(), (1-self.activations).T.half()) # each element in res: A * (1 - B)
        res += res.T # make symmetric, each element in res: A * (1 - B) + (1 - A) * B, a non-zero element indicate a pair of two different linear regions
        res = 1 - torch.sign(res) # a non-zero element now indicate two linear regions are identical
        res = res.sum(1) # for each sample's linear region: how many identical regions from other samples
        res = 1. / res.float() # contribution of each redudant (repeated) linear region
        self.n_LR = res.sum().item() # sum of unique regions (by aggregating contribution of all regions)
        del self.activations, res
        self.activations = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class Linear_Region_Collector:
    def __init__(self, models=[], input_size=(64, 3, 32, 32), sample_batch=100, dataset='cifar100', data_path=None, seed=0, input=None, target=None, data_iter=None):
        self.models = []
        self.input_size = input_size  # BCHW
        self.sample_batch = sample_batch
        self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.dataset = dataset
        self.data_path = data_path
        self.seed = seed
        self.input = input
        self.target = target
        self.data_iter = data_iter
        self.reinit(models, input_size, sample_batch, seed)

    def reinit(self, models=None, input_size=None, sample_batch=None, seed=None):
        if models is not None:
            assert isinstance(models, list)
            del self.models
            self.models = models
            for model in self.models:
                self.register_hook(model)
            self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(models))]
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(self.models))]
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            try:
                inputs, targets = self.data_iter.next()
            except Exception:
                print('error')
                # del self.loader
                # self.loader = iter(self.train_loader_)
                # inputs, targets = self.loader.next()
            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            model.forward(input_data.cuda())
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)



def compute_RN_score(model: nn.Module,  batch_size=None, image_size=None, dataset=None, path=None, seed=0, input=None, target=None, data_iter=None):
    lrc_model = Linear_Region_Collector(models=[model], input_size=(batch_size, 3, image_size, image_size),
                                        sample_batch=3, dataset=dataset, data_path=path, seed=seed, input=input, target=target, data_iter=data_iter)
    num_linear_regions = float(lrc_model.forward_batch_sample()[0])
    del lrc_model
    torch.cuda.empty_cache()
    return num_linear_regions


def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network

def round_to(number, precision, eps=1e-8):
    # round to significant figure
    dtype = type(number)
    if number == 0:
        return number
    sign = number / abs(number)
    number = abs(number) + eps
    power = math.floor(math.log(number, 10)) + 1
    if dtype == int:
        return int(sign * round(number*10**(-power), precision) * 10**(power))
    else:
        return sign * round(number*10**(-power), precision) * 10**(power)

def get_ntk_n(networks, recalbn=0, train_mode=False, num_batch=None, inputs=None, target=None):
    device = torch.cuda.current_device()
    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    for net_idx, network in enumerate(networks):
        network.zero_grad()
        inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
        logit = network(inputs_)
        if isinstance(logit, tuple):
            logit = logit[1]  # 201 networks: return features and logits
        for _idx in range(len(inputs_)):
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, W in network.named_parameters():
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            grads[net_idx].append(torch.cat(grad, -1))
            network.zero_grad()
            torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds



def compute_NTK_score(model, input, target):
    ntk_score = get_ntk_n([model], recalbn=0, train_mode=True, num_batch=1, inputs=input, target=target)[0]
    return -1 * ntk_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NAS Without Training')
    parser.add_argument('--api_loc', default='/data/zjw/code/TE-NAS/NAS-Bench-201/NAS-Bench-201-v1_0-e61699.pth', type=str, help='path to API')
    parser.add_argument('--GPU', default=1, type=str)
    parser.add_argument('--dataset', default='cifar100', help='ImageNet16-120, cifar100, cifar10')
    # parser.add_argument('--data_loc', default='/data/zjw/data/ImageNet16/')
    parser.add_argument('--data_loc', default='/data/zjw/data/cifar-100-python/')
    # parser.add_argument('--data_loc', default='/data/zjw/data/cifar-10-batches-py/')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--seed', default=2, type=int, help='manual seed')
    parser.add_argument('--samples', default=1000, type=int, help='sample networks')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api = API(args.api_loc)

    train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                               pin_memory=True)

    if args.dataset == 'cifar10':
        acc_type = 'ori-test'
        val_acc_type = 'x-valid'

    else:
        acc_type = 'x-test'
        val_acc_type = 'x-valid'
    start_timer = time.time()

    all_scores = []
    accs = []
    times = []
    score = []

    prepare_seed(args.seed)
    accuracy = []
    indices = np.random.choice(15625, args.samples, replace=False)


    index_tqdm = tqdm(indices, desc='')
    for idx in index_tqdm:
        data_iter = iter(train_loader)
        x_, target = next(data_iter)
        x, target = x_.to(device), target.to(device)
        jacobs = []
        targets = []
        start = time.time()

        config = api.get_net_config(idx, 'ImageNet16-120')
        print(config, idx)

        info_ = api.query_by_index(idx)
        acc_test = info_.get_metrics(args.dataset, acc_type)['accuracy']
        print('acc', acc_test)

        network = get_cell_based_tiny_net(config)
        network = network.to(device)
        network.train()
        ntk = compute_NTK_score(model=network, input=x, target=target)
        RN = compute_RN_score(model=network, batch_size=args.batch_size, image_size=32,
                              dataset=args.dataset, path=args.data_loc, seed=args.seed, input=x, target=target, data_iter=data_iter)
        score.append([ntk, RN, acc_test])
        print([ntk, RN, acc_test])
        times.append(time.time() - start)
    all_ntk = sorted(score, key=lambda tup: round_to(tup[0], 3), reverse=True)
    rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]
    for idx, data in enumerate(all_ntk):
        rankings[data[2]] = [idx, data[1]]
    all_rn = sorted(score, key=lambda tup: round_to(tup[1], 3), reverse=True)
    keys = rankings.keys()
    for idx, data in enumerate(all_rn):
        for idx_ in keys:
            if data[2] == idx_:
                rankings[data[2]] = [rankings[idx_][0], idx]
    rankings_list = [[k, sum(v)] for k, v in rankings.items()]
    for tup in range(len(rankings_list)):
        print(rankings_list[tup])
        accs.append(rankings_list[tup][0])
        all_scores.append(rankings_list[tup][1])
    draw_scatter(accs, all_scores)
    best_net = indices[np.nanargmin(all_scores)]
    data = DataFrame({'accuracy': accs, 'scores': all_scores})
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
