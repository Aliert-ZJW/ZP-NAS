import os, sys, time, argparse
import math
import random
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, prepare_logger
from procedures import Linear_Region_Collector, get_ntk_n
from utils import get_model_infos
from log_utils import time_string
from models import get_cell_based_tiny_net, get_search_spaces  # , nas_super_nets
from nas_201_api import NASBench201API as API
from procedures import flow
from procedures import generator as generator
from pdb import set_trace as bp


INF = 1000  # used to mark prunned operators


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


def device_(gpu):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config(xargs, index_space, class_num, edge):
    if xargs.search_space_name == 'nas-bench-201':
            model_config = edict({'name': 'DARTS-V1',
                                  'C': 16, 'N': 1, 'depth': -1, 'use_stem': True,
                                  'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                  'space': index_space, 'edge': edge,
                                  'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                 })
            return model_config
    elif xargs.search_space_name == 'darts':
        if args.dataset == "imagenet-1k":
            model_config = edict({'name': 'DARTS-V1',
                                  'C': 48, 'N': 5, 'depth': 2, 'use_stem': True, 'stem_multiplier': 3,
                                  'num_classes': class_num,
                                  'space': index_space,
                                  'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                  'super_type': xargs.super_type,
                                  'steps': 4,
                                  'multiplier': 4,
                                 })
        else:
            model_config = edict({'name': 'DARTS-V1',
                                  'C': 36, 'N': 5, 'depth': 2, 'use_stem': True, 'stem_multiplier': 3,
                                  'num_classes': class_num,
                                  'space': index_space,
                                  'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                  'super_type': xargs.super_type,
                                  'steps': 4,
                                  'multiplier': 4,
                                  })
        return model_config


def prune_func_rank(xargs, arch_parameters, index_space, class_num, edge, search_space, network_origin, input_ori, target, train_loader, precision=10, prune_number=1):
    model_config = get_config(xargs, index_space, class_num, edge)
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    if args.search_space_name == "darts":
        for alpha in arch_parameters:
            alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    prune_number = min(prune_number, alpha_active[0][0].sum()-1)  # adjust prune_number based on current remaining ops on each edge
    ntk_all = []  # (ntk, (edge_idx, op_idx))
    score_dart = []

    device = device_(xargs.gpu)
    loss = nn.CrossEntropyLoss().cuda()

    loss_list_ = []
    for i in range(xargs.repeat):
        target_ = target[i]
        input_ori_ = input_ori[i]
        model = [network_origin]
        loss_list, res_var = flow.score(model, loss, input_ori_, target_, train_loader, device)
        loss_list_.append(loss_list[0])

    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_edge in range(len(arch_parameters[idx_ct])):
            if alpha_active[idx_ct][idx_edge].sum() == 1:
                # only one op remaining
                continue
            for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                    _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                    _arch_param[idx_ct][idx_edge, idx_op] = -INF
                    # for idx, _ in enumerate(_arch_param[idx_ct][idx_edge, :]):
                    #     if idx != idx_op:
                    #         _arch_param[idx_ct][idx_edge, idx] = -INF
                    network = get_cell_based_tiny_net(model_config).cuda().train()
                    ntk_delta = []
                    for [name_ori, param_ori], [name, param] in zip(network_origin.named_parameters(), network.named_parameters()):
                        param.data.copy_(param_ori.data)
                    network.set_alphas(_arch_param)
                    repeat = xargs.repeat
                    # input_ori, target = input_ori.to(device), target.to(device)
                    # data_iter = iter(train_loader)
                    # input_ori, target = next(data_iter)
                    for j in range(xargs.repeat):
                        target_ = target[j]
                        input_ori_ = input_ori[j]
                        model = [network]
                        loss_list, res_var = flow.score(model, loss, input_ori_, target_, train_loader, device)
                        network_origin_loss = loss_list_[j]
                        network_loss = loss_list[0]
                        ntk_delta.append(network_loss)
                        print(network_origin_loss, network_loss, network_origin_loss - network_loss)
                    ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])
                    pbar.update(1)

    ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
    edge2choice = {}  # (cell_idx, edge_idx): list of (cell_idx, edge_idx, op_idx) of length prune_number
    for s, (cell_idx, edge_idx, op_idx) in ntk_all:
        if (cell_idx, edge_idx) not in edge2choice:
            edge2choice[(cell_idx, edge_idx)] = [(cell_idx, edge_idx, op_idx)]
        elif len(edge2choice[(cell_idx, edge_idx)]) < prune_number:
            edge2choice[(cell_idx, edge_idx)].append((cell_idx, edge_idx, op_idx))
        elif len(edge2choice[(cell_idx, edge_idx)]) >= prune_number:
            score_dart.append([np.mean(s), (cell_idx, edge_idx, op_idx)])
    choices_edges = list(edge2choice.values())
    for choices in choices_edges:
        for (cell_idx, edge_idx, op_idx) in choices:
            arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF
    print(arch_parameters)

    edge_groups = [(0, 2), (2, 5), (5, 9), (9, 14)]
    if args.search_space_name == "darts":
        for idx_group in range(len(edge_groups)):
            edge_group = edge_groups[idx_group]
            if edge_group[1] - edge_group[0] > 2:
                score = []
                score_ = []
                for i in range(edge_group[0], edge_group[1]):
                    for s, (cell_idx, edge_idx, op_idx) in score_dart:
                        if i == edge_idx:
                            if cell_idx == 0:
                                score.append([s, (cell_idx, edge_idx, op_idx)])
                            else:
                                score_.append([s, (cell_idx, edge_idx, op_idx)])
                score = sorted(score, key=lambda tup: round_to(tup[0], precision), reverse=True)
                for _, (cell_idx, edge_idx, op_idx) in score[:-2]:
                    choices_edges.append([(cell_idx, edge_idx, op_idx)])
                score_ = sorted(score_, key=lambda tup: round_to(tup[0], precision), reverse=True)
                for _, (cell_idx, edge_idx, op_idx) in score_[:-2]:
                    choices_edges.append([(cell_idx, edge_idx, op_idx)])

    for choices in choices_edges:
        for (cell_idx, edge_idx, op_idx) in choices:
            arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF

    return arch_parameters, choices_edges


def is_single_path(network):
    arch_parameters = network.get_alphas()
    edge_active = torch.cat([(nn.functional.softmax(alpha, 1) > 0.01).float().sum(1) for alpha in arch_parameters], dim=0)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    return True


def main(xargs):
    PID = os.getpid()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)

    ##### config & logging #####
    edge = {'1<-0': 0, '2<-0': 1, '2<-1': 2, '3<-0': 3, '3<-1': 4, '3<-2': 5}
    config = edict()
    config.class_num = class_num
    config.xshape = xshape
    config.batch_size = xargs.batch_size
    xargs.save_dir = xargs.save_dir + \
        "/{:}/seed{:}".format(xargs.timestamp, xargs.rand_seed)
    config.save_dir = xargs.save_dir
    logger = prepare_logger(xargs)
    ###############

    if xargs.dataset != 'imagenet-1k':
        search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                                           'configs/', config.batch_size, xargs.workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=xargs.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
    logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader),
                                                                                    config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

    search_space = get_search_spaces('cell', xargs.search_space_name)
    # index_space = [[space for space in search_space] for i in range(6)]
    index_space = search_space
    model_config = get_config(xargs, index_space, class_num, edge)
    network = get_cell_based_tiny_net(model_config)
    logger.log('model-config : {:}'.format(model_config))
    arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, :] = 0

    # ### all params trainable (except train_bn) #########################
    flop, param = get_model_infos(network, xshape)
    logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
    if xargs.arch_nas_dataset is None or xargs.search_space_name == 'darts':
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

    network = network.cuda()

    genotypes = {}; genotypes['arch'] = {-1: network.genotype()}

    arch_parameters_history_npy = []
    start_time = time.time()
    epoch = -1
    model_config = get_config(xargs, index_space, class_num, edge)
    network = get_cell_based_tiny_net(model_config)
    network.set_alphas(arch_parameters)
    # network = init_model(network, xargs.init + "_fanin" if xargs.init.startswith('kaiming') else xargs.init)  # for forward
    network = network.cuda()
    device = device_(xargs.gpu)
    input_ori = []
    target = []
    for i in range(xargs.repeat):
        data_iter = iter(train_loader)
        x_, target_ = next(data_iter)
        # input_ori, target = x_.to(device), target.to(device)
        input_ori.append(x_)
        target.append(target_)
    # data_iter = iter(train_loader)
    # x_, target = next(data_iter)
    # input_ori, target = x_.to(device), target.to(device)

    # while not is_single_path(network):
    epoch += 1
    torch.cuda.empty_cache()
    print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(xargs.save_dir.split("/")[-6:])))

    arch_parameters, op_pruned = prune_func_rank(xargs, arch_parameters, index_space, class_num, edge, search_space, network,
                                                 input_ori, target, train_loader, precision=xargs.precision, prune_number=xargs.prune_number)
    # rebuild supernet
    network = get_cell_based_tiny_net(model_config)
    network = network.cuda()
    network.set_alphas(arch_parameters)

    genotypes['arch'][epoch] = network.genotype()

    logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))
    np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)

    logger.log('<<<--->>> End: {:}'.format(network.genotype()))
    logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    end_time = time.time()
    logger.log('\n' + '-'*100)
    logger.log("Time spent: %d s"%(end_time - start_time))
    # check the performance from the architecture dataset
    if api is not None:
        logger.log('{:}'.format(api.query_by_arch(genotypes['arch'][epoch])))

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("TENAS")
    parser.add_argument('--data_path', type=str, help='Path to dataset') ####
    parser.add_argument('--gpu', default='0', type=str, help='use gpu')
    parser.add_argument('--dataset', default='ImageNet16-120', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')   #####
    parser.add_argument('--search_space_name', type=str, default='nas-bench-201',  help='space of operator candidates: nas-bench-201 or darts.') #####
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.')   ### 4
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')  ####   1
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')   ####
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for ntk')  ####
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')  #####
    parser.add_argument('--arch_nas_dataset', type=str, help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')   ####
    parser.add_argument('--rand_seed', type=int, help='manual seed')  ###
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')   ####  3
    parser.add_argument('--prune_number', type=int, default=1, help='number of operator to prune on each edge per round')   ####
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK and Regions')  ####   3
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')   ####
    parser.add_argument('--init', default='kaiming_uniform', help='use gaussian init')  ###  kaiming_normal
    parser.add_argument('--super_type', type=str, default='basic',  help='type of supernet: basic or nasnet-super')  ####
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
