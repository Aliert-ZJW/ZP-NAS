import argparse
from pandas import DataFrame
import random
import numpy as np
import torch
import os, sys
from tqdm import trange, tqdm
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_utils import draw_scatter
from models import get_cell_based_tiny_net
from nas_201_api import NASBench201API as API
from datasets import get_datasets
from muli_net import Net7
from config_utils import get_model_infos

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='/data/zjw/data/cifar-10-batches-py/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='/data/zjw/code/TE-NAS/NAS-Bench-201/NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--kernel', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='1', type=str)
parser.add_argument('--seed', default=5, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--activations', action='store_true')
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=1000, type=int)
parser.add_argument('--n_runs', default=1, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int,
                    help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
api = API(args.api_loc)

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    ints, y, _ = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), ints.detach()

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()


_scores = {
        'hook_logdet': hooklogdet,
        'random': random_score
        }

def get_score_func(score_name):
    return _scores[score_name]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

times = []
chosen = []
acc = []
val_acc = []
topscores = []
order_fn = np.nanargmax

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

runs = trange(args.n_runs, desc='acc: ')
start_time = time.time()
for N in runs:
    start = time.time()
    indices = np.random.randint(0, 15625, args.n_samples)
    scores = []
    acc_test = []
    flops = []
    params = []

    npstate = np.random.get_state()
    ranstate = random.getstate()
    torchstate = torch.random.get_rng_state()
    index_tqdm = tqdm(indices, desc='')
    for idx in indices:

        # try:
        config = api.get_net_config(idx, 'ImageNet16-120')
        print(config, idx)
        network = get_cell_based_tiny_net(config)
        network.to(device)
        if 'hook_' in args.score:
            network.K = np.zeros((args.batch_size, args.batch_size))
            def counting_forward_hook(module, inp, out):
                try:
                    if not module.visited_backwards:
                        return
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    inp = inp.view(inp.size(0), -1)
                    x = (inp > 0).float()
                    K = x @ x.t()
                    K2 = (1. - x) @ (1. - x.t())
                    network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
                except:
                    pass

            def counting_backward_hook(module, inp, out):
                module.visited_backwards = True
            for name, module in network.named_modules():
                if 'ReLU' in str(type(module)):
                    # hooks[name] = module.register_forward_hook(counting_hook)
                    module.register_forward_hook(counting_forward_hook)
                    module.register_backward_hook(counting_backward_hook)
        random.setstate(ranstate)
        np.random.set_state(npstate)
        torch.set_rng_state(torchstate)

        data_iterator = iter(train_loader)
        x, target = next(data_iterator)
        x2 = torch.clone(x)
        x2 = x2.to(device)
        x, target = x.to(device), target.to(device)
        jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)

        if args.kernel:
            s = get_score_func(args.score)(out, labels)
        elif 'hook_' in args.score:
            network(x2.to(device))
            s = get_score_func(args.score)(network.K, target)
        elif args.repeat < args.batch_size:
            s = get_score_func(args.score)(jacobs, labels, args.repeat)
        else:
            s = get_score_func(args.score)(jacobs, labels)

        flop, param = get_model_infos(network, xshape)
        flops.append(flop)
        params.append(param)

        scores.append(s)
        print(s)
        info_ = api.query_by_index(idx)
        acc_test_ = info_.get_metrics(args.dataset, acc_type)['accuracy']
        print('acc', acc_test_)
        acc_test.append(acc_test_)

    # draw_scatter(acc_test, scores)
    end_time = time.time()
    data = DataFrame({'accuracy': acc_test, 'scores': scores})
    lendall = data.corr(method='kendall')
    print(lendall)
    best_net = indices[np.nanargmax(scores)]
    config = api.get_net_config(best_net, 'ImageNet16-120')
    print(api.query_by_arch(config['arch_str']))
    print('#time', end_time-start_time)

#     best_arch = indices[order_fn(scores)]
#     uid_ = searchspace[best_arch]
#     topscores.append(scores[order_fn(scores)])
#     chosen.append(best_arch)
#     # acc.append(searchspace.get_accuracy(uid, acc_type, args.trainval))
#     acc.append(searchspace.get_final_accuracy(uid_, acc_type, False))
#
#     if not args.dataset == 'cifar10' or args.trainval:
#         val_acc.append(searchspace.get_final_accuracy(uid_, val_acc_type, args.trainval))
#     #    val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])
#
#     times.append(time.time() - start)
#     runs.set_description(f"acc: {mean(acc):.2f}% time:{mean(times):.2f}")
#
# print(f"Final mean test accuracy: {np.mean(acc)}")



