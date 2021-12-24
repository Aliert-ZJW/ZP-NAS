import os
import time
import argparse

# TODO please configure TORCH_HOME and data_paths before running

# TORCH_HOME = "TENAS/NAS-Bench-201/"
data_paths = {
    "cifar10": "data/cifar-10-batches-py/",
    "cifar100": "data/cifar-100-python/",
    "ImageNet16-120": "magenet16/ImageNet16/",
    "imagenet-1k": "data/imagenet-1k/",
    'TORCH_HOME': "code/TENAS/NAS-Bench-201/",
}


parser = argparse.ArgumentParser("TENAS_launch")
parser.add_argument('--space', default='darts', type=str, choices=['nas-bench-201', 'darts'], help='which nas search space to use')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
parser.add_argument('--seed', default=0, type=int, help='manual seed')
args = parser.parse_args()


##### Basic Settings
precision = 3

if args.space == "nas-bench-201":
    prune_number = 4
    batch_size = 128
    space = "nas-bench-201"  # different spaces of operator candidates, not structure of supernet
    super_type = "basic"  # type of supernet structure
    TORCH_HOME = data_paths['TORCH_HOME']
elif args.space == "darts":
    space = "darts"
    super_type = "nasnet-super"
    TORCH_HOME = ' '
    if args.dataset == "cifar10":
        prune_number = 6
        batch_size = 16
    elif args.dataset == "imagenet-1k":
        prune_number = 6
        batch_size = 16


timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))


core_cmd = "python ./prune_tenas.py \
--save_dir {save_dir} --max_nodes {max_nodes} \
--dataset {dataset} \
--data_path {data_path} \
--search_space_name {space} \
--super_type {super_type} \
--arch_nas_dataset {TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth \
--track_running_stats 1 \
--workers 0 --rand_seed {seed} \
--timestamp {timestamp} \
--precision {precision} \
--repeat 1 \
--batch_size {batch_size} \
--prune_number {prune_number} \
".format(
    save_dir="./output/prune-{space}/{dataset}".format(space=space, dataset=args.dataset),   # save path
    max_nodes=4,
    data_path=data_paths[args.dataset],
    dataset=args.dataset,
    TORCH_HOME=TORCH_HOME,
    space=space,    # search space
    super_type=super_type,   # type of supernet structure   basic or nasnet-super
    seed=args.seed,
    timestamp=timestamp,
    precision=precision,
    batch_size=batch_size,
    prune_number=prune_number
)

os.system(core_cmd)

