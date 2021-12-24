import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zero_cost_score import utils
from torch.autograd import Variable
from muli_net import Net, Net1, Net2, Net3, Net4, Net5, Net6, Net7, Net8, Net9, Net10
from resnet_test import ResNet18_1, ResNet18_2, ResNet18_3, ResNet18_4, ResNet18_5, ResNet18_6, ResNet18_7, ResNet18_8, ResNet18_9, ResNet18_10, ResNet18_11 ,ResNet18_12, ResNet18_13



parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data/zjw/data/cifar-10-batches-py/', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--save', type=str, default='/data/zjw/code/EPE-NAS/weight_pth/', help='experiment name')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
# parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'])
parser.add_argument('--search_space_name', default='darts', type=str)
parser.add_argument('--init_channels', default=36, type=int)
parser.add_argument('--resume', default=True, help='use checkpoint')
parser.add_argument('--super_type', type=str, default='nasnet-super',  help='type of supernet: basic or nasnet-super')  ####
parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')  ####   1
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
args = parser.parse_args()

args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

if args.set=='cifar100':
    CIFAR_CLASSES = 100


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  # torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  print(torch.cuda.device_count(), 'GPU')

  model = ResNet18_13()
  model = nn.DataParallel(model, device_ids=[0])
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  #valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0.0

  start_epoch = 0

  # if args.resume:
  #   path_checkpoint = os.path.join('/data/zjw/code/EPE-NAS/weight_pth/-20211111-101218/', 'checkpoint.pth.tar')  # 断点路径
  #   checkpoint = torch.load(path_checkpoint)
  #
  #   model.load_state_dict(checkpoint['state_dict'])
  #
  #   optimizer.load_state_dict(checkpoint['optimizer'])
  #   scheduler.load_state_dict(checkpoint['scheduler'])
  #   start_epoch = checkpoint['epoch']  # 设置开始的epoch

  for epoch in range(start_epoch, args.epochs):
    logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)
    scheduler.step()

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    if valid_acc > best_acc:
        best_acc = valid_acc
    logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)

    # utils.save(model, os.path.join(args.save, 'weights.pt'))
    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler.state_dict(),
    }, True, args.save)

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
        input = Variable(input).cuda()
        target = Variable(target).cuda()

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
