import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset

import utils.score_calculation as lib
from models.resnet import VirtualResNet50, VirtualResNet34
from models.wrn import WideResNet
from utils import testing as utils_testing
from utils.display_results import show_performance, print_measures

from models.resnet import ResNetModel

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--checkpoint', '-c', type=str, help='Checkpoint path to test.')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet-1k', 'imagenet-100'],
                    help='Choose between CIFAR-10, CIFAR-100, Imagenet-100, Imagenet-1k.')
parser.add_argument('--data-root', type=str, default='./data')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['wrn', 'rn34', 'rn50'], help='Choose architecture.')
parser.add_argument('--ood-method', choices=['vos', 'dream-ood'], default='vos', help='Choose OOD method.')
# Loading details
# WRN Architecture
parser.add_argument('--wrn-layers', default=40, type=int, help='total number of layers')
parser.add_argument('--wrn-widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--wrn-droprate', default=0.3, type=float, help='dropout probability')

parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--null-space-red-dim', type=int, default=-1)

args = parser.parse_args()
print(args)

cifar_test_transform = utils_testing.get_cifar_test_transforms()
in_test_transform = utils_testing.get_in_test_transforms()

if args.dataset == 'cifar10':
    test_data = dset.CIFAR10(f'{args.data_root}/cifarpy', train=False, transform=cifar_test_transform)
    num_classes = 10
elif args.dataset == 'cifar100':
    test_data = dset.CIFAR100(f'{args.data_root}/cifarpy', train=False, transform=cifar_test_transform)
    num_classes = 100
elif args.dataset == 'imagenet-1k':
    test_data = dset.ImageFolder(f'{args.data_root}/imagenet-1k/val', transform=in_test_transform)
    num_classes = 1000
elif args.dataset == 'imagenet-100':
    test_data = dset.ImageFolder(f'{args.data_root}/imagenet-100/val', transform=in_test_transform)
    num_classes = 100
else:
    assert False, 'Not an available dataset'

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
if args.ood_method == 'vos':
    if args.model == 'rn50':
        net = VirtualResNet50(num_classes, null_space_red_dim=args.null_space_red_dim)
    elif args.model == 'rn34':
        net = VirtualResNet34(num_classes, null_space_red_dim=args.null_space_red_dim)
    else:
        net = WideResNet(args.wrn_layers, num_classes, args.wrn_widen_factor, drop_rate=args.wrn_droprate,
                         null_space_red_dim=args.null_space_red_dim)
else:
    if args.model == 'rn34':
        net = ResNetModel(name='resnet34', num_classes=num_classes, null_space_red_dim=args.null_space_red_dim)
    else:
        raise ValueError(f'Unknown model {args.model}')

# Restore model
if os.path.isfile(args.checkpoint):
    net.load_state_dict(torch.load(str(args.checkpoint)))
    print('Model restored!')
else:
    raise ValueError(f'No checkpoint found at {args.checkpoint}')

net.eval()
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
elif args.ngpu == 1:
    net.cuda()

cudnn.benchmark = True

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

results_kwargs = {'model': net,
                  'ood_num_examples': ood_num_examples,
                  'num_to_avg': args.num_to_avg,
                  'test_bs': args.test_bs,
                  'score': args.score,
                  'temp': args.T,
                  'use_xent': args.use_xent,
                  'out_as_pos': args.out_as_pos,
                  'method_name': args.ood_method}
if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, args.test_bs, ood_num_examples,
                                                                 args.T, args.noise, in_dist=True)
elif args.score == 'M':
    from torch.autograd import Variable

    _, right_score, wrong_score = utils_testing.get_ood_scores(net, test_loader, ood_num_examples, args.test_bs,
                                                               score='M', use_xent=args.use_xent, in_dist=True)

    if 'cifar10_' in args.method_name:
        train_data = dset.CIFAR10(f'{args.data_root}/cifarpy', train=True, transform=cifar_test_transform)
    else:
        train_data = dset.CIFAR100(f'{args.data_root}/cifarpy', train=True, transform=in_test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                               num_workers=args.prefetch, pin_memory=True)
    num_batches = ood_num_examples // args.test_bs

    temp_x = torch.rand(2, 3, 32, 32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader)
    in_score = lib.get_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count - 1, args.noise,
                                         num_batches, in_dist=True)
    print(in_score[-3:], in_score[-103:-100])

    results_kwargs.update({'noise': args.noise,
                           'num_classes': num_classes,
                           'mahalanobis_sample_mean': sample_mean,
                           'mahalanobis_precision': precision,
                           'mahalanobis_count': count,
                           'num_batches': num_batches})
else:
    in_score, right_score, wrong_score = utils_testing.get_ood_scores(net, test_loader, ood_num_examples, args.test_bs,
                                                                      use_xent=args.use_xent, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.ood_method)

# /////////////// OOD Detection ///////////////
results_kwargs['in_score'] = in_score

res_list = []
# /////////////// Textures ///////////////
results_kwargs['ood_loader'] = utils_testing.get_textures_dataloader(args.data_root, args.test_bs)
print('\n\nTexture Detection')
res = utils_testing.get_and_print_results(**results_kwargs)
res_list.append(res)

# /////////////// SVHN /////////////// # cropped and no sampling of the test set
results_kwargs['ood_loader'] = utils_testing.get_svhn_dataloader(args.data_root, args.test_bs)
print('\n\nSVHN Detection')
res = utils_testing.get_and_print_results(**results_kwargs)
res_list.append(res)
# /////////////// Places365 ///////////////
results_kwargs['ood_loader'] = utils_testing.get_places365_dataloader(args.data_root, args.test_bs)
print('\n\nPlaces365 Detection')
res = utils_testing.get_and_print_results(**results_kwargs)
res_list.append(res)
# /////////////// LSUN-C ///////////////
results_kwargs['ood_loader'] = utils_testing.get_lsun_c_dataloader(args.data_root, args.test_bs)
print('\n\nLSUN_C Detection')
res = utils_testing.get_and_print_results(**results_kwargs)
res_list.append(res)
# /////////////// iSUN ///////////////
results_kwargs['ood_loader'] = utils_testing.get_isun_dataloader(args.data_root, args.test_bs)
print('\n\niSUN Detection')
res = utils_testing.get_and_print_results(**results_kwargs)
res_list.append(res)
# /////////////// Mean Results ///////////////

# res_list --> List of tuples to tuple of lists
auroc_list, aupr_list, fpr_list = zip(*res_list)

print('\n\nMean Test Results!!!!!')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name='vos/ffs')
