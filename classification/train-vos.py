# -*- coding: utf-8 -*-
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
from FrEIA.framework import InputNode, Node, OutputNode, GraphINN
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

from utils.training import VirtualDataParallel
from utils import training as utils_training
from models.resnet import VirtualResNet50, VirtualResNet34
from models.wrn import WideResNet

parser = argparse.ArgumentParser(description='Trains a Classifier with VOS/FFS for OOD Detection',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet-1k', 'imagenet-100'],
                    help='Choose between CIFAR-10, CIFAR-100, Imagenet-100, Imagenet-1k.')
parser.add_argument('--data-root', type=str, default='./data')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['wrn', 'rn34', 'rn50'], help='Choose architecture.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# WRN Architecture
parser.add_argument('--wrn-layers', default=40, type=int, help='total number of layers')
parser.add_argument('--wrn-widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--wrn-droprate', default=0.3, type=float, help='dropout probability')

# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

# VOS/FFS params
parser.add_argument('--vos-start-epoch', type=int, default=40)
parser.add_argument('--vos-sample-number', type=int, default=1000)
parser.add_argument('--vos-select', type=int, default=1)
parser.add_argument('--vos-sample-from', type=int, default=10000)
parser.add_argument('--vos-loss-weight', type=float, default=0.1)
parser.add_argument('--use_ffs', action='store_true')

# Fever-OOD params
parser.add_argument('--smin_loss_weight', type=float, default=0.0)
parser.add_argument('--use_conditioning', action='store_true')
parser.add_argument('--null-space-red-dim', type=int, default=-1)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# FFS Functions
def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 2048), nn.ReLU(), nn.Linear(2048, c_out))


def build_ffs_flow(n_fts):
    in1 = InputNode(n_fts, name='input1')
    layer1 = Node(in1, GLOWCouplingBlock, {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                  name=F'coupling_{0}')
    layer2 = Node(layer1, PermuteRandom, {'seed': 0}, name=F'permute_{0}')
    layer3 = Node(layer2, GLOWCouplingBlock, {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                  name=F'coupling_{1}')
    layer4 = Node(layer3, PermuteRandom, {'seed': 1}, name=F'permute_{1}')
    out1 = OutputNode(layer4, name='output1')
    flow = GraphINN([in1, layer1, layer2, layer3, layer4, out1])
    return flow

def nll(z, sldj):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
      See Also:
          Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
    prior_ll = prior_ll.flatten(1).sum(-1) - np.log(256) * np.prod(z.size()[1:])
    ll = prior_ll + sldj
    return -ll


# Random seeds
g = torch.Generator()
g.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

cifar_train_transform, cifar_test_transform = utils_training.get_cifar_transforms()
in_train_transform, in_test_transform = utils_training.get_in_transforms()

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(f'{args.data_root}/cifarpy', train=True, transform=cifar_train_transform, download=True)
    test_data = dset.CIFAR10(f'{args.data_root}/cifarpy', train=False, transform=cifar_test_transform, download=True)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(f'{args.data_root}/cifarpy', train=True, transform=cifar_train_transform, download=True)
    test_data = dset.CIFAR100(f'{args.data_root}/cifarpy', train=False, transform=cifar_test_transform, download=True)
    num_classes = 100
elif args.dataset == 'imagenet-1k':
    train_data = dset.ImageFolder(f'{args.data_root}/imagenet-1k/train', transform=in_train_transform)
    test_data = dset.ImageFolder(f'{args.data_root}/imagenet-1k/val', transform=in_test_transform)
    num_classes = 1000
elif args.dataset == 'imagenet-100':
    train_data = dset.ImageFolder(f'{args.data_root}/imagenet-100/train', transform=in_train_transform)
    test_data = dset.ImageFolder(f'{args.data_root}/imagenet-100/val', transform=in_test_transform)
    num_classes = 100
else:
    raise ValueError('Unknown dataset')

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True, generator=g,
    worker_init_fn=seed_worker, )
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True, generator=g,
    worker_init_fn=seed_worker, )

# Create model
if args.model == 'rn50':
    net = VirtualResNet50(num_classes, null_space_red_dim=args.null_space_red_dim)
elif args.model == 'rn34':
    net = VirtualResNet34(num_classes, null_space_red_dim=args.null_space_red_dim)
else:
    net = WideResNet(args.wrn_layers, num_classes, args.wrn_widen_factor, drop_rate=args.wrn_droprate,
                     null_space_red_dim=args.null_space_red_dim)

if args.null_space_red_dim > 0:
    args.model = f'{args.model}_nsr{args.null_space_red_dim}'

# FFS
n_fts = net.nChannels if args.null_space_red_dim <= 0 else args.null_space_red_dim
flow_model = None if not args.use_ffs else build_ffs_flow(n_fts)

start_epoch = 0
# Restore model if desired
if args.load != '':
    model_name = args.load.split('/')[-1]
    if 'epoch_' in model_name:
        start_epoch = int(model_name.split('_')[-1].split('.')[0])
    else:
        print('No epoch number found in model name. Using epoch=0.')
    if os.path.isfile(model_name):
        net.load_state_dict(torch.load(str(model_name)))
        print('Model restored! Epoch:', start_epoch)

if args.ngpu > 1:
    net = VirtualDataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    if flow_model is not None:
        flow_model.cuda()
    torch.cuda.manual_seed(args.seed)

cudnn.deterministic = True
cudnn.benchmark = False

# VOS Weights
weight_energy = torch.nn.Linear(num_classes, 1).cuda()
torch.nn.init.uniform_(weight_energy.weight)
eye_matrix = torch.eye(n_fts, device='cuda')
logistic_regression = torch.nn.Linear(1, 2)
logistic_regression = logistic_regression.cuda()

data_samples = [list() for _ in range(num_classes)]

# Optimizer
optimizer = torch.optim.SGD(
    list(net.parameters()) + list(weight_energy.parameters()) +
    list(logistic_regression.parameters()), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)
scheduler = utils_training.get_scheduler(optimizer, args.epochs, args.learning_rate, train_loader)


# /////////////// Training ///////////////
def train(epoch):
    net.train()  # enter train mode
    loss_avg = 0.0
    smin_loss_avg = 0.0
    nll_loss_avg = 0.0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)

        # energy regularization.
        num_sampled_objects = sum(len(data_sample) for data_sample in data_samples)
        lr_reg_loss = torch.zeros(1).cuda()[0]
        ########################################################################################
        #                               Flow Feature Synthesis                                 #
        ########################################################################################
        nll_loss = torch.zeros(1).cuda()[0]
        if num_sampled_objects == num_classes * args.vos_sample_number and epoch < args.vos_start_epoch:
            # maintaining an ID data queue for each class.
            target_numpy = target.cpu().data.numpy()
            if args.use_ffs:
                z, sldj = flow_model(output.detach().cuda())
                nll_loss = nll(z, sldj).mean()
            else:
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    data_samples[dict_key].append(output[index].detach().view(1, -1))
                    data_samples[dict_key] = data_samples[dict_key][1:]
        elif num_sampled_objects == num_classes * args.vos_sample_number and epoch >= args.vos_start_epoch:
            if args.use_ffs:
                z, sldj = flow_model(output.detach().cuda())
                nll_loss = nll(z, sldj).mean()

                # randomly sample from latent space of flow model
                with torch.no_grad():
                    z_randn = torch.randn((args.vos_sample_from, 1024), dtype=torch.float32).cuda()
                    negative_samples, _ = flow_model(z_randn, rev=True)
                    # negative_samples = torch.sigmoid(negative_samples)
                    _, sldj_neg = flow_model(negative_samples)
                    nll_neg = nll(z_randn, sldj_neg)
                    cur_samples, index_prob = torch.topk(nll_neg, args.vos_select)
                    ood_samples = negative_samples[index_prob].view(1, -1)
                    # ood_samples = torch.squeeze(ood_samples)
                    del negative_samples
                    del z_randn
            else:
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    data_samples[dict_key].append(output[index].detach().view(1, -1))
                    data_samples[dict_key] = data_samples[dict_key][1:]

                # the covariance finder needs the data to be centered.
                data_tensor = torch.stack([torch.cat(data_samples[index]) for index in range(num_classes)], 0)
                mean_embed_id = torch.mean(data_tensor, dim=1)
                centered_data = data_tensor - mean_embed_id.view(num_classes, 1, -1)
                centered_data = centered_data.flatten(0, -2)
                # add the variance.
                temp_precision = torch.mm(centered_data.t(), centered_data) / len(centered_data)
                temp_precision += 0.0001 * eye_matrix

                ood_samples = []
                for index in range(num_classes):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((args.vos_sample_from,))
                    prob_density = new_dis.log_prob(negative_samples)
                    # breakpoint()
                    # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                    # keep the data in the low density area.
                    cur_samples, index_prob = torch.topk(- prob_density, args.vos_select)
                    ood_samples.append(negative_samples[index_prob])
            if len(ood_samples) != 0:
                ood_samples = torch.cat(ood_samples)
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = utils_training.log_sum_exp(x, weight_energy, 1)
                if args.null_space_red_dim > 0:
                    predictions_ood = net.fc[2](ood_samples)
                else:
                    predictions_ood = net.fc(ood_samples)
                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = utils_training.log_sum_exp(predictions_ood, weight_energy, 1)

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                           torch.zeros(len(ood_samples)).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())

                # if epoch % 5 == 0:
                #     print(lr_reg_loss)
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if len(data_samples[dict_key]) < args.vos_sample_number:
                    data_samples[dict_key].append(output[index].detach().view(1, -1))
        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        # breakpoint()
        loss += args.vos_loss_weight * lr_reg_loss
        loss += nll_loss * 1e-4

        if args.smin_loss_weight > 0:
            fcw = net.fc.weight if args.null_space_red_dim <= 0 else net.fc[2].weight
            smin = torch.linalg.svdvals(fcw)[-1]
            if args.use_conditioning:
                smax = torch.linalg.svdvals(fcw)[0]
                smin_loss = args.smin_loss_weight * (smax / smin)
            else:
                smin_loss = args.smin_loss_weight * (1 / smin)

            loss += smin_loss
        else:
            smin_loss = 0

        loss.backward()

        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        smin_loss_avg = smin_loss_avg * 0.8 + float(smin_loss) * 0.2
        nll_loss_avg = nll_loss_avg * 0.8 + float(nll_loss * 1e-4) * 0.2

    state['train_loss'] = loss_avg
    state['smin_loss'] = smin_loss_avg
    state['nll_loss'] = nll_loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

fn = args.dataset + '_' + args.model + \
     '_' + str(args.vos_loss_weight) + \
     '_' + str(args.vos_sample_number) + '_' + str(args.vos_start_epoch) + '_' + \
     str(args.vos_select) + '_' + str(args.vos_sample_from)
if args.smin_loss_weight > 0:
    fn += f'_smin{args.smin_loss_weight}_cond{args.use_conditioning}'
if args.use_ffs:
    fn += '_ffs'
csv_file_name = os.path.join(args.save, fn + '_baseline_training_results.csv')

with open(csv_file_name, 'w') as f:
    f.write('epoch,time(s),train_loss,smin_loss_test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train(epoch)
    test()
    model_name = args.dataset + '_' + args.model + \
                 '_baseline' + '_' + str(args.vos_loss_weight) + \
                 '_' + str(args.vos_sample_number) + '_' + str(args.vos_start_epoch) + '_' + \
                 str(args.vos_select) + '_' + str(args.vos_sample_from)
    if args.smin_loss_weight > 0:
        model_name += f'_smin{args.smin_loss_weight}_cond{args.use_conditioning}'
    if args.use_ffs:
        model_name += '_ffs'
    model_name += '_epoch_'
    prev_path = model_name + str(epoch - 1) + '.pt'
    model_name = model_name + str(epoch) + '.pt'
    # Save model
    torch.save(net.state_dict(), os.path.join(args.save, model_name))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, prev_path)
    if os.path.exists(prev_path):
        os.remove(prev_path)

    # Show results

    with open(csv_file_name, 'w') as f:
        f.write('%03d,%05d,%0.6f,%0.6f,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['smin_loss'],
            state['nll_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Smin Loss {3:.4f} | NLL Loss {4:.4f} | Test Loss {5:.3f} | Test Error {6:.2f}'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state['train_loss'],
            state['smin_loss'],
            state['nll_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'])
    )
