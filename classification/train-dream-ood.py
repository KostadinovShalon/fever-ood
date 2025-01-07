# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn

from models.resnet import ResNetModel

from utils import training as utils_training

parser = argparse.ArgumentParser(description='Trains a Classifier with OOD Detection using Dream-OOD and FEVER-OOD',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet-100'],
                    help='Choose CIFAR-100 and Imagenet-100')
parser.add_argument('--model', '-m', type=str, default='r50',
                    choices=['r34', 'r50'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=160, help='Batch size.')
parser.add_argument('--ood_batch_size', type=int, default=160, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--wrn-layers', default=40, type=int, help='total number of layers')
parser.add_argument('--wrn-widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--wrn-droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

# Dream-OOD specific
parser.add_argument('--add_class', type=int, default=0, help='Add class to CIFAR-100')
parser.add_argument('--energy_weight', type=float, default=2.5, help='Energy regularization weight')
parser.add_argument('--seed', type=int, default=0, help='seed')

# FEVER-OOD specific
parser.add_argument('--smin_loss_weight', type=float, default=0.0, help='Weight for least singular value/conditioning number loss.')
parser.add_argument('--use_conditioning', action='store_true', help='Use conditioning number instead of least singular value.')
parser.add_argument('--null-space-red-dim', type=int, default=-1, help='Dimensionality reduction for null space.')

parser.add_argument('--id-root', type=str, default='./data/cifarpy', help='Path to CIFAR-100 in-distribution training data')
parser.add_argument('--ood-root', type=str, default='./data/dream-ood-cifar-outliers', help='Path to OOD data')

args = parser.parse_args()

args.save = args.save + 'dream_ood'
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

if args.dataset == 'cifar100':
    train_transform, test_transform = utils_training.get_cifar_transforms()
    train_data_in = dset.CIFAR100(args.id_root, train=True, transform=train_transform)
    test_data = dset.CIFAR100(args.id_root, train=False, transform=test_transform)

    ood_data = dset.ImageFolder(root=args.ood_root,
                                transform=trn.Compose([trn.ToTensor(), trn.ToPILImage(),
                                                       trn.RandomCrop(32, padding=4),
                                                       trn.RandomHorizontalFlip(), trn.ToTensor(),
                                                       trn.Normalize(mean, std)]))
else:
    traindir = os.path.join(args.id_root, 'train')
    valdir = os.path.join(args.id_root, 'val')
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    if args.augmix:
        train_data_in = dset.ImageFolder(
            traindir,
            trn.Compose([
                trn.AugMix(),
                trn.RandomResizedCrop(224),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                normalize,
            ])
        )
    else:
        train_data_in = dset.ImageFolder(
            traindir,
            trn.Compose([
                trn.RandomResizedCrop(224),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                normalize,
            ]))
    if args.deepaugment:
        edsr_dataset = dset.ImageFolder(
            '/nobackup-fast/dataset/my_xfdu/deepaugment/imagenet-r/DeepAugment/EDSR/',
            trn.Compose([
                trn.RandomResizedCrop(224),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                normalize,
            ]))

        cae_dataset = dset.ImageFolder(
            '/nobackup-fast/dataset/my_xfdu/deepaugment/imagenet-r/DeepAugment/CAE/',
            trn.Compose([
                trn.RandomResizedCrop(224),
                trn.RandomHorizontalFlip(),
                trn.ToTensor(),
                normalize,
            ]))
        train_data_in = torch.utils.data.ConcatDataset([train_data_in, edsr_dataset, cae_dataset])
    test_data = dset.ImageFolder(
        valdir,
        trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            normalize,
        ]))
    ood_data = dset.ImageFolder(root=args.ood_root,
                                transform=trn.Compose([trn.RandomResizedCrop(224),
                                                       trn.RandomHorizontalFlip(),
                                                       trn.ToTensor(),
                                                       normalize, ]))
if args.add_class:
    num_classes = 101
else:
    num_classes = 100

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.ood_batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'r50':
    net = ResNetModel(name='resnet50', num_classes=num_classes, null_space_red_dim=args.null_space_red_dim)
else:
    net = ResNetModel(name='resnet34', num_classes=num_classes, null_space_red_dim=args.null_space_red_dim)
for p in net.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -35, 35))


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


if args.null_space_red_dim > 0:
    args.model = f'{args.model}_nsr{args.null_space_red_dim}'

# Restore model
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
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(2)

cudnn.deterministic = True
cudnn.benchmark = False

logistic_regression = torch.nn.Sequential(
    torch.nn.Linear(1, 2)
).cuda()

optimizer = torch.optim.SGD(
    list(net.parameters()) + list(logistic_regression.parameters()),
    state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

# /////////////// Training ///////////////
criterion = torch.nn.CrossEntropyLoss()


def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    loss_energy_avg = 0.0
    smin_loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):

        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        fts, x = net.forward_repre(data)

        # backward
        optimizer.zero_grad()

        # cross-entropy from softmax distribution to uniform distribution
        if args.add_class:
            target = torch.cat([target, torch.ones(len(out_set[0])).cuda().long() * (num_classes - 1)], -1)
            loss = F.cross_entropy(x, target)
        else:
            loss = F.cross_entropy(x[:len(in_set[0])], target)
        Ec_out = torch.logsumexp(x[len(in_set[0]):], dim=1)
        Ec_in = torch.logsumexp(x[:len(in_set[0])], dim=1)
        binary_labels = torch.ones(len(x)).cuda()
        binary_labels[len(in_set[0]):] = 0

        input_for_lr = torch.cat((Ec_in, Ec_out), -1)
        output1 = logistic_regression(input_for_lr.reshape(-1, 1))
        energy_reg_loss = criterion(output1, binary_labels.long())
        loss += args.energy_weight * energy_reg_loss

        # FEVER-OOD
        smin_loss = utils_training.get_fever_ood_loss(net, args.null_space_red_dim, args.use_conditioning)
        loss += args.smin_loss_weight * smin_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        smin_loss_avg = smin_loss_avg * 0.8 + float(smin_loss) * 0.2
        loss_energy_avg = loss_energy_avg * 0.8 + float(args.energy_weight * energy_reg_loss) * 0.2

    print(scheduler.get_lr())
    print('loss energy is: ', loss_energy_avg)
    state['train_loss'] = loss_avg
    state['train_smin_loss'] = smin_loss_avg
    state['train_energy_loss'] = loss_energy_avg


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

log_file = os.path.join(args.save,
                        args.dataset + '_' + args.model + '_s' + str(args.seed) +
                        '_' + "weight_" + str(args.energy_weight) + '_training_results.csv')
model_name = args.dataset + '_' + args.model + '_baseline' + '_' + "weight_" + str(args.energy_weight)
if args.smin_loss_weight > 0:
    model_name += f'_smin{args.smin_loss_weight}_cond{args.use_conditioning}'

with open(log_file, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train()
    test()

    epoch_model_name = model_name + '_epoch_'
    prev_path = epoch_model_name + str(epoch - 1) + '.pt'
    epoch_model_name = epoch_model_name + str(epoch) + '.pt'

    # Save model
    torch.save(net.state_dict(), os.path.join(args.save, epoch_model_name))

    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, prev_path)
    if os.path.exists(prev_path):
        os.remove(prev_path)

    # Show results
    with open(log_file, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.6f,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['train_smin_loss'],
            state['train_energy_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    print(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Train Smin Loss {3:.4f} | Train Energy Loss {4:.4f} | Test Loss {5:.3f} | Test Error {6:.2f}'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state['train_loss'],
            state['train_smin_loss'],
            state['train_energy_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'])
    )
