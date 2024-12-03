# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from apex import amp

from classification.models.resnet import ResNetModel


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).
       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds
    """
    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)


parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='in100',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=80, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=80, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=8, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='energy', help='OE|energy')
parser.add_argument('--add_slope', type=int, default=0)
parser.add_argument('--add_class', type=int, default=0)
parser.add_argument('--vanilla', type=int, default=1)
parser.add_argument('--oe', type=int, default=0)
parser.add_argument('--T', type=float, default=1.0)
parser.add_argument('--my_info', type=str, default='')
parser.add_argument('--cutmix', type=int, default=0)
parser.add_argument('--augmix', type=int, default=0)
parser.add_argument('--vos', type=int, default=0)
parser.add_argument('--gan', type=int, default=0)
parser.add_argument('--r50', type=int, default=0)
parser.add_argument('--godin', type=int, default=0)
parser.add_argument('--deepaugment', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--apex', type=int, default=0)
parser.add_argument('--additional_info', type=str, default='')
parser.add_argument('--energy_weight', type=float, default=1)  # change this to 19.2 if you are using cifar-100.
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')

parser.add_argument('--smin_loss_weight', type=float, default=0.0)
parser.add_argument('--use_conditioning', action='store_true')
parser.add_argument('--null_space_red_dim', type=int, default=-1)

parser.add_argument('--id-root', type=str, default='./data/imagenet-100')
parser.add_argument('--ood-root', type=str, default='./data/ood_in100')
args = parser.parse_args()

if args.score == 'OE':
    save_info = 'oe_tune'
elif args.score == 'energy':
    save_info = 'energy_ft_sd'
else:
    raise ValueError('Unknown score type')

args.save = args.save + save_info
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

traindir = os.path.join(args.id_root, 'train')
valdir = os.path.join(args.id_root, 'val')
normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

if args.augmix:
    train_data_in = torchvision.datasets.ImageFolder(
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
    train_data_in = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))
if args.deepaugment:
    edsr_dataset = torchvision.datasets.ImageFolder(
        '/nobackup-fast/dataset/my_xfdu/deepaugment/imagenet-r/DeepAugment/EDSR/',
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))

    cae_dataset = torchvision.datasets.ImageFolder(
        '/nobackup-fast/dataset/my_xfdu/deepaugment/imagenet-r/DeepAugment/CAE/',
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))
    train_data_in = torch.utils.data.ConcatDataset([train_data_in, edsr_dataset, cae_dataset])
test_data = torchvision.datasets.ImageFolder(
    valdir,
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))
num_classes = 100

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

ood_data = dset.ImageFolder(root=args.ood_root,
                            transform=trn.Compose([trn.RandomResizedCrop(224),
                                                   trn.RandomHorizontalFlip(),
                                                   trn.ToTensor(),
                                                   normalize, ]))

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.oe_batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.r50:
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
model_found = False
if args.load != '':
    pretrained_weights = torch.load(args.load)
    # breakpoint()
    for item in list(pretrained_weights.keys()):
        pretrained_weights[item[7:]] = pretrained_weights[item]
        del pretrained_weights[item]
    net.load_state_dict(pretrained_weights, strict=False)

logistic_regression = torch.nn.DataParallel(torch.nn.Sequential(
    torch.nn.Linear(1, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 2)
).cuda(), device_ids=list(range(args.ngpu)))
optimizer = torch.optim.SGD(
    list(net.parameters()) + list(logistic_regression.parameters()),
    state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

if args.ngpu > 1:
    if args.apex:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1", loss_scale=1.0)
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

cudnn.deterministic = True
cudnn.benchmark = False

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


def train_permute():
    net.train()  # enter train mode
    loss_avg = 0.0
    loss_energy_avg = 0.0
    smin_loss_avg = 0.0

    batch_iterator = iter(train_loader_out)
    for i, in_set in enumerate(train_loader_in):
        try:
            out_set = next(batch_iterator)
        except StopIteration:

            batch_iterator = iter(train_loader_out)
            out_set = next(batch_iterator)

        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        # print(out_set[1])

        data, target = data.cuda(), target.cuda()

        # forward
        permutation_idx = torch.randperm(len(data))
        fake_target = torch.cat([target, torch.ones(len(out_set[0])).cuda() * -1], -1)
        binary_labels = torch.ones(len(data)).cuda()
        binary_labels[len(in_set[0]):] = 0
        x = net(data[permutation_idx])

        optimizer.zero_grad()

        # cross-entropy from softmax distribution to uniform distribution
        if args.add_class:
            target = torch.cat([target, torch.ones(len(out_set[0])).cuda().long() * (num_classes - 1)], -1)
            loss = F.cross_entropy(x, target)
        else:
            # breakpoint()
            loss = F.cross_entropy(x[binary_labels[permutation_idx].bool()],
                                   fake_target[permutation_idx][binary_labels[permutation_idx].bool()].long())
            Ec_out = torch.logsumexp(x[(1 - binary_labels[permutation_idx]).bool()], dim=1) / args.T
            Ec_in = torch.logsumexp(x[binary_labels[permutation_idx].bool()], dim=1) / args.T

            input_for_lr = torch.cat((Ec_in, Ec_out), -1)
            criterion = torch.nn.CrossEntropyLoss()
            # breakpoint()
            output1 = logistic_regression(input_for_lr.reshape(-1, 1))
            energy_reg_loss = criterion(output1, binary_labels.long())

            loss += args.energy_weight * energy_reg_loss
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
        loss_energy_avg = loss_energy_avg * 0.8 + float(args.energy_weight * energy_reg_loss) * 0.2
        smin_loss_avg = smin_loss_avg * 0.8 + float(smin_loss) * 0.2
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
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
    feat_all = []
    target_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            # feat_all.append(feat)
            # target_all.append(target)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)
    # breakpoint()
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

save_info = save_info + "_slope_" + str(args.add_slope) + '_' + "weight_" + str(args.energy_weight)
save_info = save_info + '_' + args.my_info + '_' + args.additional_info

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                  '_' + save_info + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
loss_min = 100
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train_permute()
    test()
    model_name = args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) + \
                 '_' + save_info
    if args.smin_loss_weight > 0:
        model_name += f'_smin{args.smin_loss_weight}_cond{args.use_conditioning}'
    model_name += '_epoch_'
    prev_path = model_name + str(epoch - 1) + '.pt'
    model_name = model_name + str(epoch) + '.pt'
    torch.save(net.state_dict(), os.path.join(args.save, model_name))

    # Let us not waste space and delete the previous model
    torch.save(net.state_dict(), os.path.join(args.save, model_name))
    if os.path.exists(prev_path):
        os.remove(prev_path)

    # Show results
    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                      '_' + save_info + f'_smin{args.smin_loss_weight}_cond{args.use_conditioning}' +
                                      '_training_results.csv'), 'a') as f:
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
