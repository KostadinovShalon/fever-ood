import argparse
import os
from scipy.linalg import null_space
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from models.densenet import DenseNet3
from models.wrn import WideResNet
from utils import svhn_loader as svhn


parser = argparse.ArgumentParser(description='Visualize the features of a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--drop_rate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--checkpoint', type=str, help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--model_name', default='res', type=str)
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--proj_method', type=str, default='umap', choices=['tsne', 'umap'])
parser.add_argument('--null_space_red_dim', type=int, default=-1)
args = parser.parse_args()
print(args)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar10_' in args.method_name:
    test_data = dset.CIFAR10(f'{args.root}/cifarpy', train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100(f'{args.root}/cifarpy', train=False, transform=test_transform)
    num_classes = 100

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model_name == 'res':
    net = WideResNet(args.layers, num_classes, args.widen_factor, drop_rate=args.drop_rate,
                     null_space_red_dim=args.null_space_red_dim)
else:
    net = DenseNet3(100, num_classes, 12, reduction=0.5, bottleneck=True, drop_rate=0.0,
                    normalizer=None,
                    k=None, info=None)
# Restore model
if os.path.isfile(args.checkpoint):
    net.load_state_dict(torch.load(args.checkpoint))
    print('Model restored')
else:
    raise ValueError("No checkpoint found at '{}'".format(args.checkpoint))

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 15
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

def concat(x): return np.concatenate(x, axis=0)
def to_np(x): return x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def get_features(loader, in_dist=False):
    _fts = []
    _class = []
    _logits = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            logits, fts = net.forward_virtual(data)
            _fts.append(to_np(fts))
            _logits.append(to_np(logits))
            targets = target.numpy().squeeze()
            _class.append(targets)

    if in_dist:
        return concat(_fts).copy(), concat(_class).copy(), concat(_logits).copy()
    else:
        return concat(_fts)[:ood_num_examples].copy(), concat(_class)[:ood_num_examples].copy(), \
            concat(_logits)[:ood_num_examples].copy()


def get_ft_energy(logits, dim=1):
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    return -to_np((args.T * torch.logsumexp(logits / args.T, dim=dim)))


id_fts, cats, id_logits = get_features(test_loader, in_dist=True)
n_id = len(cats)

# Textures dataset
textures_data = dset.ImageFolder(root=f"{args.root}/dtd/images",
                                 transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                        trn.ToTensor(), trn.Normalize(mean, std)]))
textures_loader = torch.utils.data.DataLoader(textures_data, batch_size=args.test_bs, shuffle=True,
                                              num_workers=4, pin_memory=True)
textures_fts, _, textures_logits = get_features(textures_loader, in_dist=False)
n_textures = len(textures_fts)

# SVHN
svhn_data = svhn.SVHN(root=f'{args.root}/svhn/', split="test",
                      transform=trn.Compose(
                          [  # trn.Resize(32),
                              trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.test_bs, shuffle=True,
                                          num_workers=2, pin_memory=True)
svhn_fts, _, svhn_logits = get_features(svhn_loader, in_dist=False)
n_svhn = len(svhn_fts)

# Places365
places365_data = dset.ImageFolder(root=f"{args.root}/places365/",
                                  transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                         trn.ToTensor(), trn.Normalize(mean, std)]))
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True,
                                               num_workers=2, pin_memory=True)
places365_fts, _, places365_logits = get_features(places365_loader, in_dist=False)
n_places365 = len(places365_fts)

# LSUN
lsun_data = dset.ImageFolder(root=f"{args.root}/LSUN_C",
                             transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
lsun_loader = torch.utils.data.DataLoader(lsun_data, batch_size=args.test_bs, shuffle=True,
                                          num_workers=1, pin_memory=True)
lsun_fts, _, lsun_logits = get_features(lsun_loader, in_dist=False)
n_lsun = len(lsun_fts)

# iSUN
isun_data = dset.ImageFolder(root=f"{args.root}/iSUN",
                             transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True,
                                          num_workers=1, pin_memory=True)
isun_fts, _, isun_logits = get_features(isun_loader, in_dist=False)
n_isun = len(isun_fts)

all_fts = concat([id_fts, textures_fts, svhn_fts, places365_fts, lsun_fts, isun_fts])
all_logits = concat([id_logits, textures_logits, svhn_logits, places365_logits, lsun_logits, isun_logits])
energy = get_ft_energy(all_logits)
min_energy = np.min(energy)
max_energy = np.max(energy)

indices = [n_id, n_id + n_textures, n_id + n_textures + n_svhn, n_id + n_textures + n_svhn + n_places365,
           n_id + n_textures + n_svhn + n_places365 + n_lsun]

# ID vs OOD plots
if args.proj_method == 'tsne':
    reducer = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
else:
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0)
results = reducer.fit_transform(all_fts)
if isinstance(net.fc, torch.nn.Sequential):
    fc = net.fc[-1]
else:
    fc = net.fc

# Create a scatter plot with the first 'n_id' points colored by their class and the rest as black 'x's
import matplotlib

matplotlib.rcParams.update({'font.size': 16})

plt.figure()
plt.scatter(results[:indices[0], 0], results[:indices[0], 1], c=cats[:indices[0]], cmap='Set3', s=20)
plt.colorbar().set_label('In-distribution Class')
plt.scatter(results[indices[0]:indices[1], 0], results[indices[0]:indices[1], 1], c='black', marker='x', s=20,
            label='Textures')
plt.scatter(results[indices[1]:indices[2], 0], results[indices[1]:indices[2], 1], c='blue', marker='x', s=20,
            label='SVHN')
plt.scatter(results[indices[2]:indices[3], 0], results[indices[2]:indices[3], 1], c='red', marker='x', s=20,
            label='Places365')
plt.scatter(results[indices[3]:indices[4], 0], results[indices[3]:indices[4], 1], c='green', marker='x', s=20,
            label='LSUN')
plt.scatter(results[indices[4]:, 0], results[indices[4]:, 1], c='magenta', marker='x', s=20, label='iSUN')
plt.legend()
title = 'In-distribution vs OOD, t-SNE Proj.' if args.proj_method == 'tsne' else 'In-distribution vs OOD, UMAP Proj.'
plt.title(title)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
# Increase the font size
plt.savefig(f'id-vs-ood-vos-{args.proj_method}.png')
plt.show()

plt.figure()
title = 'Free Energy Score, t-SNE Proj.' if args.proj_method == 'tsne' else 'Free Energy Score, UMAP Proj.'
plt.scatter(results[:, 0], results[:, 1], c=energy, cmap='inferno', s=10)
plt.colorbar().set_label('Free Energy Score')
plt.title(title)
# Add title to the color bar
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'energy-score-vos-{args.proj_method}.png')
plt.show()

id_fts_0 = id_fts[cats == 0]
n_id_0 = len(id_fts_0)
id_logits_0 = id_logits[cats == 0]

# id_fts_1 = id_fts[cats == 1]
# n_id_1 = len(id_fts_1)
# id_logits_1 = id_logits[cats == 1]

id_fts = id_fts_0

Wt = to_np(fc.weight)
# Get SVD
U, S, V = np.linalg.svd(Wt)
lsv_dir = V[len(S) - 1]

random_dir = np.random.randn(Wt.shape[1])
random_dir = random_dir / np.linalg.norm(random_dir)

# span = np.concatenate([np.array([i*k for i in range(-10, 0)]) for k in [100, 10, 1., 0.1]])
# span = np.concatenate([span, np.zeros(1), -span[::-1]])
span = np.linspace(-50, 50, 41)
ns = null_space(Wt).T
n_ns = 3
non_changing_features = np.concatenate([np.array([_ns * s for s in span]) for _ns in ns[:n_ns]])
lsv_features = np.array([lsv_dir * s for s in span])
random_features = np.array([random_dir * s for s in span])

cat_0_ft_center = np.mean(id_fts_0, axis=0)

lsv_features = lsv_features + cat_0_ft_center
non_changing_features = cat_0_ft_center + non_changing_features
random_features = cat_0_ft_center + random_features
all_fts = concat([id_fts, non_changing_features, lsv_features, random_features])
proj_results = reducer.fit_transform(all_fts)

non_changing_logits = torch.tensor(non_changing_features, dtype=torch.float32).cuda()
non_changing_logits = to_np(fc(non_changing_logits))

lsv_logits = torch.tensor(lsv_features, dtype=torch.float32).cuda()
lsv_logits = to_np(fc(lsv_logits))

random_logits = torch.tensor(random_features, dtype=torch.float32).cuda()
random_logits = to_np(fc(random_logits))

all_logits = concat([id_logits_0, non_changing_logits, lsv_logits, random_logits])
energy = get_ft_energy(all_logits)

plt.figure()
plt.scatter(proj_results[:n_id_0, 0], proj_results[:n_id_0, 1], c=energy[:n_id_0], s=30, cmap='inferno',
            vmin=min_energy, vmax=max_energy)

colors = ['darkgray', 'gray', 'dimgray', 'green', 'blue']
labels = ['Null Space Direction'] * n_ns + ['LSV Direction', 'Random Direction']
for i in range(n_ns + 2):
    lim_inf = n_id_0 + i * len(span)
    lim_sup = n_id_0 + (i + 1) * len(span)

    x = proj_results[lim_inf:lim_sup, 0]
    y = proj_results[lim_inf:lim_sup, 1]

    plt.plot(x, y, c=colors[i % len(colors)], label=labels[i])
    plt.scatter(x, y, c=energy[lim_inf:lim_sup], s=50, cmap='inferno', vmin=min_energy, vmax=max_energy)
plt.colorbar().set_label('Free Energy Score')
plt.scatter(proj_results[(n_id_0) + (len(span) - 1) // 2, 0], proj_results[(n_id_0) + (len(span) - 1) // 2, 1],
            c='red', s=300, marker='X')
# plt.plot()
title = 'Null Space and LSV, t-SNE Proj.' if args.proj_method == 'tsne' else 'Null Space and LSV, UMAP Proj'
plt.title(title)

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'null-space-vos-{args.proj_method}.png')
plt.show()

# plt.figure()
# plt.scatter(proj_results[:, 0], proj_results[:, 1], c=energy, cmap='inferno', s=10)
# plt.colorbar()
# plt.show()

plt.figure(figsize=(7, 7))
for i in range(n_ns - 1, n_ns + 2):
    lim_inf = n_id_0 + i * len(span)
    lim_sup = n_id_0 + (i + 1) * len(span)
    plt.plot(span, energy[lim_inf:lim_sup], c=colors[i % len(colors)], label=labels[i])
plt.legend()
plt.title('Free Energy Score Change')
plt.xlabel('Distance to the class center')
plt.ylabel('Free Energy Score')
plt.tight_layout()
plt.savefig(f'energy-change-vos-{args.proj_method}.png')
plt.show()

id_fts_0 = id_fts[cats == 0]
n_id_0 = len(id_fts_0)
id_logits_0 = id_logits[cats == 0]
id_fts = id_fts_0

if isinstance(net.fc, torch.nn.Sequential):
    fc = net.fc[-1]
else:
    fc = net.fc
Wt = to_np(fc.weight)
# Get SVD
U, S, V = np.linalg.svd(Wt)
null_subspace = V[len(S):]
null_perp_subspace = V[:len(S)]
dim = null_subspace.shape[0]
xv, yv = np.meshgrid(np.linspace(0, 100 / np.sqrt(dim), 101), np.linspace(0, 100 / np.sqrt(dim), 101))
xv_ones = np.tile(xv[..., None, None], (1, 1, null_subspace.shape[0], 1))
yv_ones = np.tile(yv[..., None, None], (1, 1, null_perp_subspace.shape[0], 1))
null_subspace_basis = null_subspace.T[None, None]
null_perp_subspace_basis = null_perp_subspace.T[None, None]
ft_grid = np.matmul(null_subspace_basis, xv_ones) + np.matmul(null_perp_subspace_basis, yv_ones)
ft_grid = ft_grid[..., 0]

cat_0_ft_center = np.mean(id_fts_0, axis=0)
ft_grid = ft_grid + cat_0_ft_center

ft_grid = to_np(fc(torch.tensor(ft_grid, dtype=torch.float32).cuda()))
energy_grid = get_ft_energy(ft_grid, dim=2)
plt.figure()
xv, yv = np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101))
plt.pcolormesh(xv, yv, energy_grid - energy_grid[0, 0], cmap='inferno', vmin=-50)
plt.colorbar().set_label('Free Energy Change')
plt.xlabel('Norm of NS Component')
plt.ylabel('Norm of NSP Component')
plt.tight_layout()
plt.show()

#
# Wt = to_np(fc.weight)
# # Get SVD
# U, S, V = np.linalg.svd(Wt)
# lsv_dirs = V[:len(S)]
# span = np.linspace(-50, 50, 41)
# lsv_features = np.array([lsv_dirs * s for s in span])
#
# id_fts_0 = id_fts[cats == 0]
# cat_0_ft_center = np.mean(id_fts_0, axis=0)
# lsv_features = lsv_features + cat_0_ft_center
#
# lsv_logits = torch.tensor(lsv_features, dtype=torch.float32).cuda()
# lsv_logits = to_np(fc(lsv_logits))
# energy = get_ft_energy(lsv_logits, dim=2).T
#
# for i in range(len(energy)):
#
#     plt.plot(span, energy[i], c=f"{i / (2 * len(energy)) + 0.5}", label=f'LSV {i + 1}')

id_energy = energy[:n_id]
textures_energy = energy[n_id:n_id + n_textures]
svhn_energy = energy[n_id + n_textures:n_id + n_textures + n_svhn]
places365_energy = energy[n_id + n_textures + n_svhn:n_id + n_textures + n_svhn + n_places365]
lsun_energy = energy[n_id + n_textures + n_svhn + n_places365:n_id + n_textures + n_svhn + n_places365 + n_lsun]
isun_energy = energy[n_id + n_textures + n_svhn + n_places365 + n_lsun:]
all_ood_energy = energy[n_id:]

from scipy.stats import gaussian_kde

id_density = gaussian_kde(id_energy)
ood_density = gaussian_kde(all_ood_energy)
#
# plt.scatter(results[indices[0]:indices[1], 0], results[indices[0]:indices[1], 1], c='black', marker='x', s=20, label='Textures')
# plt.scatter(results[indices[1]:indices[2], 0], results[indices[1]:indices[2], 1], c='blue', marker='x', s=20, label='SVHN')
# plt.scatter(results[indices[2]:indices[3], 0], results[indices[2]:indices[3], 1], c='red', marker='x', s=20, label='Places365')
# plt.scatter(results[indices[3]:indices[4], 0], results[indices[3]:indices[4], 1], c='green', marker='x', s=20, label='LSUN')
# plt.scatter(results[indices[4]:, 0], results[indices[4]:, 1], c='magenta', marker='x', s=20, label='iSUN')

x = np.linspace(-35, 0, 1000)

plt.plot(x, id_density(x), label="ID dist")
plt.plot(x, ood_density(x), label="OOD dist")
plt.legend()
plt.title('LSV Energy Change')
plt.ylabel('Free Energy Score')
plt.xlabel('Distance to the class center')
plt.tight_layout()
plt.show()
