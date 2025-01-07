import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from . import svhn_loader as svhn

from . import score_calculation as lib
from .display_results import get_measures, print_measures_with_std, print_measures

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]


def concat(x):
    return np.concatenate(x, axis=0)


def to_np(x):
    return x.data.cpu().numpy()


def get_cifar_test_transforms():
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    return test_transform


def get_in_test_transforms(dataset):
    if 'cifar' in dataset:
        return trn.Compose([
            trn.Resize(size=(224, 224), interpolation=trn.InterpolationMode.BICUBIC),
            trn.CenterCrop(size=(224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    elif 'imagenet' in dataset:
        return trn.Compose([
            trn.Resize(size=256, interpolation=trn.InterpolationMode.BICUBIC),
            trn.CenterCrop(size=224),
            trn.ToTensor(),
            trn.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    return trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


def get_ood_scores(model, loader, ood_num_examples, test_bs, score='energy', temp=1., use_xent=False, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // test_bs and in_dist is False:
                break

            data = data.cuda()

            output = model(data)
            smax = to_np(F.softmax(output, dim=1))

            if use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if score == 'energy':
                    _score.append(-to_np((temp * torch.logsumexp(output / temp, dim=1))))
                else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def get_and_print_results(in_score,
                          model, 
                          ood_loader, 
                          ood_num_examples, 
                          num_to_avg, 
                          test_bs, 
                          score='energy', 
                          temp=1., 
                          use_xent=False,
                          method_name='Baseline',
                          noise=0.,
                          num_classes=-1,
                          mahalanobis_sample_mean=None,
                          mahalanobis_precision=None,
                          mahalanobis_count=0,
                          num_batches=1,
                          out_as_pos=False):
    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, model, test_bs, ood_num_examples, temp, noise)
        elif score == 'M':
            out_score = lib.get_Mahalanobis_score(model, ood_loader, num_classes, 
                                                  mahalanobis_sample_mean, 
                                                  mahalanobis_precision, mahalanobis_count - 1,
                                                  noise, num_batches)
        else:
            out_score = get_ood_scores(model, ood_loader, ood_num_examples, test_bs, score=score, temp=temp,
                                       use_xent=use_xent)
        if out_as_pos:  # OE's defines out samples as positive
            auroc, aupr, fpr = get_measures(out_score, in_score)
        else:
            auroc, aupr, fpr = get_measures(-in_score, -out_score)
        aurocs.append(auroc)
        auprs.append(aupr)
        fprs.append(fpr)
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, method_name)
    else:
        print_measures(auroc, aupr, fpr, method_name)
    return auroc, aupr, fpr


def get_textures_dataloader(root, test_bs, resize=32, center_crop=32):
    ood_data = dset.ImageFolder(root=f"{root}/dtd/images",
                                transform=trn.Compose([trn.Resize(resize), trn.CenterCrop(center_crop),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=4, pin_memory=True)
    return ood_loader


def get_svhn_dataloader(root, test_bs):
    ood_data = svhn.SVHN(root=f'{root}/svhn/', split="test",
                         transform=trn.Compose(
                             [  # trn.Resize(32),
                                 trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=2, pin_memory=True)
    return ood_loader


def get_places365_dataloader(root, test_bs, resize=32, center_crop=32, partition='places365'):
    ood_data = dset.ImageFolder(root=f"{root}/{partition}/",
                                transform=trn.Compose([trn.Resize(resize), trn.CenterCrop(center_crop),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=2, pin_memory=True)
    return ood_loader


def get_lsun_c_dataloader(root, test_bs):
    ood_data = dset.ImageFolder(root=f"{root}/LSUN_C",
                                transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=1, pin_memory=True)
    return ood_loader


def get_lsun_resize_dataloader(root, test_bs):
    ood_data = dset.ImageFolder(root=f"{root}/LSUN_resize",
                                transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=1, pin_memory=True)
    return ood_loader


def get_isun_dataloader(root, test_bs):
    ood_data = dset.ImageFolder(root=f"{root}/iSUN",
                                transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=1, pin_memory=True)
    return ood_loader


def get_inat_dataloader(root, test_bs, resize=32, center_crop=32):
    ood_data = dset.ImageFolder(root=f"{root}/iNaturalist",
                                transform=trn.Compose([trn.Resize(resize), trn.CenterCrop(center_crop),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=2, pin_memory=True)
    return ood_loader

def get_sun_dataloader(root, test_bs, resize=256, center_crop=224):
    ood_data = dset.ImageFolder(root=f"{root}/SUN",
                                transform=trn.Compose([trn.Resize(resize), trn.CenterCrop(center_crop),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=1, pin_memory=True)
    return ood_loader
