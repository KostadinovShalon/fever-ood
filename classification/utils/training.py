from typing import Any
from itertools import chain


import numpy as np
import torch
import torchvision.transforms as trn
import torch.nn.functional as F

from .virtual_parallel_apply import virtual_parallel_apply


def get_cifar_transforms():
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                   trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    return train_transform, test_transform


def get_in_transforms():
    in_train_transform = trn.Compose([
        trn.Resize(size=224, interpolation=trn.InterpolationMode.BICUBIC),
        trn.RandomResizedCrop(size=(224, 224), scale=(0.5, 1), interpolation=trn.InterpolationMode.BICUBIC),
        trn.RandomHorizontalFlip(p=0.5),
        trn.ToTensor(),
        trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    in_test_transform = trn.Compose([
        trn.Resize(size=(224, 224), interpolation=trn.InterpolationMode.BICUBIC),
        trn.CenterCrop(size=(224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    return in_train_transform, in_test_transform


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def get_scheduler(optimizer, epochs, lr, train_loader):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / lr))


def log_sum_exp(value, weight_energy, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)


class VirtualDataParallel(torch.nn.DataParallel):

    def forward_virtual(self, *inputs: Any, **kwargs: Any) -> Any:
        with torch.autograd.profiler.record_function("DataParallel.forward_virtual"):
            if not self.device_ids:
                return self.module.forward_virtual(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError(
                        "module must have its parameters and buffers "
                        f"on device {self.src_device_obj} (device_ids[0]) but found one of "
                        f"them on device: {t.device}"
                    )

            inputs, module_kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not module_kwargs:
                inputs = ((),)
                module_kwargs = ({},)

            if len(self.device_ids) == 1:
                return self.module.forward_virtual(*inputs[0], **module_kwargs[0])
            replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
            outputs = self.virtual_parallel_apply(replicas, inputs, module_kwargs)
            return self.gather(outputs, self.output_device)

    def virtual_parallel_apply(
        self, replicas, inputs, kwargs: Any
    ):
        return virtual_parallel_apply(
            replicas, inputs, kwargs, self.device_ids[: len(replicas)]
        )
