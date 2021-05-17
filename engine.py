# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import kornia as K

from losses import DistillationLoss
import utils
import torch.nn as nn
import torch.nn.functional as F

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def patch_level_aug(input1, patch_transform, upper_limit, lower_limit):
    bs, channle_size, H, W = input1.shape
    patches = input1.unfold(2, 16, 16).unfold(3, 16, 16).permute(0,2,3,1,4,5).contiguous().reshape(-1, channle_size,16,16)
    patches = patch_transform(patches)
 
    patches = patches.reshape(bs, -1, channle_size,16,16).permute(0,2,3,4,1).contiguous().reshape(bs, channle_size*16*16, -1)
    output_images = F.fold(patches, (H,W), 16, stride=16)
    output_images = clamp(output_images, lower_limit, upper_limit)


def train_one_epoch(args, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
    mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16,16), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
                )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)
            aug_samples = aug_samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if args.use_patch_aug:
                outputs2 = model(aug_samples)
                loss = criterion(samples, outputs, targets) + criterion(aug_samples, outputs2, targets)
            else:
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
