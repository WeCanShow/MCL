#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author：fmy

import random
import copy
from data_loader import get_train_loader, get_test_loader
from utils.util import *
from models.selector import *
from config import get_arguments
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
import matplotlib.pyplot as plt


def inversion(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():

                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    attack_with_trigger(model, train_loader, target_label, trigger, mask)

    return trigger, mask

def attack_with_trigger(model, train_loader, target_label, trigger, mask):
    correct = 0
    total = 0
    trigger = trigger.to(device)
    mask = mask.to(device)
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm.tqdm(train_loader):

            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)

            _, y_pred = y_pred.max(1)
            correct += y_pred.eq(y_target).sum().item()
            total += images.size(0)
        print(correct/total)

    return trigger.cpu(), mask.cpu()


def train_step(opt, train_loader, nets, optimizer, criterions, mask, trigger, epoch):

    model = nets['model']
    backup = nets['victimized_model']

    criterionCls = criterions['criterionCls']
    cos = torch.nn.CosineSimilarity(dim=-1)

    model.train()
    backup.eval()

    for idx, (data, label) in enumerate(train_loader, start=1):

        data, label = data.clone().cuda(), label.clone().cuda()

        negative_data = copy.deepcopy(data)
        negative_data = (1 - torch.unsqueeze(mask, dim=0)) * negative_data + torch.unsqueeze(mask, dim=0) * trigger

        feature1 = model.get_final_fm(negative_data)

        feature2 = backup.get_final_fm(data)

        posi = cos(feature1, feature2.detach())
        logits = posi.reshape(-1, 1)

        feature3 = backup.get_final_fm(negative_data)

        nega = cos(feature1, feature3.detach())
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

        logits /= opt.temperature
        labels = torch.zeros(data.size(0)).cuda().long()
        cmi_loss = criterionCls(logits, labels)

        loss = cmi_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):

    test_process = []

    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['model']

    criterionCls = criterions['criterionCls']

    snet.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            output_s = snet(img)

            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = opt.log_root + '/' + opt.attack_method + '/CL_results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0]))
    df = pd.DataFrame(test_process, columns=(
        "epoch", "test_clean_acc", "test_bad_acc"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd

def cl(model, opt, trigger, mask, train_loader):
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    nets = {'model': model, 'victimized_model':copy.deepcopy(model)}

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):

        # train every epoch
        criterions = {'criterionCls': criterionCls}

        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader, nets,
                 criterions, epoch)

        print("===Epoch: {}/{}===".format(epoch + 1, opt.epochs))

        fine_defense_adjust_learning_rate(optimizer, epoch, opt.lr, opt.dataset)

        train_step(opt, train_loader, nets, optimizer, criterions, mask, trigger, epoch)

        # evaluate on testing set
        print('testing the models......')
        test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch + 1)

def reverse_engineer(opt):

    if opt.dataset == 'CIFAR10':
        dataset = 'cifar10'
        num_classes = 10
        image_size = (32,32)
        lamda_p = 0.01
    elif opt.dataset == 'imagenet':
        dataset = 'imagenet'
        num_classes = 20
        image_size = (224,224)
        lamda_p = 0.001
    elif opt.dataset == 'gtsrb':
        dataset = 'gtsrb'
        num_classes = 43
        image_size = (32,32)
        lamda_p = 0.01

    param = {
        "dataset": dataset,
        "Epochs": 100,
        "batch_size": 64,
        "lamda": lamda_p,
        "num_classes": num_classes,
        "image_size": image_size
    }
    model = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=opt.model,
                           n_classes=opt.num_class).to(opt.device)

    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)

    trigger, mask = inversion(model, opt.target_label, train_loader, param)
    cl(model, opt, trigger, mask, train_loader)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = get_arguments().parse_args()
    random.seed(opt.seed)  # torch transforms use this seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    reverse_engineer(opt)
