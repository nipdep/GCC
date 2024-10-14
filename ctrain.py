#!/usr/bin/env python
# encoding: utf-8
# File Name: ctrain.py
# Author: nipdep
# Create Time: 2024/10/12


import argparse
import copy
import os
import time
import warnings

import dgl
import numpy as np
import psutil
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter

from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
    LoadBalanceNodeClassificationDataset
)
from gcc.datasets.data_util import batcher, labeled_batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

from icecream import ic

def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=6, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"] + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model-path", type=str, default=None, help="path to save model")
    parser.add_argument("--tb-path", type=str, default=None, help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    # continuous training setting
    parser.add_argument("--ctrain", action="store_true")

    # finetune setting
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use.")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true")
    # fmt: on

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def option_update(opt):
    opt.model_name = "{}_moco_{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_hid_{}_samples_{}_nce_t_{}_nce_k_{}_rw_hops_{}_restart_prob_{}_aug_{}_ft_{}_deg_{}_pos_{}_momentum_{}".format(
        opt.exp,
        opt.moco,
        opt.dataset,
        opt.model,
        opt.num_layer,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.hidden_size,
        opt.num_samples,
        opt.nce_t,
        opt.nce_k,
        opt.rw_hops,
        opt.restart_prob,
        opt.aug,
        opt.finetune,
        opt.degree_embedding_size,
        opt.positional_embedding_size,
        opt.alpha,
    )

    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def train_finetune(
    epoch,
    train_loader,
    model,
    output_layer,
    criterion,
    optimizer,
    output_layer_optimizer,
    sw,
    opt,
):
    """
    one epoch training for moco
    """
    n_batch = len(train_loader)
    model.train()
    output_layer.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0
    device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, y = batch

        graph_q = graph_q.to(device)
        y = y.to(device)

        bsz = graph_q.batch_size

        # ===================forward=====================

        feat_q = model(graph_q)

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
        out = output_layer(feat_q)

        loss = criterion(out, y)

        # ===================backward=====================
        optimizer.zero_grad()
        output_layer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        torch.nn.utils.clip_grad_value_(output_layer.parameters(), 1)
        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        for param_group in output_layer_optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()
        output_layer_optimizer.step()

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        f1_meter.update(f1, bsz)
        epoch_f1_meter.update(f1, bsz)
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        graph_size.update(graph_q.number_of_nodes() / bsz, bsz)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "f1 {f1.val:.3f} ({f1.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    f1=f1_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            sw.add_scalar("ft_loss", loss_meter.avg, global_step)
            sw.add_scalar("ft_f1", f1_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("lr", lr_this_step, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            #  sw.add_scalar(
            #      "learning_rate", optimizer.param_groups[0]["lr"], global_step
            #  )
            loss_meter.reset()
            f1_meter.reset()
            graph_size.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg, epoch_f1_meter.avg


def test_finetune(epoch, valid_loader, model, output_layer, criterion, sw, opt):
    n_batch = len(valid_loader)
    model.eval()
    output_layer.eval()

    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()

    for idx, batch in enumerate(valid_loader):
        graph_q, y = batch

        graph_q = graph_q.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size

        # ===================forward=====================

        with torch.no_grad():
            feat_q = model(graph_q)
            assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
            out = output_layer(feat_q)
        loss = criterion(out, y)

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        epoch_loss_meter.update(loss.item(), bsz)
        epoch_f1_meter.update(f1, bsz)

    global_step = (epoch + 1) * n_batch
    sw.add_scalar("ft_loss/valid", epoch_loss_meter.avg, global_step)
    sw.add_scalar("ft_f1/valid", epoch_f1_meter.avg, global_step)
    print(
        f"Epoch {epoch}, loss {epoch_loss_meter.avg:.3f}, f1 {epoch_f1_meter.avg:.3f}"
    )
    return epoch_loss_meter.avg, epoch_f1_meter.avg


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )

def cnt_train(
    epoch, train_loader, model, model_ema, contrast, criterion, optimizer, sw, opt
):
    """
    one epoch training for moco
    """
    n_batch = len(train_loader)
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch
        
        # ic(opt.gpu)
        device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')

        # graph_q.cuda(opt.gpu)
        graph_q = graph_q.to(device)
        # graph_k.cuda(opt.gpu)
        graph_k = graph_k.to(device)

        bsz = graph_q.batch_size

        if opt.moco:
            # ===================Moco forward=====================
            feat_q = model(graph_q)
            with torch.no_grad():
                feat_k = model_ema(graph_k)

            out = contrast(feat_q, feat_k)
            prob = out[:, 0].mean()
        else:
            # ===================Negative sampling forward=====================
            # ic(graph_q.device)
            feat_q = model(graph_q)
            feat_k = model(graph_k)

            out = torch.matmul(feat_k, feat_q.t()) / opt.nce_t
            prob = out[range(graph_q.batch_size), range(graph_q.batch_size)].mean()

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)

        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        graph_size.update(
            (graph_q.number_of_nodes() + graph_k.number_of_nodes()) / 2.0 / bsz, 2 * bsz
        )
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        if opt.moco:
            moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg

def train_moco(
    epoch, train_loader, model, model_ema, contrast, criterion, optimizer, sw, opt
):
    """
    one epoch training for moco
    """
    n_batch = train_loader.dataset.total // opt.batch_size
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch
        
        # ic(opt.gpu)
        device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')

        # graph_q.cuda(opt.gpu)
        graph_q = graph_q.to(device)
        # graph_k.cuda(opt.gpu)
        graph_k = graph_k.to(device)

        bsz = graph_q.batch_size

        if opt.moco:
            # ===================Moco forward=====================
            feat_q = model(graph_q)
            with torch.no_grad():
                feat_k = model_ema(graph_k)

            out = contrast(feat_q, feat_k)
            prob = out[:, 0].mean()
        else:
            # ===================Negative sampling forward=====================
            # ic(graph_q.device)
            feat_q = model(graph_q)
            feat_k = model(graph_k)

            out = torch.matmul(feat_k, feat_q.t()) / opt.nce_t
            prob = out[range(graph_q.batch_size), range(graph_q.batch_size)].mean()

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)

        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        graph_size.update(
            (graph_q.number_of_nodes() + graph_k.number_of_nodes()) / 2.0 / bsz, 2 * bsz
        )
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        if opt.moco:
            moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg

def get_optimizer(args, model):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.learning_rate,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError
    
    return optimizer

def save_model(args, model, contrast, optimizer, epoch, phase='pre_training', checkpoint=True):
    state = {
        "opt": args,
        "model": model.state_dict(),
        "contrast": contrast.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    
    save_dir = f"{args.model_folder}/{phase}"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if checkpoint:
        save_file = f"{save_dir}/ckpt_epoch_{epoch}.pth"
    else:
        save_file = f"{save_dir}/current.pth"
    torch.save(state, save_file)
    # help release GPU memory
    del state

def load_checkpoint(save_path, model, optimizer, contrast, model_ema=None):
    print("=> loading checkpoint '{}'".format(args.resume))
    # checkpoint = torch.load(args.resume, map_location="cpu")
    checkpoint = torch.load(save_path)
    args.start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    contrast.load_state_dict(checkpoint["contrast"])
    if args.moco:
        model_ema.load_state_dict(checkpoint["model_ema"])

    print(
        "=> loaded successfully '{}' (epoch {})".format(
            args.resume, checkpoint["epoch"]
        )
    )
    del checkpoint
    torch.cuda.empty_cache()

# def main(args, trial):
def main(args):
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            pretrain_args = checkpoint["opt"]
            pretrain_args.fold_idx = args.fold_idx
            pretrain_args.gpu = args.gpu
            pretrain_args.ctrain = args.ctrain
            pretrain_args.finetune = args.finetune
            pretrain_args.resume = args.resume
            pretrain_args.cv = args.cv
            pretrain_args.dataset = args.dataset
            pretrain_args.epochs = args.epochs
            pretrain_args.num_workers = args.num_workers
            if args.dataset in GRAPH_CLASSIFICATION_DSETS:
                # HACK for speeding up finetuning on graph classification tasks
                pretrain_args.num_workers = 0
            pretrain_args.batch_size = args.batch_size
            args = pretrain_args
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    args = option_update(args)
    print(args)
    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    print("setting random seeds")

    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)

    dataset = LoadBalanceNodeClassificationDataset(
        dataset=args.dataset,
        rw_hops=args.rw_hops,
        subgraph_size=args.subgraph_size,
        restart_prob=args.restart_prob,
        positional_embedding_size=args.positional_embedding_size,
    )
    labels = dataset.data.y.argmax(dim=1).tolist()

    # Create stratified shuffle split object
    stratified_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=args.seed  # Adjust test_size as needed
    )
    # Generate train and test indices
    for train_idx, test_idx in stratified_split.split(np.zeros(len(labels)), labels):
        pass  # The split method returns the indices directly

    pre_train_dataset = torch.utils.data.Subset(dataset, train_idx)
    
    
    _dataset = dataset = NodeClassificationDatasetLabeled(
            dataset=args.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    finetune_dataset = torch.utils.data.Subset(_dataset, train_idx) # TODO: need to change the pre-training vs. finetuning precentage
    test_dataset = torch.utils.data.Subset(_dataset, test_idx)


    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used / 1024 ** 3)
    pre_train_loader = torch.utils.data.DataLoader(
        dataset=pre_train_dataset,
        batch_size=args.batch_size,
        collate_fn=labeled_batcher() if args.finetune else batcher(),
        shuffle=True if args.finetune else False,
        num_workers=args.num_workers,
        worker_init_fn=None
        if args.finetune or args.dataset != "dgl"
        else worker_init_fn,
    )

    finetuning_loader = torch.utils.data.DataLoader(
            dataset=finetune_dataset,
            batch_size=args.batch_size,
            collate_fn=labeled_batcher(),
            num_workers=args.num_workers,
        )

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            collate_fn=labeled_batcher(),
            num_workers=args.num_workers,
        )

    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)

    # create model and optimizer
    # n_data = train_dataset.total
    n_data = None

    model, model_ema = [
        GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer,
            norm=args.norm,
            gnn_model=args.model,
            degree_input=True,
        )
        for _ in range(2)
    ]

    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(
        args.hidden_size, n_data, args.nce_k, args.nce_t, use_softmax=True
    ).cuda(args.gpu)

    
    pretrain_criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
    pretrain_criterion = pretrain_criterion.cuda(args.gpu)

    ft_criterion = nn.CrossEntropyLoss()

    model = model.cuda(args.gpu)
    model_ema = model_ema.cuda(args.gpu)

    optimizer = get_optimizer(args, model)
    
    # optionally resume from a checkpoint
    args.start_epoch = 1
    load_checkpoint(args.resume, model, optimizer, contrast)

    # tensorboard logger
    sw = SummaryWriter(args.tb_folder)

    # routine
    print(args.start_epoch, args.epochs)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> pre-training...")
        time1 = time.time()
        loss = cnt_train(
                epoch,
                pre_train_loader,
                model,
                model_ema,
                contrast,
                pretrain_criterion,
                optimizer,
                sw,
                args,
            )
        time2 = time.time()
        print("epoch {}, total time {:.2f}, loss {:.4f}".format(epoch, time2 - time1, loss))

        if epoch % args.save_freq == 0:
            save_model(args, model, contrast, optimizer, epoch, checkpoint=True)
    else:
        save_model(args, model, contrast, optimizer, epoch, checkpoint=False)

    # finetune
    output_layer = nn.Linear(
        in_features=args.hidden_size, out_features=dataset.num_classes
    )
    output_layer = output_layer.cuda(args.gpu)
    output_layer_optimizer = torch.optim.Adam(
        output_layer.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    def clear_bn(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.reset_running_stats()

    model.apply(clear_bn)

    # resume from pre-trained model
    args.start_epoch = 1
    pre_trained_save_path = f"{args.model_folder}/pre_training/current.pth"
    load_checkpoint(pre_trained_save_path, model, optimizer, contrast)
    model = model.cuda(args.gpu) 

    # tensorboard logger
    sw = SummaryWriter(args.tb_folder)

    # routine
    print(args.start_epoch, args.epochs)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> finetuning...")
        time1 = time.time()
        loss, _ = train_finetune(
                epoch,
                finetuning_loader,
                model,
                output_layer,
                ft_criterion,
                optimizer,
                output_layer_optimizer,
                sw,
                args,
            )
        time2 = time.time()
        print("epoch {}, total time {:.2f}, loss {:.4f}".format(epoch, time2 - time1, loss))

        if epoch % args.save_freq == 0:
            save_model(args, model, contrast, optimizer, epoch, phase='finetune', checkpoint=True)
    else:
        save_model(args, model, contrast, optimizer, epoch, phase='finetune', checkpoint=False)

    
    valid_loss, valid_f1 = test_finetune(
        epoch, test_loader, model, output_layer, ft_criterion, sw, args
    )
    print(f"[TEST] loss: {valid_loss}, f1-score: {valid_f1}")
    return valid_f1


if __name__ == "__main__":

    warnings.simplefilter("once", UserWarning)
    args = parse_option()

    if args.cv:
        gpus = args.gpu

        def variant_args_generator():
            for fold_idx in range(10):
                args.fold_idx = fold_idx
                args.num_workers = 0
                args.gpu = gpus[fold_idx % len(gpus)]
                yield copy.deepcopy(args)

        # f1 = Parallel(n_jobs=5)(
        #     delayed(main)(args) for args in variant_args_generator()
        # )
        f1 = [main(args) for args in variant_args_generator()]
        print(f1)
        print(f"Mean = {np.mean(f1)}; Std = {np.std(f1)}")
    else:
        args.gpu = args.gpu[0]
        main(args)
    # import optuna
    # def objective(trial):
    #     args.epochs = 50
    #     args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    #     args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    #     args.restart_prob = trial.suggest_uniform('restart_prob', 0.5, 1)
    #     # args.alpha = 1 - trial.suggest_loguniform('alpha', 1e-4, 1e-2)
    #     return main(args, trial)

    # study = optuna.load_study(study_name='cat_prone', storage="sqlite:///example.db")
    # study.optimize(objective, n_trials=20)
