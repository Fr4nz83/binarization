import os
from datetime import datetime
import pandas as pd
import time
from progress.bar import Bar as Bar
import numpy as np

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import prune


from utils.models import create_model
from utils.dataloaders import get_dataloaders
from utils.misc import *
from utils.schedulers import WarmupCosineSchedule, WarmupLinearSchedule
from utils.arg_parser import get_parser

from binary_modules.binarylinear import BinaryLinearReCU




def train_model(args):
    print(args)

    if args.binary and args.prune:
        raise ValueError("You cannot prune and binarize...")

    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    name_dir = args.name + "__" + datestring
    log_dir = os.path.join(args.output_dir, name_dir)
    os.makedirs(log_dir)
    args.img_size = 224
    model = create_model(args)


    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True


    #todo what?
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps


    train_loader, val_loader = get_dataloaders(args)
    criterion = nn.CrossEntropyLoss().cuda()

    print("Learning rate", args.lr)
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    t_total = int(args.epochs * len(train_loader) / args.train_batch_size)

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    writer = SummaryWriter(log_dir=log_dir)

    columns = ["epoch", "top1"]
    df_log = pd.DataFrame(columns=columns)

    start_epoch = 0

    if args.resume:
            print("Loaded Checkpoint")
            print("Loading pretrained model from ", args.resume)
            sd = torch.load(args.resume)
            #start_epoch = sd['epoch']
            #todo occhio agli stric
            model.load_state_dict(sd['state_dict'], strict=False)
            #optimizer.load_state_dict(sd['optimizer'])
            #scheduler.load_state_dict(sd['lr_scheduler'])

    if args.prune:
        parameters_to_prune = []
        for n, v in model.named_modules():
            if "fc" in n:
                parameters_to_prune.append((v, "weight"))
                print(n)
        print(parameters_to_prune)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=args.percentage,
        )

    model = torch.nn.DataParallel(model)
    model.cuda()

    best_prec1 = -1
    best_epoch = -1

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Training with automatic mixed precision")
    else:
        scaler = None

    print("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    model.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        #todo warmup is currently included in the schedulers
        # if args.warmup_epochs != 0 and epoch <= args.warmup_epochs:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr * (epoch + 1) / (args.warmup_epochs +1)

        if args.binary_mode == "BinaryLinearReCU":
            tau = cpt_tau(epoch, args)
            for _, module in model.named_modules():
                if isinstance(module, BinaryLinearReCU):
                    module.tau = tau.cuda()


        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
        print('Current lr {:.5f}'.format(optimizer.param_groups[0]['lr']))
        loss = train(train_loader, model, criterion, optimizer, scaler, args.max_grad_norm)
        prec1 = validate(val_loader, model, criterion, scaler=None)

        #todo same as above
        # if args.warmup_epochs == 0 or epoch >= args.warmup_epochs:
        #     lr_scheduler.step()
        scheduler.step()
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('Top1', prec1, epoch)
        is_best = prec1 > best_prec1
        current_df = pd.DataFrame([[epoch, prec1]], columns=columns)
        df_log = df_log.append(current_df)


        if is_best:
            best_prec1 = prec1
            best_epoch = epoch

        print("Current Top1 {:.4f}\t Best Top1 {:.4f}".format(prec1, best_prec1))
        model_state_dict = model.module.state_dict() \
            #if len(args.gpus) > 1 else model.state_dict()
        model_optimizer = optimizer.state_dict()
        #todo same as above
        model_scheduler = scheduler.state_dict()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_state_dict,
            'best_prec1': best_prec1,
            'optimizer': model_optimizer,
            'scheduler': model_scheduler,
            'scaler': scaler

        }, is_best, path=log_dir)



    csv_path = os.path.join(log_dir, "log.csv")
    print("Log file saved to " + csv_path)
    df_log.to_csv(csv_path)
    f = open(csv_path, 'a+')
    f.write(str(args))
    f.write("\n")
    f.write("Best model at epoch {} \t Top1: {:.4f}"
            .format(best_epoch, best_prec1))
    writer.close()
    return best_prec1


def train(train_loader, model, criterion, optimizer, scaler=None, max_grad_norm=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to train mode
    model.train()
    bar = Bar('Processing', max=len(train_loader))
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)
        target_var = target

        if scaler:
            with torch.cuda.amp.autocast():
                output = model(input_var)
                loss = criterion(output, target_var)
        else:
            output = model(input_var)
            loss = criterion(output, target_var)

        # compute gradient and do SGD step

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss)

        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        check_nan_grad(model)
        optimizer.step()
        check_nan(model)
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Top1: {top1:.4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg

        )
        bar.next()
    bar.finish()
    return losses.avg


def validate(val_loader, model, criterion, scaler=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    bar = Bar('Processing', max=len(val_loader))
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(input_var)
                    loss = criterion(output, target_var)
            else:
                output = model(input_var)
                loss = criterion(output, target_var)

            output = output.float()
            if scaler:
                scaler.scale(loss)

            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Top1: {top1:.4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg

            )
            bar.next()

        bar.finish()

    return top1.avg


def check_nan(model):
    #print("Layers with nan")
    for n, p in model.named_parameters():
        #if "weight" in n and "fc" in n:
        if torch.sum(torch.isnan(p)) != 0:
            print(n)
            raise ValueError("Nan in weights")


def check_nan_grad(model):
    #print("Layers with grad nan")

    for n, p in model.named_parameters():
        if torch.sum(torch.isnan(p.grad)) != 0:
            print(n)
            raise ValueError("Nan in gradients")


def cpt_tau(epoch, args):
    print("Recomputing tau at epoch", epoch)
    a = torch.tensor(np.e)
    T_min, T_max = torch.tensor(args.tau_min).float(), torch.tensor(args.tau_max).float()
    A = (T_max - T_min) / (a - 1)
    B = T_min - A
    tau = A * torch.tensor([torch.pow(a, epoch/args.epochs)]).float() + B
    return tau


if __name__ == "__main__":
    args = get_parser().parse_args()
    train_model(args)