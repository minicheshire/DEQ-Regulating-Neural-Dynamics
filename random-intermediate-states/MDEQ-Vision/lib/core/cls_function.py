# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.cls_evaluate import accuracy
sys.path.append("../")
from utils.utils import save_checkpoint, AverageMeter
import random
from tqdm import tqdm


logger = logging.getLogger(__name__)

# TRADES loss
def trades_adv(model, data, criterion_loss, epsilon, step_size, num_steps, isTrain=False):
    model.eval()
    x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()

    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            output_adv = model(x_adv, train_step=-1, compute_jac_loss=False, spectral_radius_mode=False, writer=None)[0]
            output_nat = model(data,  train_step=-1, compute_jac_loss=False, spectral_radius_mode=False, writer=None)[0]
            loss_c = criterion_loss(F.log_softmax(output_adv, dim=1), F.softmax(output_nat, dim=1))
        grad = torch.autograd.grad(loss_c, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv

# PGD attack
def pgd(model, data, target, criterion, epsilon, step_size, num_steps, isTrain=False):
    model.eval()
    x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    with torch.enable_grad():
        for k in range(num_steps):
            x_adv.requires_grad_()
            output = model(x_adv, train_step=-1, compute_jac_loss=False, spectral_radius_mode=False, writer=None)[0]
            if not isTrain:
                index = torch.where(output.max(1)[1] == target)
                if len(index[0]) == 0: break

            model.zero_grad()
            loss_adv = criterion(output, target)

            loss_adv.backward()
            if not isTrain:
                eta = step_size * x_adv.grad.sign()[index[0], :, :, :]
                x_adv = x_adv.detach()
                x_adv[index[0], :, :, :] = x_adv[index[0], :, :, :] + eta
            else:
                eta = step_size * x_adv.grad.sign()
                x_adv = x_adv.detach() + eta

            x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv
        
def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict=None, topk=(1,5), epsilon=0, step_size=0, num_steps=0, is_mixed=False, need_adv_pretrain=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    jac_losses = AverageMeter()
    top1_adv = AverageMeter()
    top5_adv = AverageMeter()
    top1_nat = AverageMeter()
    top5_nat = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None
    global_steps = writer_dict['train_global_steps']
    update_freq = config.LOSS.JAC_INCREMENTAL

    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    criterion_loss  = nn.KLDivLoss(size_average=False)
    for i, (input, target) in enumerate(train_loader):
        # train on partial training data
        if i >= effec_batch_num: break
            
        # measure data loading time
        data_time.update(time.time() - end)

        # compute jacobian loss weight (which is dynamically scheduled)
        deq_steps = global_steps - config.TRAIN.PRETRAIN_STEPS
        if deq_steps < 0:
            # We can also regularize output Jacobian when pretraining
            factor = config.LOSS.PRETRAIN_JAC_LOSS_WEIGHT
        elif epoch >= config.LOSS.JAC_STOP_EPOCH:
            # If are above certain epoch, we may want to stop jacobian regularization training
            # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
            # will be dominating and hurt performance!)
            factor = 0
        else:
            # Dynamically schedule the Jacobian reguarlization loss weight, if needed
            factor = config.LOSS.JAC_LOSS_WEIGHT + 0.1 * (deq_steps // update_freq)
        compute_jac_loss = (torch.rand([]).item() < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)
        delta_f_thres = torch.randint(-config.DEQ.RAND_F_THRES_DELTA,2,[]).item() if (config.DEQ.RAND_F_THRES_DELTA > 0 and compute_jac_loss) else 0
        f_thres = config.DEQ.F_THRES + delta_f_thres
        b_thres = config.DEQ.B_THRES
        input  = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        loss0 = jac_loss0 = None
        if num_steps != 0 and global_steps > config.TRAIN.PRETRAIN_STEPS:

            x_adv = trades_adv(model, input, criterion_loss=criterion_loss, epsilon=epsilon, step_size=step_size, num_steps=num_steps)
            model.train()

            output_nat, jac_loss_nat, _ = model(input, train_step=(lr_scheduler._step_count-1), 
                                                compute_jac_loss=compute_jac_loss,
                                                f_thres=f_thres, b_thres=b_thres, writer=writer)

            output_adv, jac_loss_adv, _ = model(x_adv, train_step=(lr_scheduler._step_count-1), 
                                                compute_jac_loss=compute_jac_loss,
                                                f_thres=f_thres, b_thres=b_thres, writer=writer)

            loss_nat     = criterion(output_nat, target)
            jac_loss_nat = jac_loss_nat.mean()
            jac_loss_adv = jac_loss_adv.mean()
            loss_robust  = (1.0 / input.shape[0]) * criterion_loss(F.log_softmax(output_adv, dim=1), F.softmax(output_nat, dim=1))
            loss         = loss_nat + 6 * loss_robust
            jac_loss     = jac_loss_nat + jac_loss_adv

            prec1_nat, prec5_nat = accuracy(output_nat, target, topk=topk)
            prec1_adv, prec5_adv = accuracy(output_adv, target, topk=topk)

        else:
            output, jac_loss, _ = model(input, train_step=(lr_scheduler._step_count-1), 
                                        compute_jac_loss=compute_jac_loss,
                                        f_thres=f_thres, b_thres=b_thres, writer=writer)

            loss     = criterion(output, target)
            jac_loss = jac_loss.mean()
            prec1_nat, prec5_nat = accuracy(output, target, topk=topk)

        # compute gradient and do update step
        optimizer.zero_grad()
        if factor > 0:
            (loss + factor*jac_loss).backward()
        else:
            loss.backward()
        if config.TRAIN.CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP)
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if compute_jac_loss:
            jac_losses.update(jac_loss.item(), input.size(0))

#        prec1, prec5 = accuracy(output, target, topk=topk)
        top1_nat.update(prec1_nat[0], input.size(0))
        top5_nat.update(prec5_nat[0], input.size(0))
        if global_steps > config.TRAIN.PRETRAIN_STEPS:
            top1_adv.update(prec1_adv[0], input.size(0))
            top5_adv.update(prec5_adv[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}] ({3})\t' \
                  'Time {batch_time.avg:.3f}s\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.avg:.3f}s\t' \
                  'Loss {loss.avg:.5f}\t' \
                  'Jac (gamma) {jac_losses.avg:.4f} ({factor:.4f})\t' \
                  'LR {lr:.6f}\t' \
                  'natAcc@1 {top1_nat.avg:.3f}\t' \
                  'advAcc@1 {top1_adv.avg:.3f}'.format(
                      epoch, i, effec_batch_num, global_steps, batch_time=batch_time,
                      speed=input.size(0)/batch_time.avg,
                      data_time=data_time, loss=losses, jac_losses=jac_losses, factor=factor, lr=optimizer.param_groups[0]['lr'], top1_nat=top1_nat, top1_adv=top1_adv)
            if 5 in topk:
                msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
            logger.info(msg)
            
        global_steps += 1
        writer_dict['train_global_steps'] = global_steps
        
        if factor > 0 and global_steps > config.TRAIN.PRETRAIN_STEPS and (deq_steps+1) % update_freq == 0:
             logger.info(f'Note: Adding 0.1 to Jacobian regularization weight.')

def validate(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
             writer_dict=None, topk=(1,5), spectral_radius_mode=False, epsilon=0, step_size=0, num_steps=0):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    spectral_radius_mode = spectral_radius_mode and (epoch % 10 == 0)
    if spectral_radius_mode:
        sradiuses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None

    # switch to evaluate mode
    model.eval()

    if num_steps == -1:
        l = [x for (x, y) in val_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in val_loader]
        y_test = torch.cat(l, 0)
        import torchvision.transforms as transforms
        from autoattack import AutoAttack
        forward_pass = lambda x: model(x, train_step=(-1 if epoch < 0 else (lr_scheduler._step_count-1)), compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode, writer=writer)[0]
        adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard')
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=config.TEST.BATCH_SIZE_PER_GPU)
        return

    with torch.no_grad():
        end = time.time()
        # tk0 = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            if num_steps != 0:

                if num_steps == -1:
                    x_adv = adversary.run_standard_evaluation(input.cuda(), target, bs=input.shape[0])
                else:
                    x_adv = pgd(model, input, target, criterion, epsilon=epsilon, step_size=step_size, num_steps=num_steps)

                output, _, sradius = model(x_adv, 
                                    train_step=(-1 if epoch < 0 else (lr_scheduler._step_count-1)),
                                    compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode,
                                    writer=writer, isEval=True)
            else:
                # compute output
                output, _, sradius = model(input, 
                                    train_step=(-1 if epoch < 0 else (lr_scheduler._step_count-1)),
                                    compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode,
                                    writer=writer, isEval=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=topk)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if spectral_radius_mode:
                sradius = sradius.mean()
                sradiuses.update(sradius.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    if spectral_radius_mode:
        logger.info(f"Spectral radius over validation set: {sradiuses.avg}")    
    msg = 'Test: Time {batch_time.avg:.3f}\t' \
            'Loss {loss.avg:.4f}\t' \
            'Acc@1 {top1.avg:.3f}\t'.format(
                batch_time=batch_time, loss=losses, top1=top1)
    if 5 in topk:
        msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
    logger.info(msg)

    if writer:
        writer.add_scalar('accuracy/valid_top1', top1.avg, epoch)
        if spectral_radius_mode:
            writer.add_scalar('stability/sradius', sradiuses.avg, epoch)

    return top1.avg
