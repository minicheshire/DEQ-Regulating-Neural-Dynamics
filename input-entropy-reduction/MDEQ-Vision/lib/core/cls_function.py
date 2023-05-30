# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.cls_evaluate import accuracy
sys.path.append("../")
from utils.utils import save_checkpoint, AverageMeter
import random
from tqdm import tqdm
from lib.layer_utils import list2vec, vec2list
from lib.solvers import * 
import torch.autograd as autograd
logger = logging.getLogger(__name__)

        
def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict=None, topk=(1,5)):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    jac_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None
    global_steps = writer_dict['train_global_steps']
    update_freq = config.LOSS.JAC_INCREMENTAL

    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
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
        output, jac_loss, _ = model(input, train_step=(lr_scheduler._step_count-1), 
                                    compute_jac_loss=compute_jac_loss,
                                    f_thres=f_thres, b_thres=b_thres, writer=writer)
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target)
        jac_loss = jac_loss.mean()

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

        prec1, prec5 = accuracy(output, target, topk=topk)
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

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
                  'Acc@1 {top1.avg:.3f}\t'.format(
                      epoch, i, effec_batch_num, global_steps, batch_time=batch_time,
                      speed=input.size(0)/batch_time.avg,
                      data_time=data_time, loss=losses, jac_losses=jac_losses, factor=factor, top1=top1)
            if 5 in topk:
                msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
            logger.info(msg)
            
        global_steps += 1
        writer_dict['train_global_steps'] = global_steps
        
        if factor > 0 and global_steps > config.TRAIN.PRETRAIN_STEPS and (deq_steps+1) % update_freq == 0:
             logger.info(f'Note: Adding 0.1 to Jacobian regularization weight.')

def pgd(model, data, target, criterion, epsilon, step_size, num_steps, AttackGrad):
    model.eval()
    x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
    with torch.enable_grad():
        for k in range(num_steps):
            x_adv.requires_grad_()
            if AttackGrad == "Funroll-es-0":
                now_grad = None
                for kkk in range(8):
                    output = model(x_adv, train_step=-1, compute_jac_loss=False, spectral_radius_mode=False, writer=None, AttackGrad="Funroll-{}-1".format(kkk), for_attack=True)[0]
                    model.zero_grad()
                    loss_adv = criterion(output, target)
                    loss_adv.backward(retain_graph=True)
                    if now_grad is None:
                        now_grad = x_adv.grad.clone().detach()
                    else:
                        now_grad = now_grad + x_adv.grad.clone().detach()
                eta = step_size * now_grad.sign()
            else:
                output = model(x_adv, train_step=-1, compute_jac_loss=False, spectral_radius_mode=False, writer=None, AttackGrad=AttackGrad, for_attack=True)[0]  # experiment: for_attack=False
                model.zero_grad()
                loss_adv = criterion(output, target)
                loss_adv.backward()
                eta = step_size * x_adv.grad.sign()

            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def validate_observe_train(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
             writer_dict=None, topk=(1,5), spectral_radius_mode=False, epsilon=0, step_size=0, num_steps=0, attack="pgd", AttackGrad="unroll", TrainGrad="unroll", isTrain=False, logfile=""):
    # switch to evaluate mode
    model.eval()

    from autoattack import AutoAttack
    forward_pass = lambda x: model(x, train_step=-1, compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode, writer=None, AttackGrad=AttackGrad)[0]
    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='plus')

    with torch.no_grad():
        end = time.time()
        total_batch_num = len(val_loader)
        for i, (input, target) in enumerate(val_loader):
            logger.info(f"{i}")
            if os.path.exists(os.path.join(output_dir, f'train-{TrainGrad}-attack-{AttackGrad}-VAL_DATA-{attack}-{i}')): continue

            target = target.cuda(non_blocking=True)
            input = input.cuda()
            now_bsz = target.shape[0]

            if attack == "pgd" or attack == "pgd-es-0":
                x_adv = pgd(model=model, data=input, target=target, criterion=criterion, epsilon=epsilon, step_size=step_size, num_steps=num_steps, AttackGrad=AttackGrad)

            if attack == "pgd1000":
                x_adv = pgd(model=model, data=input, target=target, criterion=criterion, epsilon=epsilon, step_size=step_size, num_steps=1000, AttackGrad=AttackGrad)

            aa_name_list = ["apgd-ce", "apgd-t", "fab-t", "square", "apgd-dlr", "fab",
                            "apgd-ce-es-0", "apgd-t-es-0", "fab-t-es-0", "square-es-0", "apgd-dlr-es-0", "fab-es-0"]

            if attack in aa_name_list:
                adversary.attacks_to_run = [attack[:-5] if "es-0" in attack else attack]
                x_adv = adversary.run_standard_evaluation_individual(input, target, bs=input.shape[0])[attack[:-5] if "es-0" in attack else attack]
            
            now_x_advs = torch.cat([input, x_adv])
            interval = now_bsz
            
            rets = {}#{"CLEAN": [], "PERTURBED": []}
            for j in range(2):
                now_x_adv = now_x_advs[j*interval:(j+1)*interval,:,:,:].cuda()

                now_func, cutoffs = model.module.get_func(now_x_adv)
                output, zz, yy_list_list = model(x=now_x_adv, train_step=-1, compute_jac_loss=False,
                                            spectral_radius_mode=spectral_radius_mode, writer=None, AttackGrad=TrainGrad, for_attack=False) # Here we'd obtain the outputs from the original model, so "AttackGrad = TrainGrad"

                now_step = 0
                ret = {"x_adv":now_x_adv.cpu(), "yy_list_list": [], "res_list_list":[], "output_list": [], "preddd_list": [], "acc1": [], "good_loss":[], "bad_loss":[], "target":target.cpu()}#, "preddd": preddd.cpu()}
                if "es-0" in attack:
                    now = yy_list_list[0]
                    for yy_list in yy_list_list[1:]:
                        for jj in range(len(yy_list)):
                            now[jj] = now[jj] + yy_list[jj]

                    for jj in range(len(now)): now[jj] = now[jj] / len(yy_list_list)

                for yy_list in (yy_list_list if (not "es-0" in attack) else [now]):
                    now_output = model.module.predict(yy_list)
                    now_prec1, now_prec5 = accuracy(now_output, target, topk=topk)
                    good_loss = criterion(now_output, target)
                    
                    if False:
                        ret["yy_list_list"].append([yy.cpu() for yy in yy_list])
                        ret["res_list_list"].append(vec2list((now_func(list2vec(yy_list))-list2vec(yy_list)).cpu(), cutoffs))
                    ret["output_list"].append(now_output.cpu())
                    ret["preddd_list"].append(now_output.topk(1, 1, True, True)[1].t().squeeze().cpu())
                    ret["good_loss"].append(good_loss.item())
                    ret["acc1"].append(now_prec1[0].item())
                    now_step += 1
                    del now_output, now_prec5
                    torch.cuda.empty_cache()
                del output
                torch.cuda.empty_cache()

                if j == 0:
                    rets["CLEAN"] = ret#.append(ret)
                else:
                    rets["PERTURBED"] = ret#.append(ret)

                msg = "Batch {}/{}, {}, OutputFinal: ".format(i, total_batch_num, "CLEAN" if j==0 else "PERTURBED") + "good loss {}\tAcc@1 {}".format(round(good_loss.item(), 4), round(now_prec1[0].item(), 4))

                logger.info(msg)
                torch.save(rets, os.path.join(output_dir, "{}-{}-{}-{}".format(logfile, "TRAIN_DATA" if isTrain else "VAL_DATA", attack, i)))

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

    with torch.no_grad():
        end = time.time()
        # tk0 = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            if num_steps != 0:
                x_adv = pgd(model, input, target, criterion, epsilon=epsilon, step_size=step_size, num_steps=num_steps)
                output, _, sradius = model(x_adv, 
                                    train_step=(-1 if epoch < 0 else (lr_scheduler._step_count-1)),
                                    compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode,
                                    writer=writer)
            else:
                # compute output
                output, _, sradius = model(input, 
                                    train_step=(-1 if epoch < 0 else (lr_scheduler._step_count-1)),
                                    compute_jac_loss=False, spectral_radius_mode=spectral_radius_mode,
                                    writer=writer)

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
