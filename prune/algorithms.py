from __future__ import print_function

import time
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from prune.utils import save_checkpoint, AverageMeter, accuracy
from prune import misc
from models.preact_resnet import *
from domainbed.algorithms import AbstractMMD

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def train(
    args, 
    model, 
    device, 
    epoch_ites, 
    optimizer, 
    epoch, 
    criterion, 
    data_iterators, 
    group_wd_optimizer, 
    train_writer=None, 
    pruning_engine=None, 
    swad_algorithm=None,
    prune_done=False,
    global_iteration=0):
    """Train for one epoch on the training set also performs pruning"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_tracker, acc_tracker, loss_tracker_num = 0.0, 0.0, 0

    pruning = False if (args.algorithm=="TSP" and epoch<args.pretrain_epochs) else args.pruning
    model.train()
    if args.fixed_network:
        model.eval()    # if network is fixed then we put it to eval mode

    end = time.time()

    for batch_idx in range(epoch_ites):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        
        data_list, label_list, data_sizes = [], [], []
        for iterator in data_iterators:
            d, l = next(iterator)
            d, l = d.cuda(), l.cuda()
            data_list.append(d)
            label_list.append(l)
            data_sizes.append(d.shape[0])

        data = torch.cat(data_list, 0)
        target = torch.cat(label_list, 0)
        output = model(data)

        if args.algorithm=="TSP":
            loss = loss_with_Coral(model, epoch, data_list, label_list, args.pretrain_epochs, args.l1_coe)
        elif args.algorithm=="DG_pruning":
            loss = criterion(output, target)
            loss = loss_DG_pruning(model, data_list, label_list, criterion, loss)
        else:
            loss = criterion(output, target)

        if pruning:
            # useful for method 40 and 50 that calculate oracle
            pruning_engine.run_full_oracle(model, data, target, criterion, initial_loss=loss.item())

        # measure accuracy and record loss
        losses.update(loss.item(), data.size(0))
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))
        acc_tracker += prec1.item()
        loss_tracker += loss.item()
        loss_tracker_num += 1

        if pruning and pruning_engine.needs_hessian:
            pruning_engine.compute_hessian(loss)

        if not (pruning and args.method == 50):
            group_wd_optimizer.step()

        loss.backward()

        # step_after will calculate flops and number of parameters left
        # needs to be launched before the main optimizer,
        # otherwise weight decay will make numbers not correct
        if not (pruning and args.method == 50) and batch_idx % args.log_interval == 0:
            group_wd_optimizer.step_after()

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        global_iteration = global_iteration + 1
        if batch_idx % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, batch_idx, epoch_ites, batch_time=batch_time,
                      loss=losses, top1=top1, top5=top5))

            if train_writer is not None:
                train_writer.add_scalar('train_loss_ave', losses.avg, global_iteration)

        if pruning:
            pruning_engine.do_step(loss=loss.item(), optimizer=optimizer)

        if args.tensorboard and (batch_idx % args.log_interval == 0):
            neurons_left = int(group_wd_optimizer.get_number_neurons(print_output=args.get_flops))
            flops = int(group_wd_optimizer.get_number_flops(print_output=args.get_flops))

            train_writer.add_scalar('neurons_optimizer_left', neurons_left, global_iteration)
            train_writer.add_scalar('neurons_optimizer_flops_left', flops, global_iteration)
        else:
            if args.get_flops:
                neurons_left = int(group_wd_optimizer.get_number_neurons(print_output=args.get_flops))
                flops = int(group_wd_optimizer.get_number_flops(print_output=args.get_flops))

        tmp = global_iteration - 1
        if args.algorithm=="SWAD" and prune_done:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(model, step=tmp)
            
    # print number of parameters left:
    if args.tensorboard:
        print('neurons_optimizer_left', neurons_left, global_iteration)

    return global_iteration


def loss_with_Coral(model, epoch, data_list, label_list, pretrain_epochs=0, l1_coe=0):
    objective, penalty, nmb = 0, 0, len(data_list)

    features = [model.module.Featurizer(xi) for xi in data_list]
    classifs = [model.module.Classifier(fi) for fi in features]
    targets = [yi for yi in label_list]

    for i in range(nmb):
        objective += F.cross_entropy(classifs[i], targets[i])
        for j in range(i + 1, nmb):
            penalty += AbstractMMD.mmd2(features[i], features[j])

    objective /= nmb
    if nmb > 1:
        penalty /= (nmb * (nmb - 1) / 2)

    if epoch < pretrain_epochs:
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'gate' not in name:
                l1_reg = l1_reg + torch.norm(param, p=1)
        loss = objective + (0.15 * penalty) + l1_coe * l1_reg
    else:
        loss = objective + (0.5 * penalty)
    
    return loss


def loss_DG_pruning(model, data_list, label_list, criterion, loss):
    output_list, loss_list = [], []

    for data, label in zip(data_list, label_list):
        output_list.append(model(data))
        loss_list.append(criterion(output_list[-1], label))

    # Calculate variance
    loss_variance = 0
    for loss_domain in loss_list:
        loss_variance += (loss_domain - loss) ** 2
    loss_variance = loss_variance/len(data_list)
    loss = loss + loss_variance

    return loss


def validate(model, test_loaders, test_loader_names, device, num_each_domain, test_envs, fraction):
    test_weights = [None for _ in test_loaders]
    results = {}

    evals = zip(test_loader_names, test_loaders, test_weights)
    for name, loader, weights in evals:
        acc, loss = misc.accuracy(model, loader, weights, device)
        results[name+'_acc'] = (acc, loss)

    results_keys = sorted(results.keys())
    # misc.print_row(results_keys, colwidth=12)
    # misc.print_row([results[key][0] for key in results_keys], colwidth=12)
    # print(results)

    all_envs = list(range(len(num_each_domain)))
    envs = [x for x in all_envs if x not in test_envs]
    test_env = test_envs[0]

    test_envs.sort(reverse=True)
    for i in test_envs:
        num_each_domain[i] = 0

    weights = []
    prec_val = 0
    val_loss = 0

    sum_data = sum(num_each_domain)
    for i in num_each_domain:
        weights.append(i/sum_data)

    # print('env'+str(test_env)+'_in_acc')
    top1_test_env = results['env'+str(test_env)+'_in_acc'][0]*(1-fraction) + results['env'+str(test_env)+'_out_acc'][0]*fraction
    for i in envs:
        prec_val += weights[i]*results['env'+str(i)+'_out_acc'][0]
        val_loss += weights[i]*results['env'+str(i)+'_out_acc'][1]
    # print(' * Prec_val@1 {top1:.3f}, Prec_test_envs@1 {top1_test_env:.3f}, Loss: {losses:.3f}'.format(top1=prec1*100, top1_test_env=top1_test_env*100, losses = val_loss) )
    print(' * Prec_val@1 {top1:.3f}, Loss: {losses:.3f}'.format(top1=prec_val*100, losses = val_loss))

    return (prec_val, top1_test_env), val_loss


def validate_and_save(test_loaders, test_loader_names, model, device, best_prec, epoch, model_save_path, num_each_domain, test_envs, fraction):
    prec, val_loss = validate(model, test_loaders, test_loader_names, device, num_each_domain, test_envs, fraction)
    is_best = prec[0] > best_prec[0]
    if is_best:
        best_prec = prec
    model_state_dict = model.state_dict()

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model_state_dict,
        'prec1': prec,
    }, is_best, filename=model_save_path)
    return prec, best_prec, val_loss
    