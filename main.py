from __future__ import print_function
import argparse

import os, sys
import time
import json
import torch.nn as nn
import numpy as np
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import prune.swa_utils as swa_utils
import prune.swad as swad_module
from prune.utils import adjust_learning_rate, load_model_pytorch, get_conv_sizes
from tensorboardX import SummaryWriter
from logger import Logger
from models.preact_resnet import *
from models.resnet import resnet18, resnet34, resnet50, resnet101
from prune.utils import str2bool
from prune.misc import load_engine
from prune.algorithms import train, validate_and_save, validate
from prune.pruning_engine import PruningConfigReader

from domainbed import Datasets
from domainbed.Datasets import Multi_domain_data
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.Datasets import InfiniteDataLoader

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def main():
    parser = argparse.ArgumentParser(description='TSP based on PyTorch')
    parser.add_argument('--data', metavar='DIR', default='/imagenet', help='path to imagenet dataset')
    parser.add_argument('--dataset', default="PACS", type=str)
    parser.add_argument('--algorithm', default="TSP", type=str, help='choose in ["TSP", "SWAD", "DG_pruning", "baseline"]')
    parser.add_argument('--model', default="resnet50", type=str,
                        choices=["resnet18", "resnet50", "resnet34", "resnet101"])
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--get_flops', default=False,  type=str2bool, nargs='?',
                        help='add hooks to compute flops')
    parser.add_argument('--mgpu', default=False,  type=str2bool, nargs='?',
                        help='use data paralization via multiple GPUs')
    parser.add_argument('--tensorboard', type=str2bool, nargs='?',
                        help='Log progress to TensorBoard')
    parser.add_argument('--save_dir', default='test', type=str,
                        help='experiment name(folder) to store model and logs')

    # ============================PRUNING
    parser.add_argument('--pruning', default=True, type=str2bool, nargs='?',
                        help='enable or not pruning, def False')
    parser.add_argument('--pruning_config', default=None, type=str,
                        help='path to pruning configuration file, will overwrite all pruning parameters in arguments')
    parser.add_argument('--group_wd_coeff', type=float, default=1e-8,
                        help='group weight decay')
    parser.add_argument('--load_model', default='', type=str,
                        help='path to model weights')
    parser.add_argument('--fixed_network', default=False,  type=str2bool, nargs='?',
                        help='fix network for oracle or criteria computation')
    parser.add_argument('--pruning_mask_from', default='', type=str,
                        help='path to mask file precomputed')
    parser.add_argument('--compute_flops', default=True,  type=str2bool, nargs='?',
                        help='if True, will run dummy inference of batch 1 before training to get conv sizes')
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--checkpoint_freq', default=400, type=int)
    parser.add_argument('--freeze_bn', default=True, type=str2bool)


    args = parser.parse_args()
    with open(args.pruning_config, "r") as f:
        config = json.load(f)
    for key in config:
        parser.add_argument('--'+key, default=config[key])
        # args.key = config[key]
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    global group_wd_optimizer
    prune_done = False
    global_iteration = 0
    best_prec = (0, 0)

    # dataset loading section
    dataset = vars(Datasets)[args.dataset](args.data,
            args.test_envs, augment=True)
    num_each_domain = dataset.nums
    num_train_envs = len(dataset.environments) - len(args.test_envs)
    num_classes = dataset.num_classes
    in_splits, out_splits = Datasets.get_split_data(args, dataset)
    train_dataset, val_dataset = Multi_domain_data(in_splits, args.test_envs), Multi_domain_data(out_splits, args.test_envs)
    train_loader, _ = Datasets.get_loaders(args.batch_size, train_dataset, val_dataset, num_train_envs)

    epoch_ites = len(train_loader)
    in_train_splits = train_dataset.t_env_splits()

    # test_loaders
    test_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)]

    test_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    test_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    # load model
    try:
        model = eval(args.model)(args, num_classes=num_classes)
    except:
        raise Exception("The model is not supported!")
    # print("model is defined")

    if use_cuda and not args.mgpu:
        model = model.to(device)
    elif args.mgpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    print("model is set to device: use_cuda {}, args.mgpu {}".format(use_cuda, args.mgpu))

    kwargs = {'num_workers': args.num_workers} if use_cuda else {}
    data_iterators = [InfiniteDataLoader(x, batch_size=args.batch_size, shuffle=True, **kwargs) for x in in_train_splits]
    output_sizes = get_conv_sizes(args, model)
    weight_decay = args.wd
    if args.fixed_network:
        weight_decay = 0.0
    cudnn.benchmark = True

    # define objective
    criterion = nn.CrossEntropyLoss()

    # logging part
    log_save_folder = "%s"%args.save_dir
    if not os.path.exists(log_save_folder):
        os.makedirs(log_save_folder)
    if not os.path.exists("%s/models" % (log_save_folder)):
        os.makedirs("%s/models" % (log_save_folder))
    train_writer = None
    if args.tensorboard:
        try:
            # tensorboardX v1.6
            train_writer = SummaryWriter(log_dir="%s"%(log_save_folder))
        except:
            # tensorboardX v1.7
            train_writer = SummaryWriter(logdir="%s"%(log_save_folder))
    time_point = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    textfile = "%s/log_%s.txt" % (log_save_folder, time_point)
    model_save_path = "%s/models/checkpoint.weights"%(log_save_folder)
    folder_to_write = "%s"%log_save_folder+"/"
    log_folder = folder_to_write
    stdout = Logger(textfile)
    sys.stdout = stdout
    print(" ".join(sys.argv))

    # initializing parameters for pruning
    pruning_settings = dict()
    prune_done = False
    if not (args.pruning_config is None):
        pruning_settings_reader = PruningConfigReader()
        pruning_settings_reader.read_config(args.pruning_config)
        pruning_settings = pruning_settings_reader.get_parameters()

    # overwrite parameters from config file with those from command line
    # needs manual entry here
    has_attribute = lambda x: any([x in a for a in sys.argv])
    if has_attribute('pruning-momentum'):
        pruning_settings['pruning_momentum'] = vars(args)['pruning_momentum']
    pruning_engine, group_wd_optimizer, optimizer = load_engine(args, model, pruning_settings, weight_decay, output_sizes, log_folder, train_writer)

    ###=======================end for pruning
    # loading model file
    if args.load_model:
        start_epoch, best_prec = load_model_pytorch(args, model, args.load_model, args.model)

    swad_algorithm, swad_best_pre = None, (0, 0)
    if args.algorithm == "SWAD":
        swad_algorithm = swa_utils.AveragedModel(model)
        swad_cls = getattr(swad_module, args.swad_method) 
        swad = swad_cls(args.n_converge, args.n_tolerance, args.tolerance_ratio, validate=validate)
        swad_save_path = "%s/models/swad_checkpoint.weights"%(args.save_dir)
        swad.scope = 0

    results, _ = validate(model, test_loaders, test_loader_names, device, num_each_domain, args.test_envs, args.holdout_fraction)
    for epoch in range(start_epoch, args.epochs):
        if args.algorithm == "TSP" and epoch==args.pretrain_epochs:
            # remove dropout layers
            load_from = model.state_dict()
            model_nodrop = eval(args.model)(args, num_classes=num_classes, drop=False)
            model = model_nodrop

            output_sizes = get_conv_sizes(args, model)
            if use_cuda and not args.mgpu:
                model = model.to(device)
            elif args.mgpu:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.to(device)
            model.load_state_dict(load_from, strict=True)
            print("Dropout layers have been removed from model")
            pruning_engine, group_wd_optimizer, optimizer = load_engine(args, model, pruning_settings, weight_decay, output_sizes, log_folder, train_writer)

        adjust_learning_rate(args, optimizer, epoch, args.zero_lr_for_epochs, train_writer)
        global_iteration = train(args, 
                            model, 
                            device, 
                            epoch_ites, 
                            optimizer, 
                            epoch, 
                            criterion, 
                            data_iterators, 
                            group_wd_optimizer, 
                            train_writer, 
                            pruning_engine, 
                            swad_algorithm, 
                            prune_done,
                            global_iteration)

        if args.pruning and pruning_engine.method == 50:
            continue
        if prune_done:
            prec, best_prec, val_loss = validate_and_save(test_loaders, test_loader_names, model, device, best_prec, epoch, model_save_path, num_each_domain, args.test_envs, args.holdout_fraction)
        else:
            prec, _, val_loss = validate_and_save(test_loaders, test_loader_names, model, device, best_prec, epoch, model_save_path, num_each_domain, args.test_envs, args.holdout_fraction)

        if pruning_engine.prune_neurons_max == pruning_engine.pruned_neurons:
            prune_done = True
        # evaluate on validation set
        if args.algorithm=="SWAD" and best_prec!=(0, 0):
            if swad.scope < args.range:
                swad.update_and_evaluate(swad_algorithm, prec[0], val_loss)
                swad.scope += 1
                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    train_writer.add_scalar('SWAD valley is dead -> early stop !', 1, global_iteration-1)
            elif swad.scope == args.range:
                print("Evaluating SWAD model:")
                swad_algo = swad.get_final_model()
                _, swad_best_pre, _ = validate_and_save(test_loaders, test_loader_names, swad_algo, device, swad_best_pre, epoch, swad_save_path, num_each_domain, args.test_envs, args.holdout_fraction)
                print("End evaluation of SWAD model.")
                swad = swad_cls(args.n_converge, args.n_tolerance, args.tolerance_ratio, validate=validate)
                swad.scope = 0

            # reset
            # print("here to check:", swad_algorithm.che)
            swad_algorithm = swa_utils.AveragedModel(model)

    if args.algorithm == "SWAD":
        print("The best checkpoint of SWAD:")
        print(' * Prec_val@1 {top1:.3f}, Prec_test_envs@1 {top1_test_env:.3f}, Loss: {losses:.3f}'.format(top1=swad_best_pre[0]*100, top1_test_env=swad_best_pre[1]*100, losses = val_loss) )

    print("The best checkpoint:")
    print(' * Prec_val@1 {top1:.3f}, Prec_test_envs@1 {top1_test_env:.3f}, Loss: {losses:.3f}'.format(top1=best_prec[0]*100, top1_test_env=best_prec[1]*100, losses = val_loss) )
   

if __name__ == '__main__':
    main()
