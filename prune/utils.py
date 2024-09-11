import torch
from collections import OrderedDict
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    try:
        torch.save(state, filename)
        if is_best:
            torch.save(state, filename.replace("checkpoint", "best_model"))
    except:
        print("didn't save checkpoint file")


def adjust_learning_rate(args, optimizer, epoch, zero_lr_for_epochs, train_writer):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    if args.algorithm == "TSP" and epoch >= args.pretrain_epochs:
        epo_t = epoch - args.pretrain_epochs
    else:
        epo_t = epoch

    lr = args.lr * (args.lr_decay_scalar ** (epo_t // args.lr_decay_every)) # * (0.1 ** (epoch // (2*args.lr_decay_every)))

    if zero_lr_for_epochs > -1:
        if epoch <= zero_lr_for_epochs:
            lr = 0.0
    # log to TensorBoard
    if args.tensorboard:
        train_writer.add_scalar('learning_rate', lr, epoch)
    print("learning rate adjusted:", lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_fixed(args, optimizer, epoch, zero_lr_for_epochs, train_writer):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (args.lr_decay_scalar ** (epoch // args.lr_decay_every))
    if zero_lr_for_epochs > -1:
        if epoch <= zero_lr_for_epochs:
            lr = 0.0
    # log to TensorBoard
    if epoch == args.lr_decay_every:
        lr_scale = args.lr_decay_scalar
        lr = args.lr * (args.lr_decay_scalar ** (epoch // args.lr_decay_every))
        if args.tensorboard:
            train_writer.add_scalar('learning_rate', lr, epoch)
        print("learning rate adjusted:", lr, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*lr_scale


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_model_pytorch(args, model, load_model, model_name):
    # print(model = model.)
    model.eval()
    
    print("=> loading checkpoint '{}'".format(load_model))
    checkpoint = torch.load(load_model)
    best_prec1 = (0, 0)

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
        keys = list(load_from.keys())
        para_num = {'resnet18':122, 'resnet34':218, 'resnet50':320, 'resnet101':626}
        if "update_count" not in keys[0]:
            load_from = OrderedDict([(k, load_from[k]) for k in keys[0:0+para_num[model_name]]])
        else:
            load_from = OrderedDict([(k, load_from[k]) for k in keys[1:1+para_num[model_name]]])
        load_from = OrderedDict([(k.replace("featurizer.network.", "module.Featurizer."), v) for k, v in load_from.items()])
        load_from = OrderedDict([(k.replace("classifier.", "module.Classifier."), v) for k, v in load_from.items()])
    else:
        raise Exception("The state_dict is not found in checkpoint!")

    for key, item in model.state_dict().items():
        # if we add gate that is not in the saved file
        if key not in load_from and 'gate' in key:
            load_from[key] = item

    model.load_state_dict(load_from, strict=True)
    epoch_from = 0
    if 'epoch' in checkpoint.keys():
        epoch_from = checkpoint['epoch']
    # print("=> loaded checkpoint '{}'".format(load_model))
    return epoch_from, best_prec1


def dynamic_network_change_local(model):
    '''
    Methods attempts to modify network in place by removing pruned filters.
    Works with ResNet101 for now only
    :param model: reference to torch model to be modified
    :return:
    '''
    # change network dynamically given a pruning mask

    # step 1: model adjustment
    # lets go layer by layer and get the mask if we have parameter in pruning settings:

    pruning_maks_input = None
    prev_model1 = None
    prev_model2 = None
    prev_model3 = None

    pruning_mask_indexes = None
    gate_track = -1

    skip_connections = list()

    current_skip = 0
    DO_SKIP = True

    gate_size = -1
    current_skip_mask_size = -1

    for module_indx, m in enumerate(model.modules()):
        pruning_mask_indexes = None
        if not hasattr(m, "do_not_update"):
            if isinstance(m, torch.nn.Conv2d):
                print("interm layer", gate_track, module_indx)
                print(m)
                if 1:
                    if pruning_maks_input is not None:
                        print("fixing interm layer", gate_track, module_indx)

                        m.weight.data = m.weight.data[:, pruning_maks_input]

                        pruning_maks_input = None
                        print("weight size now", m.weight.data.shape)
                        m.in_channels = m.weight.data.shape[1]
                        m.out_channels = m.weight.data.shape[0]

                        if DO_SKIP:
                            print("doing skip connection")
                            if m.weight.data.shape[0] == current_skip_mask_size:
                                m.weight.data = m.weight.data[current_skip_mask]
                                print("weight size after skip", m.weight.data.shape)

                    if module_indx in [23, 63, 115, 395]:
                        if DO_SKIP:
                            print("fixing interm skip")
                            m.weight.data = m.weight.data[:, prev_skip_mask]

                            print("weight size now", m.weight.data.shape)
                            m.in_channels = m.weight.data.shape[1]
                            m.out_channels = m.weight.data.shape[0]

                            if DO_SKIP:
                                print("doing skip connection")
                                if m.weight.data.shape[0] == current_skip_mask_size:
                                    m.weight.data = m.weight.data[current_skip_mask]
                                    print("weight size after skip", m.weight.data.shape)

            if isinstance(m, torch.nn.BatchNorm2d):
                print("interm layer BN: ", gate_track, module_indx)
                print(m)
                if DO_SKIP:
                    print("doing skip connection")
                    if m.weight.data.shape[0] == current_skip_mask_size:
                        m.weight.data = m.weight.data[current_skip_mask]
                        print("weight size after skip", m.weight.data.shape)

                        # m.weight.data = m.weight.data[current_skip_mask]
                        m.bias.data = m.bias.data[current_skip_mask]
                        m.running_mean.data = m.running_mean.data[current_skip_mask]
                        m.running_var.data = m.running_var.data[current_skip_mask]
        else:
            # keeping track of gates:
            gate_track += 1

            then_pass = False
            if gate_track < 4:
                # skipping skip connections
                then_pass = True
                skip_connections.append(m.weight)
                current_skip = -1

            if not then_pass:
                pruning_mask = m.weight
                if gate_size!=m.weight.shape[0]:
                    current_skip += 1
                    current_skip_mask_size = skip_connections[current_skip].data.shape[0]

                    if skip_connections[current_skip].data.shape[0] != 2048:
                        current_skip_mask = skip_connections[current_skip].data.nonzero().view(-1)
                    else:
                        current_skip_mask = (skip_connections[current_skip].data + 1.0).nonzero().view(-1)
                    prev_skip_mask_size = 64
                    prev_skip_mask = range(64)
                    if current_skip > 0:
                        prev_skip_mask_size = skip_connections[current_skip - 1].data.shape[0]
                        prev_skip_mask = skip_connections[current_skip - 1].data.nonzero().view(-1)

                gate_size = m.weight.shape[0]

                if 1:
                    print("fixing layer", gate_track, module_indx)
                    print(m)
                    print(pruning_mask)

                    if 1.0 in pruning_mask:
                        pruning_mask_indexes = pruning_mask.nonzero().view(-1)
                    else:
                        pruning_mask_indexes = []
                    m.weight.data = m.weight.data[pruning_mask_indexes]
                    for prev_model in [prev_model1, prev_model2, prev_model3]:
                        if isinstance(prev_model, torch.nn.Conv2d):
                            print("prev fixing layer", prev_model, gate_track, module_indx)
                            prev_model.weight.data = prev_model.weight.data[pruning_mask_indexes]
                            print("weight size", prev_model.weight.data.shape)

                            if DO_SKIP:
                                print("doing skip connection")

                                if prev_model.weight.data.shape[1] == current_skip_mask_size:
                                    prev_model.weight.data = prev_model.weight.data[:, current_skip_mask]
                                    print("weight size", prev_model.weight.data.shape)

                                if module_indx in [53, 105, 385]:  # add one more layer for this transition
                                    print("doing skip connection")

                                    if prev_model.weight.data.shape[1] == prev_skip_mask_size:
                                        prev_model.weight.data = prev_model.weight.data[:, prev_skip_mask]
                                        print("weight size", prev_model.weight.data.shape)

                        if isinstance(prev_model, torch.nn.BatchNorm2d):
                            print("prev fixing layer", prev_model, gate_track, module_indx)
                            prev_model.weight.data = prev_model.weight.data[pruning_mask_indexes]
                            prev_model.bias.data = prev_model.bias.data[pruning_mask_indexes]
                            prev_model.running_mean.data = prev_model.running_mean.data[pruning_mask_indexes]
                            prev_model.running_var.data = prev_model.running_var.data[pruning_mask_indexes]

                pruning_maks_input = pruning_mask_indexes

        prev_model3 = prev_model2
        prev_model2 = prev_model1
        prev_model1 = m

    if DO_SKIP:
        # fix gate layers
        gate_track = 0

        for module_indx, m in enumerate(model.modules()):
            if hasattr(m, "do_not_update"):
                gate_track += 1
                if gate_track < 4:
                    if m.weight.shape[0] < 2048:
                        m.weight.data = m.weight.data[m.weight.nonzero().view(-1)]

    print("printing conv layers")
    for module_indx, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.Conv2d):
            print(module_indx, "->", m.weight.data.shape)

    print("printing bn layers")
    for module_indx, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            print(module_indx, "->", m.weight.data.shape)

    print("printing gate layers")
    for module_indx, m in enumerate(model.modules()):
        if hasattr(m, "do_not_update"):
            print(module_indx, "->", m.weight.data.shape, m.size_mask)



def add_hook_for_flops(args, model):
    # add output dims for FLOPs computation
    if 1:
        for module_indx, m in enumerate(model.modules()):
            if isinstance(m, torch.nn.Conv2d):
                def forward_hook(self, input, output):
                    self.weight.output_dims = output.shape

                m.register_forward_hook(forward_hook)


def get_conv_sizes(args, model):
    output_sizes = None
    if args.compute_flops:
        # add hooks to compute dimensions of the output tensors for conv layers
        add_hook_for_flops(args, model)
        if 1:
            dummy_input = torch.rand(1, 3, 224, 224)
            # run inference
            with torch.no_grad():
                model(dummy_input)
            # store flops
            output_sizes = list()
            for param in model.parameters():
                if hasattr(param, 'output_dims'):
                    output_dims = param.output_dims
                    output_sizes.append(output_dims)

    return output_sizes


def connect_gates_with_parameters_for_flops(model_name, named_parameters):
    '''
    Function creates a mapping between gates and parameter index to map flops
    :return:
    returns a list with mapping, each element is a gate id, entries are corresponding parameters
    '''
    if "resnet" not in model_name:
        print("connect_gates_with_parameters_for_flops only supports resnet for now")
        return -1

    # for skip connections, first 4 for them:
    gate_to_param_map = [list() for _ in range(4)]

    for param_id, (name, param) in enumerate(named_parameters):
        if "layer" not in name:
            # skip because we don't prune the first layer
            continue

        if ("conv1" in name) or ("conv2" in name):
            gate_to_param_map.append(param_id)

        if "conv3" in name:
            # third convolution contributes to skip connection only
            skip_block_id = int(name[name.find("layer")+len("layer")])
            gate_to_param_map[skip_block_id-1].append(param_id)

    return gate_to_param_map