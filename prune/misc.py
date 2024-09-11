import hashlib
import sys
from shutil import copyfile
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prune.group_lasso_optimizer import group_lasso_decay
from prune.pruning_engine import pytorch_pruning, prepare_pruning_list
from prune.utils import AverageMeter, connect_gates_with_parameters_for_flops
from collections import Counter


def load_engine(args, model, pruning_settings, weight_decay, output_sizes,  log_folder, train_writer):
    parameters_for_update = []
    parameters_for_update_named = []
    for name, m in model.named_parameters():
        if "gate" not in name:
            parameters_for_update.append(m)
            parameters_for_update_named.append((name, m))
        # else:
        #     print("skipping parameter", name, "shape:", m.shape)
    total_size_params = sum([np.prod(par.shape) for par in parameters_for_update])
    print("Total number of parameters, w/o usage of bn consts: ", total_size_params)

    optimizer = optim.SGD(parameters_for_update, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)

    # helping optimizer to implement group lasso (with very small weight that doesn't affect training)
    # will be used to calculate number of remaining flops and parameters in the network
    group_wd_optimizer = group_lasso_decay(parameters_for_update, group_lasso_weight=args.group_wd_coeff, named_parameters=parameters_for_update_named, output_sizes=output_sizes)
    pruning_parameters_list = prepare_pruning_list(pruning_settings, model, model_name=args.model,
                                                        pruning_mask_from=args.pruning_mask_from, name=args.save_dir)
    # print("Total pruning layers:", len(pruning_parameters_list))

    pruning_engine = pytorch_pruning(pruning_parameters_list, pruning_settings=pruning_settings, log_folder=log_folder)
    pruning_engine.connect_tensorboard(train_writer)
    pruning_engine.dataset = args.dataset
    pruning_engine.model = args.model
    pruning_engine.pruning_mask_from = args.pruning_mask_from
    pruning_engine.load_mask()
    gates_to_params = connect_gates_with_parameters_for_flops(args.model, parameters_for_update_named)
    pruning_engine.gates_to_params = gates_to_params
    return pruning_engine, group_wd_optimizer, optimizer


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.named_parameters = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.named_parameters[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.named_parameters[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    print(args_str)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)
        
        
def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            length = len(x)
            x = x.to(device)
            y = y.to(device)
            p = network(x)
            loss = criterion(p, y)
            losses.update(loss.item(), length)
            if weights is None:
                batch_weights = torch.ones(length)
            else:
                batch_weights = weights[weights_offset : weights_offset + length]
                weights_offset += length
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    #  network.train()

    return correct / total, losses.avg

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
