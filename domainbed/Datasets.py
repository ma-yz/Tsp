import os
import torch
from PIL import ImageFile
import numpy as np
from torchvision import transforms
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from prune import misc

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    # "SVIRO",
    # # WILDS datasets
    # "WILDSCamelyon",
    # "WILDSFMoW"
]


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
        

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


# Split each env into an 'in-split' and an 'out-split'. We'll train a model 
# on all in-splits except the test env, and evaluate on all splits.
def get_split_data(args, dataset):
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):

        out, in_ = split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.seed, env_i))

        in_splits.append((in_, None))
        out_splits.append((out, None))
    return in_splits, out_splits


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def get_loaders(bs, train_dataset, val_dataset, num_train_envs):
    # train_dataset = Multi_domain_data(in_splits, args.test_envs)
    # val_dataset = Multi_domain_data(out_splits, args.test_envs)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs*num_train_envs,
        shuffle=True,
        pin_memory=True,
    )

    # self.in_splits = in_splits
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=bs, 
        shuffle=False, 
        pin_memory=True
    )
    return train_loader, val_loader


class Multi_domain_data(torch.utils.data.Dataset):
    def __init__(self, multi_set, test_envs):
        super(Multi_domain_data, self).__init__()
        sum_samples = []
        self.sets = []
        for i, env in enumerate(multi_set):
            if i in test_envs:
                continue
            env, _ = env
            self.sets.append(env)
            if len(sum_samples) == 0:
                sum_samples.append(len(env))
            else:
                sum_samples.append(len(env)+sum_samples[-1])
        self.sum_samples = sum_samples


    def __getitem__(self, key):
        for i, _ in enumerate(self.sum_samples):
            if key < self.sum_samples[i]:
                if i == 0:
                    return self.sets[i][key]
                return self.sets[i][key-self.sum_samples[i-1]]
    
    def __len__(self):
        return self.sum_samples[-1]


    def t_env_splits(self):
        return self.sets


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        self.environments = sorted(environments)
        self.nums = []

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(self.environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)
            self.nums.append(len(env_dataset))
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, path, test_envs, augment):
        self.dir = path
        super().__init__(self.dir, test_envs, augment)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, path, test_envs, augment):
        self.dir = path
        super().__init__(self.dir, test_envs, augment)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, path, test_envs, augment):
        self.dir = path
        super().__init__(self.dir, test_envs, augment)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, path, test_envs, augment):
        self.dir = path
        super().__init__(self.dir, test_envs, augment)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, path, test_envs, augment):
        self.dir = path
        super().__init__(self.dir, test_envs, augment)

