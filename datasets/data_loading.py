import os
import logging
import random
import numpy as np
import webdataset as wds

import torch
import torchvision
import torchvision.transforms as transforms

from conf import complete_data_dir_path
from datasets.corruptions_datasets import create_cifarc_dataset, create_imagenetc_dataset


logger = logging.getLogger(__name__)


def identity(x):
    return x


def get_transform(dataset_name: str, preprocess=None):
    """
    Get the transformation pipeline
    Note that the data normalization is done within the model
    Input:
        dataset_name: Name of the dataset
        adaptation: Name of the adaptation method
    Returns:
        transforms: The data pre-processing (and augmentation)
    """
    if dataset_name in ["cifar10", "cifar100"]:
        transform = transforms.Compose([transforms.ToTensor()])
    elif dataset_name in ["cifar10_c", "cifar100_c"]:
        transform = None
    elif dataset_name in ["imagenet_c", "ccc"]:
        # note that ImageNet-C and CCC are already resized and centre cropped (to size 224)
        # if use resnet50, there is a pre-normalizaion layer
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        if preprocess:
            # set transform to the corresponding input transformation of the restored model
            transform = preprocess
        else:
            # use classical ImageNet transformation procedure
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])
    return transform


def get_test_loader(setting: str, dataset_name : str, data_root_dir: str, domain_name: str, domain_names_all: list, 
                    severity: int, num_examples: int, rng_seed: int, batch_size: int = 128, shuffle: bool = False, 
                    workers: int = 4, preprocess=None):
    """
    Create the test data loader
    Input:
        setting: Name of the considered setting
        dataset_name: Name of the dataset
        data_root_dir: Path of the data root directory
        domain_name: Name of the current domain
        domain_names_all: List containing all domains
        severity: Severity level in case of corrupted data
        num_examples: Number of test samples for the current domain
        rng_seed: A seed number
        batch_size: The number of samples to process in each iteration
        shuffle: Whether to shuffle the data. Will destroy pre-defined settings
        workers: Number of workers used for data loading
    Returns:
        test_loader: The test data loader
    """

    data_dir = complete_data_dir_path(data_root_dir, dataset_name)
    transform = get_transform(dataset_name, preprocess)

    # create the test dataset
    if domain_name == "none":
        test_dataset, _ = get_source_loader(dataset_name,
                                            data_root_dir, batch_size,
                                            train_split=False, workers=workers)
    else:
        if dataset_name in ["cifar10_c", "cifar100_c"]:
            test_dataset = create_cifarc_dataset(dataset_name=dataset_name,
                                                 severity=severity,
                                                 data_dir=data_dir,
                                                 corruption=domain_name,
                                                 corruptions_seq=domain_names_all,
                                                 transform=transform,
                                                 setting=setting)
            
            # randomly subsample the dataset if num_examples is specified
            if num_examples != -1:
                num_samples_orig = len(test_dataset)
                # logger.info(f"Changing the number of test samples from {num_samples_orig} to {num_examples}...")
                test_dataset.samples = random.sample(test_dataset.samples, k=min(num_examples, num_samples_orig))

        elif dataset_name == "imagenet_c":
            test_dataset = create_imagenetc_dataset(n_examples=num_examples,
                                                    severity=severity,
                                                    data_dir=data_dir,
                                                    corruption=domain_name,
                                                    corruptions_seq=domain_names_all,
                                                    transform=transform,
                                                    setting=setting)

        elif dataset_name == "ccc":
            logger.info(f"Using the following data transformation:\n{transform}")
            workers = 1
            url = os.path.join(data_root_dir, "CCC", domain_name,"serial_{00000..99999}.tar") # Uncoment this to use a local copy of CCC
            # domain_name = "baseline_20_transition+speed_1000_seed_44" # choose from: baseline_<0/20/40>_transition+speed_<1000/2000/5000>_seed_<43/44/45>
            # url = f'https://mlcloud.uni-tuebingen.de:7443/datasets/CCC/{domain_name}/serial_{{00000..99999}}.tar'

            test_dataset = (wds.WebDataset(url)
                    .decode("pil")
                    .to_tuple("input.jpg", "output.cls")
                    .map_tuple(transform, identity)
            )

        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    try:
        # shuffle the test sequence; deterministic behavior for a fixed random seed
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        random.shuffle(test_dataset.samples)

        if "continual_cdc" in setting:
            new_sample_sequence = []
            remaining_samples = {domain: [x for x in test_dataset.samples if x[2] == domain] for domain in domain_names_all}
            remaining_batches = {domain : len(samples)//batch_size + 1 for domain, samples in remaining_samples.items()}

            while remaining_samples:
                selected_domains = np.random.choice(list(remaining_samples.keys()), 1)[0]
                num_selected_batches = np.random.choice(list(range(1, remaining_batches[selected_domains]+1)))
                num_selected_samples = num_selected_batches*batch_size
                new_sample_sequence += remaining_samples[selected_domains][:num_selected_samples]
                remaining_samples[selected_domains] = remaining_samples[selected_domains][num_selected_samples:]
                remaining_batches[selected_domains] -= num_selected_batches
                if len(remaining_samples[selected_domains]) == 0:
                    del remaining_samples[selected_domains]
                    del remaining_batches[selected_domains]

            test_dataset.samples = new_sample_sequence

    except AttributeError:
        logger.warning("Attribute 'samples' is missing. Continuing without shuffling, sorting or subsampling the files...")

    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def get_source_loader(dataset_name: str, data_root_dir: str, batch_size: int, train_split: bool = True, 
                      num_samples: int = -1, percentage: float = 1.0, workers: int = 4, preprocess=None):
    """
    Create the source data loader
    Input:
        dataset_name: Name of the dataset
        data_root_dir: Path of the data root directory
        batch_size: The number of samples to process in each iteration
        train_split: Whether to use the training or validation split
        num_samples: Number of source samples used during training
        percentage: (0, 1] Percentage of source samples used during training
        workers: Number of workers used for data loading
    Returns:
        source_dataset: The source dataset
        source_loader: The source data loader
    """

    # create the correct source dataset name
    src_dataset_name = dataset_name.split("_")[0] if dataset_name != "ccc" else "imagenet"

    # complete the data root path to the full dataset path
    data_dir = complete_data_dir_path(data_root_dir, dataset_name=src_dataset_name)

    # get the data transformation
    transform = get_transform(src_dataset_name, preprocess)

    # create the source dataset
    if dataset_name in ["cifar10", "cifar10_c"]:
        source_dataset = torchvision.datasets.CIFAR10(root=data_root_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name in ["cifar100", "cifar100_c"]:
        source_dataset = torchvision.datasets.CIFAR100(root=data_root_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
    elif dataset_name in ["imagenet", "imagenet_c", "imagenet_k", "ccc"]:
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                       split=split,
                                                       transform=transform)
    else:
        raise ValueError("Dataset not supported.")

    if percentage < 1.0 or num_samples >= 0:    # reduce the number of source samples
        assert percentage > 0.0, "The percentage of source samples has to be in range 0.0 < percentage <= 1.0"
        assert num_samples > 0, "The number of source samples has to be at least 1"
        if src_dataset_name in ["cifar10", "cifar100"]:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples > 0 else int(np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples > 0 else int(np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)

        logger.info(f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")

    # create the source data loader
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                drop_last=False)
    logger.info(f"Number of images and batches in source loader: #img = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_dataset, source_loader
