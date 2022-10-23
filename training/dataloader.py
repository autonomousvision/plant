import logging
import numpy as np
import omegaconf
import diskcache
from beartype import beartype

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


@beartype
def get_dataloader(
    cfg:            omegaconf.dictconfig.DictConfig, 
    shared_dict:    diskcache.core.Cache                = None,
):

    if cfg.model.name == "PlanCNN":
        raise NotImplementedError("PlanCNN is not implemented yet")
        # from training.PlanCNN.dataset import generate_batch
        # from training.PlanCNN.dataset import PlanCNNDataset as Dataset
    elif cfg.model.name == "PlanT":
        from training.PlanT.dataset import generate_batch
        from training.PlanT.dataset import PlanTDataset as Dataset
    elif cfg.model.name == "PerceptionPlanT":
        raise NotImplementedError("PerceptionPlanT is not implemented yet")
        # from training.PerceptionPlanT.dataset import generate_batch
        # from training.PerceptionPlanT.dataset import PlanTDataset as Dataset
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    
    if cfg.benchmark == 'lav':
        # we train without T2 and T5

        logging.info(f'LAV training without T2 and T5')
        train_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split='train')
        val_set = Dataset(cfg.data_dir, cfg, shared_dict=shared_dict, split='val')
    
    elif cfg.benchmark == 'longest6':
        logging.info(f'Longest6 training with all towns')
        dataset = Dataset(
            cfg.data_dir, cfg, shared_dict=shared_dict, split="all"
        )

        # we validate on CARLA closed-loop, so we don't have a proper validation set here
        if cfg.trainset_size >= 1:
            train_set = dataset

            # use a very small subset of the trainset as validation set
            train_length = int(len(dataset) * 0.98)
            val_length = len(dataset) - train_length
            _, val_set = random_split(dataset, [train_length, val_length])

        else:
            train_length = int(len(dataset) * cfg.trainset_size)
            val_length = int(len(dataset) * 0.02)
            test_length = len(dataset) - train_length - val_length

            train_set, val_set, _ = random_split(
                dataset, [train_length, val_length, test_length]
            )
    else: 
        raise ValueError(f"Unknown benchmark: {cfg.benchmark}")
        
    logging.info(f'Train set size: {len(train_set)}')
    logging.info(f'Validation set size: {len(val_set)}')

    if cfg.custom_sampler and cfg.gpus > 1:
        # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
        # During training we cache the dataset on the fast storage of the local compute nodes.
        # For this we need to use a custom sampler.
        # TODO: hacky way to get RANK to work with multiple GPUs
        # file_name = logging.getLogger().root.handlers[1].baseFilename
        # try:
        #     rank = int(file_name[-5])
        # except:
        #     rank = 0

        # logging.info(f"Use custom sampler with rank {rank}")
        # logging.info(f"Rank: {rank}")

        # part_len = len(train_set) // cfg.gpus
        # indices = np.arange(
        #     rank * part_len, min(len(train_set), (1 + rank) * part_len), 1
        # )
        # sampler_train = SubsetRandomSampler(indices)

        train_loader = DataLoader(
            train_set,
            shuffle=True,
            # sampler=sampler_train,
            pin_memory=True,
            batch_size=cfg.model.training.batch_size,
            collate_fn=generate_batch,
            num_workers=cfg.model.training.num_workers,
        )

        # part_len2 = len(val_set) // cfg.gpus
        # indices2 = np.arange(
        #     rank * part_len2, min(len(val_set), (1 + rank) * part_len2), 1
        # )
        # sampler_val = SubsetRandomSampler(indices2)
        val_loader = DataLoader(
            val_set,
            shuffle=False,
            # sampler=sampler_val,
            pin_memory=True,
            batch_size=cfg.model.training.batch_size,
            collate_fn=generate_batch,
            num_workers=cfg.model.training.num_workers,
        )
    else:
        logging.info("Use default sampler")
        train_loader = DataLoader(
            train_set,
            shuffle=True,
            pin_memory=True,
            batch_size=cfg.model.training.batch_size,
            collate_fn=generate_batch,
            num_workers=cfg.model.training.num_workers,
        )

        val_loader = DataLoader(
            val_set,
            shuffle=False,
            pin_memory=True,
            batch_size=cfg.model.training.batch_size,
            collate_fn=generate_batch,
            num_workers=cfg.model.training.num_workers,
        )

    return train_loader, val_loader
