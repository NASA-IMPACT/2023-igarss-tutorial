# Copyright (c) OpenMMLab. All rights reserved.
import random
import os

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, build_optimizer, build_runner, get_dist_info
from mmcv.utils import build_from_cfg
from mmcv.parallel import collate
from functools import partial
from datetime import datetime

from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset, InMemoryDataset, process_and_augment_s1, \
    process_and_augment_s2, process_test_im, load_flood_train_data, load_flood_val_data
from mmseg.utils import find_latest_checkpoint, get_root_logger


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    rank, world_size = get_dist_info()

    init_fn = partial(
        worker_init_fn, num_workers=cfg.data.workers_per_gpu, rank=rank,
        seed=cfg.seed) if cfg.seed is not None else None

    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
	#s2==HLS
    cfg.data.train.img_dir = 'S2'
    if "S2" in cfg.data.train.img_dir:
	# update dataset path, scene, label, csv file
        dataset = load_flood_train_data(
            '/rtmp/akumar/HLS_merged/subsetted_USSE/scenes/',
            '/rtmp/akumar/HLS_merged/subsetted_USSE/masks/',
            '/rtmp/akumar/HLS_merged/subsetted_USSE/training_data.csv')
        logger.info("dataset {dataset}")
        dataset = InMemoryDataset(dataset, process_and_augment_s2)

    else:
        print("S1")
        dataset = load_flood_train_data(
            '/rtmp/cphillip/burn_scar_data/scenes/',
            '/rtmp/cphillip/burn_scar_data/masks/',
            '/rtmp/cphillip/burn_scar_data/master_list.csv'
        )
        print(dataset)
        dataset = InMemoryDataset(dataset, process_and_augment_s1)

    reduce_train_set = False
    if reduce_train_set:
        reduced_dataset = list()
        for index, data in enumerate(dataset):
            if index % 4 == 0:
                reduced_dataset.append(data)
            else:
                continue
        dataset = reduced_dataset

    dataset = [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    # if cfg.get('runner') is None:
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=cfg.epoch_config["epochs"])
    # warnings.warn(
    #     'config is now expected to have a `runner` section, '
    #     'please set `runner` in your config.', UserWarning)

    # Added custom logging
    save_path = '/rhome/sroy/hls_test4/hls-foundation/hls/downstream/finetune_ckpts/f/finetune_ckpts/'
    uuid_experiment = datetime.now()
    uuid_experiment = str(int(round(uuid_experiment.timestamp())))
    current_save_path = str(save_path) + 'Experiment_' + str(uuid_experiment) + "/"
    print(current_save_path)
    # Creating saving directory
    os.makedirs(current_save_path, exist_ok=True)
    print("dir created")

    cfg.work_dir = str(current_save_path + "/training/")

    cfg.checkpoint_config = dict(
        # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
        by_epoch=True,  # Whether count by epoch or not.
        interval=5,
        out_dir=current_save_path)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        if "S2" in cfg.data.train.img_dir:
	#change
            val_dataset = load_flood_val_data(
                '/rtmp/akumar/HLS_merged/subsetted_USSE/validation/',
                '/rtmp/akumar/HLS_merged/subsetted_USSE/validation/',
                '/rtmp/akumar/HLS_merged/subsetted_USSE/validation_data.csv'
            )
            val_dataset = InMemoryDataset(val_dataset, process_and_augment_s2)

        else:
            val_dataset = load_flood_val_data(
                '/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled/S1Hand/',
                '/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled/LabelHand/',
                '/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_valid_data.csv'
            )
            val_dataset = InMemoryDataset(val_dataset, process_and_augment_s1)

        val_dataset = [val_dataset]
        val_dataloader = build_dataloader(
            val_dataset[0],
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            dist=distributed,
            seed=cfg.seed,
            drop_last=True)
        data_loaders.append(val_dataloader)

        # val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # val_dataloader = build_dataloader(
        #     val_dataset,
        #     samples_per_gpu=1,
        #     workers_per_gpu=cfg.data.workers_per_gpu,
        #     dist=distributed,
        #     shuffle=False)

        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        print("register the eval hook for validation data")
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
