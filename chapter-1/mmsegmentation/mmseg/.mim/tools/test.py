# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import os.path as osp
import pandas as pd
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes
from mmseg.datasets import build_dataloader, build_dataset, InMemoryDataset, load_flood_test_data, process_test_im
from metrics import metrics
from datetime import datetime
import os
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
             'not be supported in version v0.22.0. Override some settings in the '
             'used config, the key-value pair in xxx=yyy format will be merged '
             'into config file. If the value to be overwritten is a list, it '
             'should be like key="[a,b]" or key=a,b It also allows nested '
             'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
             'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False)

    # prepare data loaders

    eval_paths = list()
    path_to_test = '/rtmp/akumar/HLS_merged/subsetted_USSE/validation_data.csv'
    # path_to_hold_out = '/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_bolivia_data_S2.csv'
    eval_paths.append(path_to_test)
    # eval_paths.append(path_to_hold_out)
    for path_to_split in eval_paths:
        if "bolivia" in path_to_split:
            hold_out_eval = True
        else:
            hold_out_eval = False

        dataset = load_flood_test_data(
            '/rtmp/akumar/HLS_merged/subsetted_USSE/validation/',
            '/rtmp/akumar/HLS_merged/subsetted_USSE/validation/',
            path_to_split
        )
        print("len(dataset)", len(dataset))
        dataset = InMemoryDataset(dataset, process_test_im)
        loader_cfg = dict(
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.data.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader'
            ]
        })
        test_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('test_dataloader', {})
        }
        # build the dataloader
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        print("len(data_loader)", len(data_loader))

        # build the model and load checkpoint
        ids_pixel_wise_recalc = [
            # 1678283804,  # MM
            # 1678352378,  # MM
            # 1678360472,  # MM
            # 1678371726,  # MM
            # 1678439366,  # MM
  
            # 1678450904,  # JJ
            # 1678605282,  # JJ
            # 1678604727,  # JJ
            # 1678604822,  # JJ
            # 1678647778,  # JJ
            # 1678647939,  # JJ
            # 1678656380,  # JJ
            # 1678656518,  # JJ
            1680870249 #SR
        ]

        for uuid_experiment in ids_pixel_wise_recalc:
            test_path = '/rhome/sroy/hls_test4/hls-foundation/hls/downstream/finetune_ckpts/f/finetune_ckpts/'
            test_path = str(test_path) + 'Experiment_' + str(uuid_experiment)
            # Creating saving directory
            assert os.path.exists(test_path), "Directory for testing seems to be not available."
            os.makedirs(test_path + "/inferences", exist_ok=True)

            n_epoch = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            for epoch in n_epoch:
                print("epoch", epoch)
                cfg.model.train_cfg = None
                model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
                fp16_cfg = cfg.get('fp16', None)
                if fp16_cfg is not None:
                    wrap_fp16_model(model)
                checkpoint = load_checkpoint(model, "{0}/training/epoch_{1}.pth".format(test_path, epoch),
                                             map_location='cpu')
                if 'CLASSES' in checkpoint.get('meta', {}):
                    model.CLASSES = checkpoint['meta']['CLASSES']
                else:
                    print('"CLASSES" not found in meta, use dataset.CLASSES instead')
                    model.CLASSES = dataset.CLASSES
                if 'PALETTE' in checkpoint.get('meta', {}):
                    model.PALETTE = checkpoint['meta']['PALETTE']
                else:
                    print('"PALETTE" not found in meta, use dataset.PALETTE instead')
                    model.PALETTE = dataset.PALETTE

                # clean gpu memory when starting a new evaluation.
                torch.cuda.empty_cache()
                eval_kwargs = {} if args.eval_options is None else args.eval_options

                # Deprecated
                efficient_test = eval_kwargs.get('efficient_test', False)
                if efficient_test:
                    warnings.warn(
                        '``efficient_test=True`` does not have effect in tools/test.py, '
                        'the evaluation and format results are CPU memory efficient by '
                        'default')

                eval_on_format_results = (
                        args.eval is not None and 'cityscapes' in args.eval)
                if eval_on_format_results:
                    assert len(args.eval) == 1, 'eval on format results is not ' \
                                                'applicable for metrics other than ' \
                                                'cityscapes'
                if args.format_only or eval_on_format_results:
                    if 'imgfile_prefix' in eval_kwargs:
                        tmpdir = eval_kwargs['imgfile_prefix']
                    else:
                        tmpdir = '.format_cityscapes'
                        eval_kwargs.setdefault('imgfile_prefix', tmpdir)
                    mmcv.mkdir_or_exist(tmpdir)
                else:
                    tmpdir = None

                if not distributed:
                    warnings.warn(
                        'SyncBN is only supported with DDP. To be compatible with DP, '
                        'we convert SyncBN to BN. Please use dist_train.sh which can '
                        'avoid this error.')
                    if not torch.cuda.is_available():
                        assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                            'Please use MMCV >= 1.4.4 for CPU training!'
                    model = revert_sync_batchnorm(model)
                    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
                    results, labels = single_gpu_test(
                        model,
                        data_loader,
                        args.show,
                        args.show_dir,
                        False,
                        args.opacity,
                        pre_eval=args.eval is not None and not eval_on_format_results,
                        format_only=args.format_only or eval_on_format_results,
                        format_args=eval_kwargs)
                else:
                    model = MMDistributedDataParallel(
                        model.cuda(),
                        device_ids=[torch.cuda.current_device()],
                        broadcast_buffers=False)
                    results = multi_gpu_test(
                        model,
                        data_loader,
                        args.tmpdir,
                        args.gpu_collect,
                        False,
                        pre_eval=args.eval is not None and not eval_on_format_results,
                        format_only=args.format_only or eval_on_format_results,
                        format_args=eval_kwargs)
                for index, result in enumerate(results):
                    cv2.imwrite(os.path.join("{0}/inferences/pred{1}_hold_out_{2}.png".format(test_path, index, hold_out_eval)), result * 255, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    cv2.imwrite(os.path.join("{0}/inferences/label{1}_hold_out_{2}.png".format(test_path, index, hold_out_eval)), labels[index] * 120, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                    np.save(os.path.join("{0}/inferences/pred{1}_hold_out_{2}".format(test_path, index, hold_out_eval)), result)
                    np.save(os.path.join("{0}/inferences/label{1}_hold_out_{2}".format(test_path, index, hold_out_eval)), labels[index])
                print("len(results)", len(results))
                print(results[0].shape)
                print("len(labels)", len(labels))
                print(labels[0].shape)

                results = np.concatenate(results).ravel()
                labels = np.concatenate(labels).ravel()
                print(results.shape)
                print(labels.shape)

                accuracy, bal_accuracy, precision, precision_weighted, recall, recall_weighted, iou_score, f1_score, f1_micro, f1_macro, f0_5, f0_1, f10, precision_per_class, recall_per_class, fscore_per_class, support_per_class, iou_score_per_class = metrics(
                    y_pred=results, y_true=labels)

                results_df = pd.DataFrame({"Epoch": [epoch],
                                           "Checkpoint": [test_path],
                                           "Test IoU": [iou_score],
                                           "Test Acc": [accuracy],
                                           "Test Prec": [precision],
                                           "Test Prec Weighted": [precision_weighted],
                                           "Test Recall": [recall],
                                           "Test Recall Weighted": [recall_weighted],
                                           "Test Bal Acc": [bal_accuracy],
                                           "Test F1": [f1_score],
                                           "Test f1_micro": [f1_micro],
                                           "Test f1_macro": [f1_macro],
                                           "Test F0.1": [f0_1],
                                           "Test F0.5": [f0_5],
                                           "Test F10": [f10],
                                           "Test F1 class 1": [fscore_per_class[0]],
                                           "Test F1 class 2": [fscore_per_class[1]],
                                           "Test Recall class 1": [recall_per_class[0]],
                                           "Test Recall class 2": [recall_per_class[1]],
                                           "Test Precision class 1": [precision_per_class[0]],
                                           "Test Precision class 2": [precision_per_class[1]],
                                           "Test IoU class 1": [iou_score_per_class[0]],
                                           "Test IoU class 2": [iou_score_per_class[1]],
                                           })
                print(results_df)

                # results_df.to_csv("/dccstor/geofm-finetuning/flood_mapping/inferences/{0}_epoch_{1}_single_eval.csv".format(test_path, epoch), mode='a', index=False, header=True)
                results_df.to_csv(
                    "{0}/inferences/pixel_wise_single_eval_hold_out_{1}.csv".format(test_path, hold_out_eval), mode='a',
                    index=False, header=False)

                rank, _ = get_dist_info()
                if rank == 0:
                    if args.out:
                        warnings.warn(
                            'The behavior of ``args.out`` has been changed since MMSeg '
                            'v0.16, the pickled outputs could be seg map as type of '
                            'np.array, pre-eval results or file paths for '
                            '``dataset.format_results()``.')
                        print(f'\nwriting results to {args.out}')
                        mmcv.dump(results, args.out)
                    if args.eval:
                        eval_kwargs.update(metric=args.eval)
                        metric = dataset.evaluate(results, **eval_kwargs)
                        metric_dict = dict(config=args.config, metric=metric)
                        mmcv.dump(metric_dict, json_file, indent=4)
                        if tmpdir is not None and eval_on_format_results:
                            # remove tmp dir when cityscapes evaluation
                            shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()