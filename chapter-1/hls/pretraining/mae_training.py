import argparse
import functools
import time
from collections import deque

import torch
import torch.distributed as dist
import torch.optim as optim
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.distributed import DistributedSampler

from mae.models_mae import MaskedAutoencoderViT
# from preprocessing.dataset import HLS2Dataset as HLSDataset


import os
import json
import random
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter


class HLSDataset(Dataset):
    def __init__(self, data_path, num_frames, img_size=224, bands=["B02", "B03", "B04", "B05"],
                 random_cropping=True, remove_cloud=True, normalize=True,
                 # small trick: as all original values are integers, we make mean values as non-integer
                 # so normalized values have no 0, so that we can mark nodata as 0 because it is a good practice
                 # to fill nodata as mean/median.
                 mean=[431.5, 713.5, 747.5, 2512.5], std=[336, 388, 521, 1062], indices=None):
        self.data_path = data_path
        self.num_frames = num_frames
        self.img_size = img_size
        self.bands = bands
        self.random_cropping = random_cropping
        self.remove_cloud = remove_cloud
        self.normalize = normalize
        self.mean = mean  # corresponding mean per band for normalization purpose
        self.std = std  # corresponding std per band for normalization purpose

        self.layout = self.get_data_layout()
        if indices is not None:
            self.quality_only = True
            self.indices = indices
        else:
            self.quality_only = False
            self.indices = self.get_indices()

    def get_data_layout(self):
        layout = {}
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.tif'):
                    tile = file.split('.')[2]
                    date = file.split('.')[3].split('T')[0]
                    band = file.split('.')[6]
                    if tile not in layout:
                        layout[tile] = {}
                    if date not in layout[tile]:
                        layout[tile][date] = {}
                    layout[tile][date][band] = os.path.join(root, file)
        return layout  # {'T14SMA': {'2017005': {'B02': '/home/lchu/hls/HLS.L30.T14SMA.2017005T170835.v2.0.B02.tif', ...

    def get_indices(self):
        indices = []
        tiles = sorted(self.layout.keys())
        for tile in tiles:
            dates = sorted(self.layout[tile].keys())
            start_date_indices = range(len(dates) - self.num_frames + 1)
            for start_date_idx in start_date_indices:
                if self.random_cropping:
                    col_start = random.randrange(3660 - self.img_size + 1)
                    row_start = random.randrange(3660 - self.img_size + 1)
                    indices.append((tile, dates[start_date_idx:start_date_idx + self.num_frames], col_start, row_start))
                else:
                    for col_start in range(0, 3660 - self.img_size, self.img_size):
                        for row_start in range(0, 3660 - self.img_size, self.img_size):
                            indices.append(
                                (tile, dates[start_date_idx:start_date_idx + self.num_frames], col_start, row_start))
        return indices  # (tile_id, dates, col_start, row_start)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.quality_only:
            tile_id, dates, col_start, row_start = self.indices[index]
            res = []
            for date in dates:
                channels = []
                for band in self.bands:
                    tif_file = self.layout[tile_id][date][band]
                    with rasterio.open(tif_file) as src:
                        img = src.read(1, window=Window(col_start, row_start, self.img_size,
                                                        self.img_size))  # img_size, img_size
                        channels.append(img)
                        if img.shape != (224, 224):
                            print(self.indices[index])
                            print(src.read(1).shape)
                            print(tile_id, date, band, img.shape, col_start, row_start)
                channels = np.stack(channels, -1)  # img_size, img_size, C
                channels = (channels - self.mean) / self.std
                res.append(channels)
            res = np.stack(res, 0)  # num_frames, img_size, img_size, C
            return np.moveaxis(res, -1, 0).astype('float32')  # C, num_frames, img_size, img_size

        tile_id, dates, col_start, row_start = self.indices[index]
        # we re-genereate random col_start and row_start here to make sure different data in different epochs,
        # otherwise, self.indices will remain same across different epochs as it is generated during dataset creation.
        if self.random_cropping:
            col_start = random.randrange(3660 - self.img_size + 1)
            row_start = random.randrange(3660 - self.img_size + 1)
        res = []
        for date in dates:
            channels = []
            for band in self.bands:
                tif_file = self.layout[tile_id][date][band]
                with rasterio.open(tif_file) as src:
                    img = src.read(1, window=Window(col_start, row_start, self.img_size,
                                                    self.img_size))  # img_size, img_size
                    channels.append(img)
            channels = np.stack(channels, -1)  # img_size, img_size, C

            if self.remove_cloud:
                # mark all cloud and cloud shadow pixels as nodata
                fmask_file = self.layout[tile_id][date]["Fmask"]
                with rasterio.open(fmask_file) as src:
                    fmask = src.read(1, window=Window(col_start, row_start, self.img_size, self.img_size))
                cloud_mask = fmask << 4 >> 5 != 0  # mask = (fmask << 4 >> 5 != 0) & (fmask << 4 >> 5 != 2)
                channels[cloud_mask] = -9999  # Mark all cloud pixels as nodata

            if self.normalize:
                channels = np.where(channels == -9999, 0.0001,
                                    (channels - self.mean) / self.std)  # don't normalize on nodata
            else:
                channels = channels * 0.0001  # if not normalize, just rescale

            res.append(channels)
        res = np.stack(res, 0)  # num_frames, img_size, img_size, C
        return np.moveaxis(res, -1, 0).astype('float32')  # C, num_frames, img_size, img_size


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # data loader related
    parser.add_argument('--train_dir', default='/dev/shm/train', type=str,
                        help='Path to the train data directory.')
    parser.add_argument('--val_dir', default='/dev/shm/val', type=str,
                        help='Path to the validation data directory.')
    parser.add_argument('--num_frames', default=3, type=int,
                        help='Number of frames in a sample.')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Input image size.')
    parser.add_argument('--bands', default=["B02", "B03", "B04", "B05"], type=str, nargs='+',
                        help='Spectral bands to use.',
                        choices=["B02", "B03", "B04", "B05", "B06", "B07", "B09", "B10", "B11"])
    parser.add_argument('--random_cropping', action='store_true',
                        help='Use random cropping for input data. Default = True')
    parser.add_argument('--no_random_cropping', action='store_false', dest='random_cropping')
    parser.set_defaults(random_cropping=True)
    parser.add_argument('--data_loader_num_workers', default=2, type=int,
                        help='Number of data loader workers.')

    # model related
    parser.add_argument('--num_layers', default=12, type=int,
                        help='Number of layers in the model.')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Input patch size.')
    parser.add_argument('--embed_dim', default=768, type=int,
                        help='Number of embeddings dimensions.')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='Number of heads in the model.')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--tubelet_size', default=1, type=int,
                        help='Temporal patch size.')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='Path to a checkpoint file to load from.')

    # training related
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    # LR decay not being used
    # parser.add_argument('--lr_decay', type=float, default=0.85,
    #                     help='Learning rate decay')
    parser.add_argument('--num_epochs', default=50, type=int)

    # logging related
    parser.add_argument('--base_log_dir', default='/workspace/data/lchu/hls/logs',
                        help='Path to the root directory where to save log files.')
    parser.add_argument('--base_checkpoint_dir', default='/workspace/data/lchu/hls/checkpoints',
                        help='Path to root directory where to save checkpoints.')
    parser.add_argument('--base_visualization_dir', default='/workspace/data/lchu/hls/vis',
                        help='Path to the root directory where to save visualizations.')
    parser.add_argument('--base_job_info_dir', default='/workspace/data/lchu/hls/jobs',
                        help='Path to the root directory where to save job info file.')

    return parser


def train(
        model,
        mask_ratio,
        local_rank,
        rank,
        train_loader,
        optimizer,
        epoch,
        sampler=None,
        profiler=None,
        scheduler=None,
        vis_path=None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="Training Epoch"
        )
    # start = time.time()
    for i, batch in enumerate(train_loader):
        # end = time.time()
        # if rank == 0:
        #     print(f"data loading time: {end - start}")
        #
        # if rank == 0:
        #     print(f"test random seed: {batch.sum()}")

        optimizer.zero_grad()

        loss, pred, mask = model(batch.to(local_rank), mask_ratio)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        if rank == 0:
            inner_pbar.update(1)
        if profiler:
            profiler.step()
        # start = time.time()
        # if rank == 0:
        #     print(f"model training time: {start - end}")

        scheduler.step()

        if epoch == 100:
            if rank == 0 and vis_path is not None:
                os.makedirs(vis_path, exist_ok=True)
                torch.save(batch.detach().cpu(), os.path.join(vis_path, f'input_{i}.pt'))
                torch.save(model.module.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu(), os.path.join(vis_path, f'mask_{i}.pt'))
                torch.save(model.module.unpatchify(pred).detach().cpu(), os.path.join(vis_path, f'pred_{i}.pt'))
        dist.barrier()

    # consolidate final loss number - do not use .reduce here, requires global synch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    if rank == 0:
        inner_pbar.close()

        print(f"Train Epoch: \t{epoch}, Loss: \t{train_loss:.4f}")
    return train_loss


def validation(model, mask_ratio, local_rank, rank, test_loader):
    model.eval()
    ddp_loss = torch.zeros(2).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(test_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in test_loader:

            loss, pred, mask = model(batch.to(local_rank), mask_ratio)

            ddp_loss[0] += loss.item()  # sum up batch loss
            ddp_loss[1] += 1

            if rank == 0:
                inner_pbar.update(1)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / ddp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def fsdp_main(args):
    """main process, run within each individual GPU process"""

    # TODO: can we get the time the job was submitted?
    start_time = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"

    # debug nan gradient
    torch.autograd.set_detect_anomaly(True)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    ### Configs
    # data loader related
    train_dir = args.train_dir
    val_dir = args.val_dir
    num_frames = args.num_frames
    img_size = args.img_size
    bands = args.bands
    random_cropping = args.random_cropping
    num_workers = args.data_loader_num_workers

    # model related
    num_layers = args.num_layers
    patch_size = args.patch_size
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    mask_ratio = args.mask_ratio
    tubelet_size = args.tubelet_size
    checkpoint = args.checkpoint

    # training related
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.num_epochs

    # logging related
    job_id = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    ckpt_dir = os.path.join(args.base_checkpoint_dir, job_id)
    vis_dir = os.path.join(args.base_visualization_dir, job_id)
    job_info_dir = os.path.join(args.base_job_info_dir)
    tensorboard_log_dir = os.path.join(args.base_log_dir, "tensorboard", job_id)
    csv_log_dir = os.path.join(args.base_log_dir, "csv", job_id)

    # save job info in a yaml file
    if rank == 0:
        params_dict = dict(vars(args))

        # Add more info
        params_dict['job_id'] = job_id
        params_dict['checkpoint_dir'] = ckpt_dir
        params_dict['visualization_dir'] = vis_dir
        params_dict['tensorboard_dir'] = tensorboard_log_dir
        params_dict['csv_dir'] = csv_log_dir
        params_dict['world_size'] = world_size
        params_dict['job_start_time'] = start_time
        params_dict['job_finish_time'] = 'NA'

        os.makedirs(job_info_dir, exist_ok=True)
        with open(os.path.join(job_info_dir, f'{job_id}.yaml'), 'w') as f:
            yaml.safe_dump(params_dict, f, default_flow_style=None, sort_keys=False)

    # set seed in a way that:
    # 1. ensure reproducibility
    # 2. make sure each gpu has different seed to make sure different
    # gpus crop the same images randomly
    random.seed(2022+rank)
    torch.cuda.manual_seed(2022+rank)
    torch.manual_seed(2022+rank)

    # distributed setup
    dist.init_process_group("nccl")
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # get input metadata
    with open("/workspace/data/lchu/hls/meta/image_meta-hls_l30-201707_201712-mc000-ws3_224_224-st1_224_224.json") as f:
        input_meta_data = json.load(f)

    # create model
    model = MaskedAutoencoderViT(img_size=img_size, patch_size=patch_size,
                 num_frames=num_frames, tubelet_size=tubelet_size,
                 in_chans=len(input_meta_data['bands']), embed_dim=embed_dim, depth=num_layers, num_heads=num_heads,
                 decoder_embed_dim=int(embed_dim/2), decoder_depth=8, decoder_num_heads=num_heads,
                 mlp_ratio=4., norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6), norm_pix_loss=False)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params.\n")

    # ____________ create batch dataset
    train_dataset = HLSDataset(train_dir, num_frames=num_frames, img_size=img_size, bands=input_meta_data['bands'],
                               random_cropping=random_cropping, remove_cloud=True, normalize=True,
                               mean=input_meta_data['image_mean'], std=input_meta_data['image_standard_deviation'],
                               indices=[[idx[0], idx[1], idx[3], idx[2]] for idx in input_meta_data['indices']
                                        if idx[2] != 3584 and idx[3] != 3584])
    if rank == 0:
        print(f"--> Training set len = {len(train_dataset)}")

    val_dataset = HLSDataset(val_dir, num_frames=num_frames, img_size=img_size, bands=input_meta_data['bands'],
                             random_cropping=random_cropping, remove_cloud=True, normalize=True,
                             mean=input_meta_data['image_mean'], std=input_meta_data['image_standard_deviation'],
                             indices=[[idx[0], idx[1], idx[3], idx[2]] for idx in input_meta_data['indices']
                                        if idx[2] != 3584 and idx[3] != 3584][:100])
    if rank == 0:
        print(f"--> Validation set len = {len(val_dataset)}")

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": batch_size, "sampler": train_sampler}
    test_kwargs = {"batch_size": 2, "sampler": val_sampler}
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": False,
        "drop_last": True
    }
    train_kwargs.update(common_kwargs)
    test_kwargs.update(common_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    torch.cuda.set_device(local_rank)

    torch.cuda.empty_cache()

    model = model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    if checkpoint:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        model.load_state_dict(
            torch.load(checkpoint, map_location=map_location))

    # optimizer and learning rate decay
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr*10, steps_per_epoch=len(train_loader), epochs=epochs)

    best_val_loss = float("inf")

    # --- main training loop
    if rank == 0:
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        dq = deque(maxlen=3)
        training_start_time = time.time()

    # torch profiler
    torch_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "profile_traces"
        ),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )
    torch_profiler = None

    # Log Writers
    if rank == 0:
        tensorboard_writer = SummaryWriter(tensorboard_log_dir)
        os.makedirs(csv_log_dir, exist_ok=True)
        log_writer = open(os.path.join(csv_log_dir, "summary.txt"), "a")

    if rank == 0:
        mem_alloc_tracker = []

    # -- Start Training -----
    for epoch in range(1, epochs + 1):
        if rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

            t0 = time.time()
        train_loss = train(
            model,
            mask_ratio,
            local_rank,
            rank,
            train_loader,
            optimizer,
            epoch,
            sampler=train_sampler,
            profiler=torch_profiler,
            scheduler=scheduler,
            vis_path=vis_dir,
        )

        curr_val_loss = validation(model, mask_ratio, local_rank, rank, test_loader)

        # Write logs in two formats: tensorboard and csv.
        if rank == 0:
            tensorboard_writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            tensorboard_writer.add_scalars("Loss", {
                "train": train_loss,
                "test":  curr_val_loss
            }, epoch)
            log_writer.write(f"{epoch},{scheduler.get_last_lr()[0]},{train_loss},{curr_val_loss}\n")
            # flush on each write to avoid log loss due to unexpected exit
            tensorboard_writer.flush()
            log_writer.flush()

        if rank == 0:
            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_loss.item())

            val_acc_tracking.append(curr_val_loss.item())

            mem_alloc_tracker.append(
                round((torch.cuda.memory_allocated() / 1024 ** 3), ndigits=4)
            )

        # save this epochs checkpoint if val loss is current best
        if curr_val_loss < best_val_loss:
            if rank == 0:
                print(f"--> saving model ...")
                filename = f"epoch-{epoch}-loss-{round(curr_val_loss.item(), 4)}.pt"
                checkpoint_file = os.path.join(ckpt_dir, filename)
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_file)
                print(f"--> saved {checkpoint_file} to COS")

        # announce new val loss record:
        if rank == 0 and curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            print(f"-->>>> New Val Loss Record: {best_val_loss}")

    # init_end_event.record()
    if rank == 0:

        total_training_time = time.time() - training_start_time
        print(f"Total training time = {total_training_time:.2f}")
        print("Times per epoch:")
        for i, val in enumerate(dur):
            print(f"epoch {i}, time {val:.2f}")
        print()

        # training is done...show some training stats for memory use.
        print(f"total memory allocated: {mem_alloc_tracker}")

        print(f"Training accuracy: {train_acc_tracking}")
        print(f"Validation accuracy: {val_acc_tracking}")
        print(f"\n Best Val accuracy: {best_val_loss}")

        # memory summary
        print(f"CUDA Memory Summary After Last training:\n {torch.cuda.memory_summary()}")

        # close tensorboard writer
        tensorboard_writer.flush()
        tensorboard_writer.close()
        log_writer.close()

        # Update job info file
        with open(os.path.join(job_info_dir, f'{job_id}.yaml'), 'r') as f:
            params_dict = yaml.safe_load(f)
            params_dict['job_finish_time'] = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"

        with open(os.path.join(job_info_dir, f'{job_id}.yaml'), 'w') as f:
            yaml.safe_dump(params_dict, f, default_flow_style=None, sort_keys=False)

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    # dist.barrier()
    # dist.destroy_process_group()


# ------------------ Main functions above ------------


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    fsdp_main(args)

