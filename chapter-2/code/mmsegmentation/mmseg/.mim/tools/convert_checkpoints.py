import torch

file_path = '/dccstor/geofm-finetuning/pretrain_ckpts/mae_weights/2023-04-01_02-37-43/epoch-402-loss-0.0739.pt'
out_path = '/dccstor/geofm-finetuning/pretrain_ckpts/mae_weights/2023-04-01_02-37-43/epoch-402-loss-0.0739_old_version.pt'

ckpt = torch.load(file_path, map_location='cpu')
torch.save(ckpt['model'], out_path)