import os

import omegaconf
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from module.trainer import Trainer
from module.utils import fix_seed

import warnings

warnings.filterwarnings(action="ignore")


def main_worker(rank, ngpus_per_node, config) -> None:
    print(f"Use GPU {config.rank_list[rank]} for training")
    fix_seed(config.train.seed)

    config.ddp.rank = config.ddp.rank * ngpus_per_node + rank
    dist.init_process_group(
        backend=config.ddp.dist_backend,
        rank=config.ddp.rank,
        world_size=config.ddp.world_size,
        init_method=config.ddp.dist_url,
    )

    scaler = torch.cuda.amp.GradScaler() if config.train.amp else None

    # trainer
    trainer = Trainer(
        config=config,
        scaler=scaler,
        rank=rank,
        ngpus_per_node=ngpus_per_node,
    )
    version = trainer.fit()

    return None


if __name__ == "__main__":
    config_path = "config/clip_config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    # GPU setting
    VISIBLE_GPUS = "0,1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
    config.rank_list = [int(i) for i in VISIBLE_GPUS.split(",")]

    # 'world_size' means total number of processes to run
    ngpus_per_node = torch.cuda.device_count()
    config.ddp.world_size = ngpus_per_node * config.ddp.world_size

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
