import os
import itertools
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import sys

sys.path.append("..")

from .utils import AverageMeter
from model.clip import CLIP
from dataset.datamodule import CLIPDataModule


class Trainer:
    def __init__(self, config, scaler, rank, ngpus_per_node):
        self.config = config
        self.rank = rank
        self.nprocs = torch.cuda.device_count()
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

        # model
        self.model = CLIP(**config.model)
        self.model = self.model.to(self.device, non_blocking=True)
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        self.scaler = scaler

        # datamodule(dm)
        config.datamodule.batch_size = int(
            config.datamodule.batch_size / ngpus_per_node
        )
        self.dm = CLIPDataModule(**config.datamodule)
        self.train_loader = self.dm.train_dataloader()
        self.val_loader = self.dm.val_dataloader()

        # optimizer
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # model-saving options
        self.version = 0
        self.ckpt_paths = []
        while True:
            ckpt_dir = self.config.train.ckpt_dir
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)

            self.save_path = os.path.join(
                ckpt_dir,
                f"version-{self.config.datamodule.dataset_name}-{self.version}",
            )
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        self.summarywriter = SummaryWriter(self.save_path)

        self.global_step = 0
        self.global_val_loss = 1e5
        self.eval_step = self.config.train.eval_step
        logging.basicConfig(
            filename=os.path.join(self.save_path, "experiment.log"),
            level=logging.INFO,
            format="%(asctime)s > %(message)s",
        )

        # experiment-logging options
        self.best_result = {"version": self.version}

    def configure_optimizers(self):
        params = [
            {
                "params": self.model.image_encoder.parameters(),
                "lr": self.config.train.image_encoder_lr,
            },
            {
                "params": self.model.text_encoder.parameters(),
                "lr": self.config.train.text_encoder_lr,
            },
            {
                "params": itertools.chain(
                    self.model.image_projection.parameters(),
                    self.model.text_projection.parameters(),
                ),
                "lr": self.config.train.proj_head_lr,
                "weight_decay": self.config.train.weight_decay,
            },
        ]
        # optimizer
        optimizer = optim.AdamW(params, weight_decay=0.0)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.config.train.patience,
            factor=self.config.train.factor,
        )
        return optimizer, lr_scheduler

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        model: nn.Module,
    ) -> None:
        logging.info(
            f"Val loss decreased ({self.global_val_loss:.4f} → {val_loss:.4f}). Saving model ..."
        )
        self.global_val_loss = val_loss

        ckpt_path = os.path.join(self.save_path, f"epoch_{epoch}_{val_loss:.4f}.pt")

        save_top_k = self.config.train.save_top_k
        self.ckpt_paths.append(ckpt_path)
        if save_top_k < len(self.ckpt_paths):
            for path in self.ckpt_paths[:-save_top_k]:
                os.remove(path)

            self.ckpt_paths = self.ckpt_paths[-save_top_k:]

        torch.save(self.model.state_dict(), ckpt_path)

    def fit(self) -> dict:
        for epoch in tqdm(range(self.config.train.epochs), desc="epoch"):
            self.train_sampler.set_epoch(epoch)

            logging.info(f"* Learning Rate: {self.optimizer.param_groups[0]['lr']:.5f}")
            result = self._train_epoch(epoch)

            # update checkpoint
            if self.rank == 0 and result["val_loss"] < self.global_val_loss:
                self.save_checkpoint(epoch, result["val_loss"], self.model)

            if self.rank == 0:
                self.lr_scheduler.step(result["val_loss"])
            else:
                self.lr_scheduler.step()

        if self.rank == 0:
            self.summarywriter.close()
        return self.version if self.rank == 0 else None

    def _train_epoch(self, epoch: int) -> dict:
        train_loss = AverageMeter()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="train_steps",
            total=len(self.train_loader),
            disable=self.rank in [0],
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            if self.config.amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    loss = outputs["loss"]
                dist.barrier()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

            train_loss.update(loss.item())

            if self.rank == 0:
                self.global_step += 1
                if self.global_step % self.eval_step == 0:
                    logging.info(
                        f"[DDP Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {loss.item():.3f}"
                    )

        train_loss = train_loss.avg

        if self.rank == 0:
            val_loss = self.validate(epoch)

            # tensorboard writing
            self.summarywriter.add_scalars(
                "lr", {"lr": self.optimizer.param_groups[0]["lr"]}, epoch
            )
            self.summarywriter.add_scalars(
                "loss/step", {"val": val_loss, "train": train_loss}, self.global_step
            )
            self.summarywriter.add_scalars(
                "loss/epoch", {"val": val_loss, "train": train_loss}, epoch
            )

            logging.info(
                f"** global step: {self.global_step}, val loss: {val_loss:.4f}%"
            )
            return {"val_loss": val_loss}

        return None

    def validate(self, epoch: int) -> dict:
        val_loss = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader),
                desc="valid_steps",
                total=len(self.val_loader),
            ):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(batch)
                loss = outputs["loss"]
                val_loss.update(loss.item())

        return val_loss.avg
