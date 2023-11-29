import os
import itertools
import logging

import yaml
import omegaconf

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import sys

sys.path.append("..")

from .utils import AverageMeter
from ..model.clip import CLIP
from ..dataset.datamodule import CLIPDataModule


class Trainer:
    def __init__(self, config, scaler):
        self.config = config
        self.nprocs = torch.cuda.device_count()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # model
        self.model = CLIP(**config.model)
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
        self.scaler = scaler

        # datamodule(dm)
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
                f"{self.config.datamodule.dataset_name}-version-{self.version}",
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

        config.version = self.version
        with open(
            os.path.join(self.save_path, "config.yaml"), "w", encoding="utf8"
        ) as outfile:
            yaml.dump(
                omegaconf.OmegaConf.to_container(config),
                outfile,
                default_flow_style=False,
                allow_unicode=True,
            )

        # experiment-logging options
        self.best_result = {"version": self.version}

    def configure_optimizers(self):
        params = [
            {
                "params": self.model.module.img_encoder.parameters(),
                "lr": self.config.train.img_encoder_lr,
            },
            {
                "params": self.model.module.text_encoder.parameters(),
                "lr": self.config.train.text_encoder_lr,
            },
            {
                "params": itertools.chain(
                    self.model.module.img_projection.parameters(),
                    self.model.module.text_projection.parameters(),
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

        ckpt_path = os.path.join(
            self.save_path, f"epoch_{epoch}_loss_{val_loss:.4f}.pt"
        )

        save_top_k = self.config.train.save_top_k
        self.ckpt_paths.append(ckpt_path)
        if save_top_k < len(self.ckpt_paths):
            for path in self.ckpt_paths[:-save_top_k]:
                os.remove(path)

            self.ckpt_paths = self.ckpt_paths[-save_top_k:]

        torch.save(model.module.state_dict(), ckpt_path)

    def fit(self) -> dict:
        for epoch in tqdm(range(self.config.train.epochs), desc="epoch"):
            logging.info(f"* Learning Rate: {self.optimizer.param_groups[0]['lr']:.5f}")
            result = self._train_epoch(epoch)

            # update checkpoint
            if result["val_loss"] < self.global_val_loss:
                self.save_checkpoint(epoch, result["val_loss"], self.model)

            self.lr_scheduler.step(result["val_loss"])

        self.summarywriter.close()
        return self.version

    def _train_epoch(self, epoch: int) -> dict:
        train_loss = AverageMeter()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="train_steps",
            total=len(self.train_loader),
        ):
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "caption"}

            self.optimizer.zero_grad()
            if self.config.dp.amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    loss = outputs["loss"].mean()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = outputs["loss"].mean()
                loss.backward()
                self.optimizer.step()

            train_loss.update(loss.item())

            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                logging.info(
                    f"[DDP Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {loss.item():.3f}"
                )

        train_loss = train_loss.avg
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

        logging.info(f"** global step: {self.global_step}, val loss: {val_loss:.4f}")
        return {"val_loss": val_loss}

    def validate(self, epoch: int) -> dict:
        val_loss = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader),
                desc="valid_steps",
                total=len(self.val_loader),
            ):
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if k != "caption"
                }

                outputs = self.model(batch)
                loss = outputs["loss"].mean()
                val_loss.update(loss.item())

        return val_loss.avg
