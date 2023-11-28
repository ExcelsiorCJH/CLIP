import os

import torch
import omegaconf

from .module.trainer import Trainer
from .module.utils import fix_seed

import warnings

warnings.filterwarnings(action="ignore")


def main(config) -> None:
    fix_seed(config.train.seed)

    scaler = torch.cuda.amp.GradScaler() if config.dp.amp else None

    # trainer
    trainer = Trainer(config=config, scaler=scaler)
    version = trainer.fit()

    return None


if __name__ == "__main__":
    config_path = "src/config/clip_config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    # transformers - tokenizers warning off
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(config)
