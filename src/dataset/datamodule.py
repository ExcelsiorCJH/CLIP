import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .dataset import CLIPDataset
from .utils import get_transform


class CLIPDataModule:
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        img_dir: str,
        tokenizer_name: str,
        img_size: int = 224,
        txt_max_length: int = 200,
        val_size: float = 0.2,
        test_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.img_dir = img_dir
        self.tokenizer_name = tokenizer_name
        self.img_size = img_size
        self.txt_max_length = txt_max_length
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.setup()

    def setup(self):
        # load data
        self.df = pd.read_csv(self.data_path)

        # tokenizer & transform
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.train_transform = get_transform(img_size=self.img_size, stage="train")
        self.test_transform = get_transform(img_size=self.img_size, stage="test")

        # train/val/test split
        train_df, self.test_df = train_test_split(
            self.df, test_size=self.test_size, shuffle=True
        )
        self.train_df, self.val_df = train_test_split(
            train_df, test_size=self.val_size, shuffle=True
        )

        # train/val/test set
        self.trainset = CLIPDataset(
            self.train_df,
            self.img_dir,
            self.tokenizer,
            self.train_transform,
            self.txt_max_length,
        )

        self.valset = CLIPDataset(
            self.val_df,
            self.img_dir,
            self.tokenizer,
            self.test_transform,
            self.txt_max_length,
        )
        self.testset = CLIPDataset(
            self.test_df,
            self.img_dir,
            self.tokenizer,
            self.test_transform,
            self.txt_max_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
