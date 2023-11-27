import os
import random

import cv2
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel


class CLIPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        tokenizer,
        transform,
        txt_max_length: int = 200,
    ):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.txt_max_length = txt_max_length

        # dataframe
        self.data = df

        # encoded_captions
        captions = self.data["caption"].tolist()
        self.encoded_captions = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.txt_max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data.iloc[idx]["image"]
        caption = self.data.iloc[idx]["caption"]

        # txt prep
        item = {k: v[idx] for k, v in self.encoded_captions.items()}

        # img prep
        img_path = os.path.join(self.img_dir, fname)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]

        item["image"] = img
        item["caption"] = caption

        return item
