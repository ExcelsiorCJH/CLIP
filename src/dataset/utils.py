import albumentations as A

from albumentations.pytorch import ToTensorV2


def get_transform(img_size: int, max_pixel_value: float = 255.0, stage: str = "train"):
    if stage == "train":
        transform = A.Compose(
            [
                A.Resize(img_size, img_size, always_apply=True),
                A.Normalize(max_pixel_value=max_pixel_value, always_apply=True),
                ToTensorV2(),
            ]
        )
    else:
        transform = A.Compose(
            [
                A.Resize(img_size, img_size, always_apply=True),
                A.Normalize(max_pixel_value=max_pixel_value, always_apply=True),
                ToTensorV2(),
            ]
        )

    return transform
