import cv2
from albumentations import (
    CLAHE,
    Compose,
    GaussNoise,
    HorizontalFlip,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    RandomRotate90,
    Resize,
    Affine,
    VerticalFlip,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size=384):
    """Medium-strength online augmentation for segmentation training."""
    return Compose([
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.3),
        Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.08, 0.08),
            rotate=(-20, 20),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.6,
        ),
        RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
        HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3,
        ),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        GaussNoise(std_range=(10.0 / 255.0, 40.0 / 255.0), p=0.2),
        Resize(
            image_size,
            image_size,
            interpolation=cv2.INTER_LINEAR,
        ),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size=384):
    return Compose([
        Resize(
            image_size,
            image_size,
            interpolation=cv2.INTER_LINEAR,
        ),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
