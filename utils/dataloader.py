import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


try:
    from utils.transform import build_train_transform
except Exception:
    build_train_transform = None


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, split_file=None, dataset_root=None, use_augmentation=False):
        self.trainsize = trainsize
        self.use_augmentation = use_augmentation

        if split_file is not None:
            if dataset_root is None:
                raise ValueError('dataset_root must be set when split_file is provided')

            dataset_root = Path(dataset_root)
            self.images, self.gts = self._load_from_split(Path(split_file), dataset_root)
        else:
            self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.images = sorted(self.images)
            self.gts = sorted(self.gts)

        self.filter_files()
        self.size = len(self.images)
        if self.use_augmentation:
            if build_train_transform is None:
                raise ImportError('Augmentation is enabled but utils.transform.build_train_transform is unavailable.')
            self.train_aug = build_train_transform(image_size=self.trainsize)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        if self.use_augmentation:
            image_np = np.array(image)
            gt_np = np.array(gt)
            augmented = self.train_aug(image=image_np, mask=gt_np)

            image = augmented['image']
            gt = augmented['mask']

            image = torch.from_numpy(image).permute(2, 0, 1).float()
            gt = torch.from_numpy(gt).unsqueeze(0).float()
            if gt.max() > 1:
                gt = gt / 255.0
            return image, gt

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def _load_from_split(self, split_file, dataset_root):
        if not split_file.exists():
            raise FileNotFoundError(f'Split file not found: {split_file}')

        images = []
        gts = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f'Invalid split line in {split_file}: {line}')

                img_rel, gt_rel = parts
                img_path = (dataset_root / img_rel).resolve()
                gt_path = (dataset_root / gt_rel).resolve()
                images.append(str(img_path))
                gts.append(str(gt_path))

        if len(images) == 0:
            raise RuntimeError(f'No samples found in split file: {split_file}')

        return images, gts

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    split_file=None,
    dataset_root=None,
    use_augmentation=False,
):

    dataset = PolypDataset(
        image_root,
        gt_root,
        trainsize,
        split_file=split_file,
        dataset_root=dataset_root,
        use_augmentation=use_augmentation,
    )
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
