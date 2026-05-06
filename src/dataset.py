"""
DeepFashion In-Shop Clothes Retrieval Dataset
Handles train / query / gallery splits with item_id-level ground truth.
"""

import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

import random
from collections import defaultdict

# ─────────────────────────────────────────────
#  Parsing helpers
# ─────────────────────────────────────────────

def parse_eval_partition(partition_file: str):
    """
    Returns:
        splits       -> {'train': [...], 'query': [...], 'gallery': [...]}
        img_to_item  -> {img_path: item_id}

    Dataset format:
        image_path item_id split
    """

    splits = {
        "train": [],
        "query": [],
        "gallery": []
    }

    img_to_item = {}

    with open(partition_file, "r") as f:
        lines = f.readlines()

    # Skip:
    # line 1 -> number of images
    # line 2 -> column names
    lines = lines[2:]

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 3:
            continue

        img_path = parts[0]
        item_id = parts[1]
        split = parts[2].lower()

        img_to_item[img_path] = item_id

        if split in splits:
            splits[split].append(img_path)

    return splits, img_to_item


def parse_bboxes(bbox_file: str):
    """
    Returns dict:
        {relative_img_path -> [x1, y1, x2, y2]}
    """

    img_to_bbox = {}

    with open(bbox_file, "r") as f:
        lines = f.readlines()

    # Skip metadata/header
    lines = lines[2:]

    for line in lines:
        parts = line.strip().split()

        if len(parts) >= 6:
            img_path = parts[0]

            try:
                bbox = [int(x) for x in parts[2:6]]
                img_to_bbox[img_path] = bbox
            except:
                pass

    return img_to_bbox


# ─────────────────────────────────────────────
#  Dataset class
# ─────────────────────────────────────────────

class DeepFashionDataset(Dataset):

    def __init__(
        self,
        img_root,
        image_paths,
        img_to_item,
        img_to_bbox=None,
        transform=None,
        use_gt_bbox=True,
    ):

        self.img_root = Path(img_root)
        self.image_paths = image_paths
        self.img_to_item = img_to_item
        self.img_to_bbox = img_to_bbox or {}
        self.transform = transform
        self.use_gt_bbox = use_gt_bbox

        # Create label mapping
        self.item_ids = sorted(
            list(set(img_to_item[p] for p in image_paths))
        )

        self.item_to_label = {
            item_id: idx
            for idx, item_id in enumerate(self.item_ids)
        }

    # ─────────────────────────────────────────

    def _load_image(self, rel_path):

        full_path = self.img_root / rel_path

        image = Image.open(full_path).convert("RGB")

        # Optional GT bbox crop
        if self.use_gt_bbox and rel_path in self.img_to_bbox:

            x1, y1, x2, y2 = self.img_to_bbox[rel_path]

            # Fix invalid bbox annotations
            if x2 <= x1 or y2 <= y1:
                return image
            
            # Clamp to image boundaries
            w, h = image.size
            
            x1 = max(0, min(x1, w - 1))
            x2 = max(1, min(x2, w))
            
            y1 = max(0, min(y1, h - 1))
            y2 = max(1, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return image
            
            image = image.crop((x1, y1, x2, y2))

        return image

    # ─────────────────────────────────────────

    def __len__(self):
        return len(self.image_paths)

    # ─────────────────────────────────────────

    def __getitem__(self, idx):

        rel_path = self.image_paths[idx]

        image = self._load_image(rel_path)

        if self.transform:
            image = self.transform(image)

        item_id = self.img_to_item[rel_path]

        label = self.item_to_label[item_id]

        return image, item_id, label, rel_path


# ─────────────────────────────────────────────
#  CLIP transforms
# ─────────────────────────────────────────────

def get_clip_transform(
    image_size=224,
    augment=False
):

    CLIP_MEAN = (
        0.48145466,
        0.4578275,
        0.40821073
    )

    CLIP_STD = (
        0.26862954,
        0.26130258,
        0.27577711
    )

    if augment:

        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0)
            ),

            transforms.RandomHorizontalFlip(),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                CLIP_MEAN,
                CLIP_STD
            ),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            CLIP_MEAN,
            CLIP_STD
        ),
    ])


# ─────────────────────────────────────────────
#  Dataloader builder
# ─────────────────────────────────────────────

def build_dataloaders(
    dataset_root,
    batch_size=64,
    num_workers=4,
    image_size=224,
    use_gt_bbox=True,
):

    root = Path(dataset_root)

    partition_file = root / "list_eval_partition.txt"
    bbox_file = root / "list_bbox_inshop.txt"

    img_root = root / "img"

    # FIXED PARSER
    splits, img_to_item = parse_eval_partition(
        str(partition_file)
    )

    # Optional bbox parsing
    if bbox_file.exists():
        img_to_bbox = parse_bboxes(str(bbox_file))
    else:
        img_to_bbox = {}

    loaders = {}

    for split in ["train", "query", "gallery"]:

        augment = (split == "train")

        dataset = DeepFashionDataset(
            img_root=str(img_root),
            image_paths=splits[split],
            img_to_item=img_to_item,
            img_to_bbox=img_to_bbox,
            transform=get_clip_transform(
                image_size=image_size,
                augment=augment
            ),
            use_gt_bbox=use_gt_bbox,
        )

        if split == "train":

    sampler = ClassBalancedSampler(
        dataset,
        classes_per_batch=batch_size // 4,
        samples_per_class=4,
    )

    loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    else:
    
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        loaders[split] = loader

        print(
            f"[Dataset] {split:8s}: "
            f"{len(dataset):,} images | "
            f"{len(dataset.item_ids):,} unique items"
        )

    return loaders

class ClassBalancedSampler(Sampler):
    """
    Creates batches like:
        32 classes × 4 images = 128 batch

    Ensures SupCon always has positives.
    """

    def __init__(
        self,
        dataset,
        classes_per_batch=32,
        samples_per_class=4,
    ):
        self.dataset = dataset
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class

        self.class_to_indices = defaultdict(list)

        for idx, rel_path in enumerate(dataset.image_paths):

            item_id = dataset.img_to_item[rel_path]
        
            self.class_to_indices[item_id].append(idx)

        self.classes = list(self.class_to_indices.keys())

        self.batch_size = (
            classes_per_batch * samples_per_class
        )

    def __iter__(self):

        random.shuffle(self.classes)

        batch = []

        for cls in self.classes:

            indices = self.class_to_indices[cls]

            if len(indices) >= self.samples_per_class:
                sampled = random.sample(
                    indices,
                    self.samples_per_class
                )
            else:
                sampled = random.choices(
                    indices,
                    k=self.samples_per_class
                )

            batch.extend(sampled)

            if len(batch) == self.batch_size:
                yield from batch
                batch = []

    def __len__(self):
        return len(self.classes) * self.samples_per_class
