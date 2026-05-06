"""
DeepFashion In-Shop Clothes Retrieval Dataset
Handles train / query / gallery splits with item_id-level ground truth.
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────
#  Parsing helpers
# ─────────────────────────────────────────────

def parse_eval_partition(partition_file: str) -> dict[str, list[str]]:
    """
    Returns dict: {'train': [...], 'query': [...], 'gallery': [...]}
    Each value is a list of relative image paths.
    """
    splits = {"train": [], "query": [], "gallery": []}
    with open(partition_file) as f:
        n = int(f.readline().strip())
        f.readline()  # skip header
        for _ in range(n):
            parts = f.readline().strip().split()
            img_path, split = parts[0], parts[1].lower()
            if split in splits:
                splits[split].append(img_path)
    return splits


def parse_item_ids(description_file: str) -> dict[str, str]:
    """
    Returns dict: {relative_img_path -> item_id}
    list_description_inshop.txt format:
        N
        image_name item_id  ...extra cols
    """
    img_to_item = {}
    with open(description_file) as f:
        n = int(f.readline().strip())
        f.readline()  # skip header
        for _ in range(n):
            parts = f.readline().strip().split()
            if len(parts) >= 2:
                img_to_item[parts[0]] = parts[1]
    return img_to_item


def parse_bboxes(bbox_file: str) -> dict[str, list[int]]:
    """
    Returns dict: {relative_img_path -> [x1, y1, x2, y2]}
    list_bbox_inshop.txt format:
        N
        image_name  item_id  x_1  y_1  x_2  y_2
    """
    img_to_bbox = {}
    with open(bbox_file) as f:
        n = int(f.readline().strip())
        f.readline()  # skip header
        for _ in range(n):
            parts = f.readline().strip().split()
            if len(parts) >= 6:
                img_to_bbox[parts[0]] = [int(x) for x in parts[2:6]]
    return img_to_bbox


# ─────────────────────────────────────────────
#  Dataset classes
# ─────────────────────────────────────────────

class DeepFashionDataset(Dataset):
    """
    Generic dataset for train / query / gallery split.

    Args:
        img_root      : Path to the root image directory (contains 'img/')
        image_paths   : List of relative image paths for this split
        img_to_item   : Dict mapping img_path -> item_id
        img_to_bbox   : Dict mapping img_path -> [x1,y1,x2,y2] (optional)
        transform     : torchvision transform applied to the PIL image
        use_gt_bbox   : If True, crop using ground-truth bbox before transform
    """

    def __init__(
        self,
        img_root: str,
        image_paths: list[str],
        img_to_item: dict[str, str],
        img_to_bbox: dict[str, list[int]] | None = None,
        transform=None,
        use_gt_bbox: bool = True,
    ):
        self.img_root = Path(img_root)
        self.image_paths = image_paths
        self.img_to_item = img_to_item
        self.img_to_bbox = img_to_bbox or {}
        self.transform = transform
        self.use_gt_bbox = use_gt_bbox

        # Build unique sorted item_ids and label mapping
        self.item_ids = sorted(set(img_to_item[p] for p in image_paths if p in img_to_item))
        self.item_to_label = {item: idx for idx, item in enumerate(self.item_ids)}

    # ── helpers ──────────────────────────────

    def _load_image(self, rel_path: str) -> Image.Image:
        full = self.img_root / rel_path
        img = Image.open(full).convert("RGB")
        if self.use_gt_bbox and rel_path in self.img_to_bbox:
            x1, y1, x2, y2 = self.img_to_bbox[rel_path]
            img = img.crop((x1, y1, x2, y2))
        return img

    # ── dunder ───────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        rel_path = self.image_paths[idx]
        img = self._load_image(rel_path)
        if self.transform:
            img = self.transform(img)
        item_id = self.img_to_item.get(rel_path, "unknown")
        label = self.item_to_label.get(item_id, -1)
        return img, item_id, label, rel_path


# ─────────────────────────────────────────────
#  CLIP-compatible transforms
# ─────────────────────────────────────────────

def get_clip_transform(image_size: int = 224, augment: bool = False):
    """
    CLIP expects images normalised with specific mean/std.
    For training we optionally add light augmentations.
    """
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ─────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────

def build_dataloaders(
    dataset_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    use_gt_bbox: bool = True,
):
    """
    Returns a dict: {'train': DataLoader, 'query': DataLoader, 'gallery': DataLoader}
    """
    root = Path(dataset_root)
    partition_file   = root / "list_eval_partition.txt"
    description_file = root / "list_description_inshop.txt"
    bbox_file        = root / "list_bbox_inshop.txt"
    img_root         = root  # images are at root/img/...

    splits      = parse_eval_partition(str(partition_file))
    img_to_item = parse_item_ids(str(description_file))
    img_to_bbox = parse_bboxes(str(bbox_file)) if bbox_file.exists() else {}

    loaders = {}
    for split in ("train", "query", "gallery"):
        augment = (split == "train")
        ds = DeepFashionDataset(
            img_root=str(img_root),
            image_paths=splits[split],
            img_to_item=img_to_item,
            img_to_bbox=img_to_bbox,
            transform=get_clip_transform(image_size, augment=augment),
            use_gt_bbox=use_gt_bbox,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        print(f"[Dataset] {split:8s}: {len(ds):6,d} images | "
              f"{len(ds.item_ids):5,d} unique items")

    return loaders
