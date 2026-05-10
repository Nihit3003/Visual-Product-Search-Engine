"""
Improved DeepFashion In-Shop Dataset

Upgrades:
- safer image loading
- GT bbox support
- stronger augmentations
- hard-negative stability
- ViT-L compatible preprocessing
"""

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torchvision import transforms


# =========================================================
# PARSERS
# =========================================================

def parse_eval_partition(partition_file: str):

    splits = {
        "train": [],
        "query": [],
        "gallery": [],
    }

    img_to_item = {}

    with open(partition_file, "r") as f:

        lines = f.readlines()[2:]

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 3:
            continue

        img_path = parts[0]

        if img_path.startswith("img/"):

            img_path = img_path[4:]

        item_id = parts[1]

        split = parts[2].lower()

        img_to_item[img_path] = item_id

        if split in splits:

            splits[split].append(img_path)

    return splits, img_to_item


def parse_bboxes(bbox_file: str):

    img_to_bbox = {}

    with open(bbox_file, "r") as f:

        lines = f.readlines()[2:]

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 7:
            continue

        img_path = parts[0]

        if img_path.startswith("img/"):

            img_path = img_path[4:]

        try:

            x1 = int(parts[3])
            y1 = int(parts[4])
            x2 = int(parts[5])
            y2 = int(parts[6])

            img_to_bbox[img_path] = (
                x1,
                y1,
                x2,
                y2
            )

        except Exception:

            continue

    return img_to_bbox


# =========================================================
# CATEGORY
# =========================================================

def infer_category(rel_path: str):

    p = rel_path.lower()

    if any(k in p for k in [
        "tee",
        "shirt",
        "top",
        "blouse",
        "tank",
    ]):

        return "top"

    if any(k in p for k in [
        "pant",
        "short",
        "jean",
        "legging",
        "skirt",
    ]):

        return "bottom"

    if "dress" in p:

        return "dress"

    return "unknown"


# =========================================================
# BBOX CROP
# =========================================================

def bbox_crop(
    image,
    bbox,
    pad=0.08
):

    x1, y1, x2, y2 = bbox

    w, h = image.size

    px = int((x2 - x1) * pad)
    py = int((y2 - y1) * pad)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)

    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    if x2 <= x1 or y2 <= y1:

        return image

    return image.crop((
        x1,
        y1,
        x2,
        y2
    ))


# =========================================================
# MAIN DATASET
# =========================================================

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

        self.img_to_category = {

            p: infer_category(p)

            for p in image_paths
        }

        self.item_ids = sorted(
            list(set(
                img_to_item[p]
                for p in image_paths
            ))
        )

        self.item_to_label = {

            item_id: idx

            for idx, item_id
            in enumerate(self.item_ids)
        }

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def _load_image(
        self,
        rel_path
    ):

        full_path = (
            self.img_root /
            rel_path
        )

        try:

            image = Image.open(
                full_path
            ).convert("RGB")

        except Exception:

            return Image.new(
                "RGB",
                (224, 224),
                color=(255, 255, 255)
            )

        if (
            self.use_gt_bbox
            and
            rel_path in self.img_to_bbox
        ):

            try:

                image = bbox_crop(
                    image,
                    self.img_to_bbox[rel_path]
                )

            except Exception:

                pass

        return image

    # =====================================================
    # LEN
    # =====================================================

    def __len__(self):

        return len(self.image_paths)

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(
        self,
        idx
    ):

        rel_path = self.image_paths[idx]

        image = self._load_image(
            rel_path
        )

        if self.transform:

            image = self.transform(
                image
            )

        item_id = self.img_to_item[
            rel_path
        ]

        label = self.item_to_label[
            item_id
        ]

        category = self.img_to_category[
            rel_path
        ]

        return (
            image,
            item_id,
            label,
            rel_path,
            category
        )


# =========================================================
# HARD NEGATIVE DATASET
# =========================================================

class HardNegTripletDataset(Dataset):

    def __init__(
        self,
        rows,
        img_root,
        bbox_map,
        hard_neg_pool,
        transform,
    ):

        self.img_root = Path(img_root)

        self.bbox_map = bbox_map

        self.hard_neg_pool = hard_neg_pool

        self.transform = transform

        groups = defaultdict(list)

        for r in rows:

            groups[r["item_id"]].append(
                r["image_name"]
            )

        self.items = [

            (iid, imgs)

            for iid, imgs in groups.items()

            if len(imgs) >= 2
        ]

    # =====================================================
    # LOAD
    # =====================================================

    def _load(
        self,
        name
    ):

        p = self.img_root / name

        try:

            image = Image.open(
                p
            ).convert("RGB")

        except Exception:

            image = Image.new(
                "RGB",
                (224, 224),
                color=(255, 255, 255)
            )

        bbox = self.bbox_map.get(name)

        if bbox is not None:

            try:

                image = bbox_crop(
                    image,
                    bbox
                )

            except Exception:

                pass

        return self.transform(
            image
        )

    # =====================================================
    # LEN
    # =====================================================

    def __len__(self):

        return len(self.items)

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(
        self,
        idx
    ):

        item_id, imgs = self.items[idx]

        anc_name, pos_name = random.sample(
            imgs,
            2
        )

        hn_pool = self.hard_neg_pool.get(
            anc_name,
            []
        )

        if len(hn_pool) > 0:

            neg_name = random.choice(
                hn_pool
            )

        else:

            neg_name = anc_name

        anchor = self._load(
            anc_name
        )

        positive = self._load(
            pos_name
        )

        negative = self._load(
            neg_name
        )

        return (
            anchor,
            positive,
            negative
        )


# =========================================================
# TRANSFORMS
# =========================================================

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
                scale=(0.65, 1.0)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.2,
                hue=0.03,
            ),

            transforms.RandomPerspective(
                distortion_scale=0.15,
                p=0.2,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                CLIP_MEAN,
                CLIP_STD
            ),
        ])

    return transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            CLIP_MEAN,
            CLIP_STD
        ),
    ])


# =========================================================
# BUILDERS
# =========================================================

def build_dataloaders(
    dataset_root,
    batch_size=64,
    num_workers=4,
    image_size=224,
    use_gt_bbox=True,
):

    root = Path(dataset_root)

    partition_file = (
        root /
        "list_eval_partition.txt"
    )

    bbox_file = (
        root /
        "list_bbox_inshop.txt"
    )

    img_root = root / "img"

    splits, img_to_item = parse_eval_partition(
        str(partition_file)
    )

    img_to_bbox = parse_bboxes(
        str(bbox_file)
    )

    loaders = {}

    for split in [
        "train",
        "query",
        "gallery"
    ]:

        augment = (
            split == "train"
        )

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

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

        loaders[split] = loader

        print(
            f"[Dataset] {split:8s}: "
            f"{len(dataset):,} images"
        )

    return loaders
