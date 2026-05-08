"""
Improved DeepFashion In-Shop Dataset

Enhancements:
- robust bbox handling
- category-aware metadata
- hard-negative friendly sampling
- improved augmentation
- safer image loading
- better SupCon batch construction
"""

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler
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

        if len(parts) < 6:
            continue

        img_path = parts[0]

        try:

            bbox = [int(x) for x in parts[2:6]]

            img_to_bbox[img_path] = bbox

        except Exception:
            continue

    return img_to_bbox


# =========================================================
# CATEGORY PARSER
# =========================================================

def infer_category(rel_path: str):

    p = rel_path.lower()

    if any(k in p for k in [
        "tee",
        "shirt",
        "top",
        "blouse",
        "tank"
    ]):
        return "top"

    if any(k in p for k in [
        "pant",
        "short",
        "jean",
        "legging",
        "skirt"
    ]):
        return "bottom"

    if "dress" in p:
        return "dress"

    return "unknown"


# =========================================================
# DATASET
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

        # -----------------------------------------
        # labels
        # -----------------------------------------

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

        # -----------------------------------------
        # categories
        # -----------------------------------------

        self.img_to_category = {
            p: infer_category(p)
            for p in image_paths
        }

        # -----------------------------------------
        # category groups
        # -----------------------------------------

        self.category_to_indices = defaultdict(list)

        for idx, p in enumerate(image_paths):

            cat = self.img_to_category[p]

            self.category_to_indices[cat].append(idx)

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def _load_image(self, rel_path):

        full_path = self.img_root / rel_path

        try:

            image = Image.open(full_path).convert("RGB")

        except Exception:

            # corrupted image fallback
            image = Image.new(
                "RGB",
                (224, 224),
                color=(255, 255, 255)
            )

            return image

        # -------------------------------------------------
        # GT bbox crop
        # -------------------------------------------------

        if self.use_gt_bbox and rel_path in self.img_to_bbox:

            try:

                x1, y1, x2, y2 = self.img_to_bbox[rel_path]

                w, h = image.size

                # clamp

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))

                x2 = max(1, min(x2, w))
                y2 = max(1, min(y2, h))

                # invalid bbox

                if x2 <= x1 or y2 <= y1:
                    return image

                # reject tiny crops

                bw = x2 - x1
                bh = y2 - y1

                if bw < 20 or bh < 20:
                    return image

                image = image.crop((x1, y1, x2, y2))

            except Exception:

                return image

        return image

    # =====================================================
    # LEN
    # =====================================================

    def __len__(self):

        return len(self.image_paths)

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, idx):

        rel_path = self.image_paths[idx]

        image = self._load_image(rel_path)

        if self.transform:

            image = self.transform(image)

        item_id = self.img_to_item[rel_path]

        label = self.item_to_label[item_id]

        category = self.img_to_category[rel_path]

        return (
            image,
            item_id,
            label,
            rel_path,
            category
        )


# =========================================================
# CLIP TRANSFORMS
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

            transforms.RandomHorizontalFlip(),

            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.15,
                hue=0.02,
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
# HARD NEGATIVE BALANCED SAMPLER
# =========================================================

class ClassBalancedSampler(Sampler):

    """
    Creates retrieval-friendly batches.

    Improvements:
    - class balancing
    - category-aware hard negatives
    - stronger SupCon learning
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

        self.class_to_category = {}

        for idx, rel_path in enumerate(
            dataset.image_paths
        ):

            item_id = dataset.img_to_item[rel_path]

            category = dataset.img_to_category[rel_path]

            self.class_to_indices[item_id].append(idx)

            self.class_to_category[item_id] = category

        self.classes = list(
            self.class_to_indices.keys()
        )

        # -----------------------------------------
        # category groupings
        # -----------------------------------------

        self.category_to_classes = defaultdict(list)

        for cls in self.classes:

            cat = self.class_to_category[cls]

            self.category_to_classes[cat].append(cls)

        self.batch_size = (
            classes_per_batch *
            samples_per_class
        )

    # =====================================================
    # ITER
    # =====================================================

    def __iter__(self):

        random.shuffle(self.classes)

        batch = []

        used_classes = set()

        for cls in self.classes:

            if cls in used_classes:
                continue

            used_classes.add(cls)

            indices = self.class_to_indices[cls]

            # -------------------------------------
            # positive sampling
            # -------------------------------------

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

            # -------------------------------------
            # category-aware hard negatives
            # -------------------------------------

            category = self.class_to_category[cls]

            same_cat_classes = [
                c for c in
                self.category_to_classes[category]
                if c != cls
            ]

            random.shuffle(same_cat_classes)

            for neg_cls in same_cat_classes[:2]:

                neg_indices = self.class_to_indices[neg_cls]

                batch.append(
                    random.choice(neg_indices)
                )

            # -------------------------------------
            # emit batch
            # -------------------------------------

            if len(batch) >= self.batch_size:

                batch = batch[:self.batch_size]

                yield from batch

                batch = []

    # =====================================================
    # LEN
    # =====================================================

    def __len__(self):

        return (
            len(self.classes)
            * self.samples_per_class
        )


# =========================================================
# BUILD DATALOADERS
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

    if bbox_file.exists():

        img_to_bbox = parse_bboxes(
            str(bbox_file)
        )

    else:

        img_to_bbox = {}

    loaders = {}

    for split in [
        "train",
        "query",
        "gallery"
    ]:

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

        # -----------------------------------------
        # train loader
        # -----------------------------------------

        if split == "train":

            sampler = ClassBalancedSampler(
                dataset,
                classes_per_batch=max(
                    8,
                    batch_size // 6
                ),
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

        # -----------------------------------------
        # eval loader
        # -----------------------------------------

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
