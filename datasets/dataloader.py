from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from datasets.dataset import MRIRestorationDataset
from datasets.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
)


def collate_skip_none(batch):
    """
    Remove invalid samples (None) from a batch.

    If every sample in the batch is invalid,
    return None so the training loop can skip it.
    """

    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    return default_collate(batch)


def build_transform(augment: bool):
    """
    Returns the training augmentation pipeline,
    or None if augment=False.
    """

    if not augment:
        return None

    return Compose([
        RandomHorizontalFlip(probability=0.5),
        RandomVerticalFlip(probability=0.5),
        RandomRotation90(),
    ])


def get_dataset(dataset_cfg: dict, split: str):
    """
    Build a MRIRestorationDataset for the given split.

    Expected dataset_cfg keys:
        train_csv
        val_csv
        test_csv
        crop_size
        seed
        augment
    """

    csv_key = f"{split}_csv"

    if csv_key not in dataset_cfg:
        raise KeyError(
            f"Dataset config missing '{csv_key}'"
        )

    crop_size = dataset_cfg.get("crop_size", 256)
    seed = dataset_cfg.get("seed", 42)

    augment = (
        dataset_cfg.get("augment", True)
        if split == "train"
        else False
    )

    transform = build_transform(augment)

    return MRIRestorationDataset(
        csv_path=dataset_cfg[csv_key],
        split=split,
        crop_size=crop_size,
        transform=transform,
        seed=seed,
    )


def get_dataloader(
    dataset_cfg: dict,
    train_cfg: dict,
    split: str
):
    """
    Build a DataLoader for the given split.

    Expected train_cfg keys:
        batch_size
        num_workers
        pin_memory
    """

    dataset = get_dataset(dataset_cfg, split)

    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", True)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        collate_fn=collate_skip_none,
    )


def get_dataloaders(
    dataset_cfg: dict,
    train_cfg: dict
):
    """
    Build train, validation, and test dataloaders.
    """

    return {
        split: get_dataloader(
            dataset_cfg,
            train_cfg,
            split
        )
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":


    dataset_cfg = {
        "train_csv": "data/train.csv",
        "val_csv": "data/val.csv",
        "test_csv": "data/test.csv",
        "crop_size": 256,
        "seed": 42,
        "augment": True,
    }

    train_cfg = {
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
    }

    loaders = get_dataloaders(
        dataset_cfg,
        train_cfg
    )

    for split, loader in loaders.items():

        batch = next(iter(loader))

        if batch is None:
            print(f"[{split}] All samples in the first batch were invalid.")
            continue

        print(
            f"[{split}] "
            f"batches: {len(loader)} | "
            f"input {batch['Input'].shape} | "
            f"target {batch['Target'].shape}"
        )