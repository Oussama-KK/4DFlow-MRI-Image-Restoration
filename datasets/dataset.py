import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from datasets.preprocessing import (
    preprocess,
    preprocess_with_mask,
)


class MRIRestorationDataset(Dataset):
    """
    Paired MRI restoration dataset.
    Input, Target and Mask are spatially synchronized.

    CSV must contain columns:
        Input, Target

    Args:
        csv_path   : path to train.csv / val.csv / test.csv
        split      : "train", "val", or "test"
        crop_size  : spatial size of the output patch
        transform  : augmentation pipeline from transforms.py
        seed       : seed for reproducibility
    """

    def __init__(
        self,
        csv_path,
        split="train",
        crop_size=256,
        transform=None,
        seed=None,
    ):
        assert split in ("train", "val", "test"), (
            f"Invalid split: {split}"
        )

        # Load CSV
        self.df = pd.read_csv(csv_path)

        self.split = split
        self.crop_size = crop_size

        # Augmentations only during training
        self.transform = transform if split == "train" else None

        # Check required CSV columns
        required_cols = {"Input", "Target"}
        missing = required_cols - set(self.df.columns)

        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}"
            )

        # generate synchronized crops
        self._master_rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _make_synced_rngs(self):
        """
        Create two NumPy random generators using
        the same seed.

        This guarantees that Input and Target
        receive the same random crop.
        """

        seed = int(
            self._master_rng.integers(
                0,
                2**31 - 1
            )
        )

        rng_input = np.random.default_rng(seed)
        rng_target = np.random.default_rng(seed)

        return rng_input, rng_target

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        input_path = row["Input"]
        target_path = row["Target"]

        # Create synchronized random generators

        if self.split == "train":

            rng_input, rng_target = (
                self._make_synced_rngs()
            )

        else:

            rng_input = None
            rng_target = None


        # Preprocess Input + create mask

        input_result = preprocess_with_mask(
            input_path,
            split=self.split,
            crop_size=self.crop_size,
            rng=rng_input,
        )

        if input_result is None:

            print(
                f"[WARNING] Skipping image (too small): "
                f"{input_path}"
            )

            return None

        input_image, mask = input_result


        # Preprocess Target

        target_image = preprocess(
            target_path,
            split=self.split,
            crop_size=self.crop_size,
            rng=rng_target,
        )

        if target_image is None:

            print(
                f"[WARNING] Skipping image (too small): "
                f"{target_path}"
            )

            return None



        # Apply synchronized augmentation

        if self.transform is not None:

            # Remove channel dimension (1,H,W) -> (H,W)

            inp = input_image[0]
            tgt = target_image[0]
            msk = mask[0]


            # Apply the same transform  

            inp, tgt, msk = self.transform(
                inp,
                tgt,
                msk
            )

            # Add channel dimension back (H,W) -> (1,H,W)

            input_image = inp[np.newaxis, ...]
            target_image = tgt[np.newaxis, ...]
            mask = msk[np.newaxis, ...]


        # Return sample

        return {
            "Input": input_image.astype(np.float32),
            "Target": target_image.astype(np.float32),
            "Mask": mask.astype(np.float32),

            "input_path": input_path,
            "target_path": target_path,
        }