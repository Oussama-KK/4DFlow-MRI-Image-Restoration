import random
import numpy as np


class RandomHorizontalFlip:
    """
    Randomly flip Input, Target and Mask horizontally.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, input_img, target_img, mask):

        if random.random() < self.probability:

            input_img = np.flip(
                input_img,
                axis=1
            ).copy()

            target_img = np.flip(
                target_img,
                axis=1
            ).copy()

            mask = np.flip(
                mask,
                axis=1
            ).copy()

        return input_img, target_img, mask


class RandomVerticalFlip:
    """
    Randomly flip Input, Target and Mask vertically.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, input_img, target_img, mask):

        if random.random() < self.probability:

            input_img = np.flip(
                input_img,
                axis=0
            ).copy()

            target_img = np.flip(
                target_img,
                axis=0
            ).copy()

            mask = np.flip(
                mask,
                axis=0
            ).copy()

        return input_img, target_img, mask


class RandomRotation90:
    """
    Randomly rotate Input, Target and Mask
    by 0, 90, 180 or 270 degrees.
    """

    def __init__(self):
        pass

    def __call__(self, input_img, target_img, mask):

        k = random.randint(0, 3)

        input_img = np.rot90(
            input_img,
            k
        ).copy()

        target_img = np.rot90(
            target_img,
            k
        ).copy()

        mask = np.rot90(
            mask,
            k
        ).copy()

        return input_img, target_img, mask


class Compose:
    """
    Apply multiple transforms sequentially.

    All spatial transformations are applied
    identically to:

        Input
        Target
        Mask
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(
        self,
        input_img,
        target_img,
        mask
    ):

        for transform in self.transforms:

            input_img, target_img, mask = transform(
                input_img,
                target_img,
                mask
            )

        return input_img, target_img, mask



