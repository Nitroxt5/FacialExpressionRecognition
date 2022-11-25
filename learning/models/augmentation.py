from keras.models import Sequential
from keras import layers
import logging
import numpy as np


class Augmentator:
    def __init__(self):
        self._img_augmentation = Sequential(
            [
                # layers.RandomFlip(),
                # layers.RandomRotation(factor=0.2),
                layers.RandomRotation(factor=1/12),
                # layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
            ]
        )

    def perform_augmentation(self, data):
        augmented_data = np.zeros(shape=data.shape)
        for i, img in enumerate(data, start=1):
            if i % 10000 == 0:
                logging.info(f'Augmented {i} rows')
            augmented_data[i - 1] = self._img_augmentation(img)
        return augmented_data
