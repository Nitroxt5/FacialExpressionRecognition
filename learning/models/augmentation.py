from keras.models import Sequential
from keras import layers
import logging
import numpy as np


class Augmentator:
    def __init__(self):
        self._rot_aug = layers.RandomRotation(factor=1/12)
        self._trans_aug = layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
        self._rot_trans_aug = Sequential([
            layers.RandomRotation(factor=1/12),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
        ])

    def perform_augmentation(self, data, name='rot'):
        augmentator = self._get_augmentator(name)
        augmented_data = np.zeros(shape=data.shape)
        for i, img in enumerate(data, start=1):
            if i % 10000 == 0:
                logging.info(f'Augmented {i} rows')
            augmented_data[i - 1] = augmentator(img)
        return augmented_data

    def _get_augmentator(self, name: str):
        if name == 'rot':
            return self._rot_aug
        if name == 'trans':
            return self._trans_aug
        return self._rot_trans_aug
