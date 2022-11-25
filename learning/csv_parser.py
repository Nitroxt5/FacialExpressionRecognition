import csv
import logging
import numpy as np
import os
from PIL import Image


class Parser:
    def __init__(self, project_path: str):
        self._project_path = project_path
        self._data_folder = 'challenges-in-representation-learning-facial-expression-recognition-challenge'
        self._img_folder = 'images'
        self._data_root = os.path.join(self._project_path, self._data_folder)
        self._img_root = os.path.join(self._project_path, self._img_folder)
        self._size = 48
        self._shape = (self._size, self._size)

    def csv_with_usage_to_sample(self, filename: str):
        x: dict[str, list[list[int]]] = {'Training': [], 'PublicTest': [], 'PrivateTest': []}
        y: dict[str, list[int]] = {'Training': [], 'PublicTest': [], 'PrivateTest': []}
        with open(os.path.join(self._data_root, filename), newline='') as csvfile:
            data = csv.DictReader(csvfile, delimiter=',')
            for i, row in enumerate(data, start=1):
                if i % 10000 == 0:
                    logging.info(f'Read {i} rows from file: {filename}')
                x[row['Usage']].append(list(map(int, row['pixels'].split(sep=' '))))
                y[row['Usage']].append(int(row['emotion']))
        return x, y

    def csv_no_usage_to_sample(self, filename: str):
        x: list[list[int]] = []
        y: list[int] = []
        with open(os.path.join(self._data_root, filename), newline='') as csvfile:
            data = csv.DictReader(csvfile, delimiter=',')
            for i, row in enumerate(data, start=1):
                if i % 10000 == 0:
                    logging.info(f'Read {i} rows from file: {filename}')
                x.append(list(map(int, row['pixels'].split(sep=' '))))
                try:
                    y.append(int(row.get('emotion')))
                except TypeError:
                    y.append(7)
        return x, y

    def sample_to_images(self, folder: str, pixels: list[list[int]], emotions: list[int]):
        img_root = os.path.join(self._img_root, folder)
        for i, (pixel_row, emotion) in enumerate(zip(pixels, emotions), start=1):
            if i % 10000 == 0:
                logging.info(f'Processed {i} rows')
            bitmap = np.ndarray(shape=self._shape, dtype=int, buffer=np.array(pixel_row))
            img = Image.fromarray(bitmap).convert('P')
            img.save(os.path.join(img_root, f'{emotion}_img_{i}.bmp'))

    def sample_to_csv(self, filename: str, data: list[dict]):
        fieldnames = ('emotion', 'pixels')
        with open(os.path.join(self._data_root, filename), newline='', mode='w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
