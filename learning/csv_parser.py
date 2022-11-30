import csv
import logging
import numpy as np
import os
from PIL import Image
from random import shuffle


class Parser:
    def __init__(self, project_path: str):
        self._project_path = project_path
        self._data_folder = 'challenges-in-representation-learning-facial-expression-recognition-challenge'
        self._img_folder = 'images'
        self._data_root = os.path.join(self._project_path, self._data_folder)
        self._img_root = os.path.join(self._project_path, self._img_folder)
        self._size = 48
        self._shape = (self._size, self._size)
        self._labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'ND']

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
            img.save(os.path.join(img_root, f'{self._labels[emotion]}_img_{i}.bmp'))

    def sample_to_csv(self, filename: str, data: list[dict], mode='w', fieldnames=('emotion', 'pixels')):
        with open(os.path.join(self._data_root, filename), newline='', mode=mode) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if mode == 'w':
                writer.writeheader()
            writer.writerows(data)

    def split_csv(self, filename: str, sample_type: str):
        split_data: list[list[dict]] = [[] for _ in range(7)]
        with open(os.path.join(self._data_root, filename), newline='') as csvfile:
            data = csv.DictReader(csvfile, delimiter=',')
            for i, row in enumerate(data, start=1):
                if i % 10000 == 0:
                    logging.info(f'Read {i} rows from file: {filename}')
                split_data[int(row['emotion'])].append(row)
        for i, row in enumerate(split_data):
            filtered_row = [el for el in row if el['Usage'] == sample_type]
            self.sample_to_csv(f'{self._labels[i]}_{sample_type}.csv', filtered_row,
                               fieldnames=('emotion', 'Usage', 'pixels'))

    def count_data(self, filename: str):
        input_data, output_data = self.csv_with_usage_to_sample(filename)
        counts: dict[str, dict[str, int]] = {}
        for key, value in output_data.items():
            counts[key] = {self._labels[i]: value.count(i) for i in range(7)}
        return counts

    def get_uniform_train_sample(self, filename: str, usage=True):
        if usage:
            input_data, output_data = self.csv_with_usage_to_sample(filename)
            input_data, output_data = input_data['Training'], output_data['Training']
        else:
            input_data, output_data = self.csv_no_usage_to_sample(filename)
        min_count = min([output_data.count(i) for i in range(7)])
        data = list(zip(input_data, output_data))
        shuffle(data)
        ret_input_data, ret_output_data = [], []
        counts = [0 for _ in range(7)]
        for inp, outp in data:
            if counts[outp] == min_count:
                continue
            counts[outp] += 1
            ret_input_data.append(inp)
            ret_output_data.append(outp)
        return ret_input_data, ret_output_data
