import os
import sys
import logging
import numpy as np
from tensorflow import keras
from learning.models.augmentation import Augmentator
from learning.csv_parser import Parser
from learning.models.res_net import ResNet


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler("log.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ])
    np.set_printoptions(precision=2, floatmode='fixed')
    parser = Parser(os.getcwd())
    augment = Augmentator()

    input_data, output_data = parser.csv_with_usage_to_sample('icml_face_data.csv')
    aug_input_data, aug_output_data = parser.csv_no_usage_to_sample('augmented_data_rot_30.csv')

    # transform data into images
    # parser.sample_to_images('augmented', aug_input_data, aug_output_data)

    input_data['Training'] += aug_input_data
    output_data['Training'] += aug_output_data

    input_data_np = {key: np.array(value).reshape((len(value), 48, 48, 1)) / 255 for key, value in input_data.items()}
    output_data_np = {key: keras.utils.to_categorical(np.array(value), 7) for key, value in output_data.items()}

    # augment data and write it to a csv
    # augmented_data_np = augment.perform_augmentation(input_data_np['Training'])
    # augmented_data_np = augmented_data_np.reshape((augmented_data_np.shape[0], 48 * 48)) * 255
    # augmented_data = augmented_data_np.tolist()
    # for i in range(len(augmented_data)):
    #     augmented_data[i] = ' '.join(list(map(to_int_and_str, augmented_data[i])))
    # augmented_data_to_csv = [{'emotion': output_data['Training'][i],
    #                           'pixels': augmented_data[i]} for i in range(len(output_data['Training']))]
    # parser.sample_to_csv('augmented_data_rot_30.csv', augmented_data_to_csv)

    res_net = ResNet((input_data_np['Training'], output_data_np['Training']),
                     (input_data_np['PublicTest'], output_data_np['PublicTest']),
                     (input_data_np['PrivateTest'], output_data_np['PrivateTest']))

    # res_net.summary()

    # load saved models
    # res_net._model = keras.models.load_model(f'{os.getcwd()}\\old_dataset_nn\\model_3')
    # res_net._model = keras.models.load_model(f'{os.getcwd()}\\new_dataset_nn_30_rot\\model_69')

    res_net.compile()
    res_net.fit()
    res_net.evaluate()
    res_net.save()
    matrix = res_net.create_confusion_matrix('pred')
    print(matrix)


def to_int_and_str(el):
    return str(int(el))


if __name__ == '__main__':
    main()


