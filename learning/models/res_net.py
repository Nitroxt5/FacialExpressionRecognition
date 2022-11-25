import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ResNet:
    def __init__(self, train_data: tuple, val_data: tuple, test_data: tuple, input_shape=(48, 48, 1)):
        self._train = train_data
        self._val = val_data
        self._test = test_data
        self._input_shape = input_shape
        self._callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, verbose=1),
                           keras.callbacks.ModelCheckpoint(filepath='new_dataset_nn/model_{epoch}', save_best_only=True,
                                                           verbose=1, monitor='val_accuracy')]
        self._labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self._create_model()

    def _create_model(self):
        inputs = keras.Input(shape=self._input_shape, name="img")
        x = layers.Conv2D(48, 3, activation="relu")(inputs)
        x = layers.Conv2D(96, 3, activation="relu")(x)
        block_1_output = layers.MaxPooling2D(3)(x)

        x = layers.Conv2D(96, 3, activation="relu", padding="same")(block_1_output)
        x = layers.Conv2D(96, 3, activation="relu", padding="same")(x)
        block_2_output = layers.add([x, block_1_output])

        x = layers.Conv2D(96, 3, activation="relu", padding="same")(block_2_output)
        x = layers.Conv2D(96, 3, activation="relu", padding="same")(x)
        block_3_output = layers.add([x, block_2_output])

        x = layers.Conv2D(96, 3, activation="relu")(block_3_output)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(384, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(7, activation='softmax')(x)

        self._model = keras.Model(inputs, outputs, name="face_recognition_res_net")

    def summary(self):
        self._model.summary()

    def compile(self):
        self._model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit(self):
        self._model.fit(self._train[0], self._train[1], batch_size=48, epochs=300,
                        validation_data=self._val, callbacks=self._callbacks)

    def evaluate(self):
        print(self._model.evaluate(self._test[0], self._test[1]))

    def save(self):
        self._model.save('test_model')

    def create_confusion_matrix(self, normalize: [str, None]):
        prediction = tf.argmax(self._model.predict(self._test[0]), axis=-1)
        real = tf.argmax(self._test[1], axis=-1)
        result = confusion_matrix(real, prediction, normalize=normalize)
        matrix_plot = ConfusionMatrixDisplay(result, display_labels=self._labels)
        fig, ax = plt.subplots(figsize=(8, 8))
        matrix_plot.plot(ax=ax)
        plt.show()
        return result
