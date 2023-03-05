import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score


class ResNet:
    def __init__(self, train_data: tuple, val_data: tuple, test_data: tuple, input_shape=(48, 48, 1)):
        self._train = train_data
        self._val = val_data
        self._test = test_data
        self._input_shape = input_shape
        self._callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, verbose=1),
                           keras.callbacks.ModelCheckpoint(filepath='cleared_3/model_{epoch}', save_best_only=True,
                                                           verbose=1, monitor='val_accuracy')]
        self._labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self._model = self._create_model()

    def _create_model(self):
        inputs = keras.Input(shape=self._input_shape, name="img")
        x = layers.Conv2D(48, 3, activation="relu")(inputs)
        x = layers.Conv2D(96, 3, activation="relu")(x)
        block_1_output = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(96, 3, activation="relu", padding="same")(block_1_output)
        x = layers.Conv2D(96, 3, activation="relu", padding="same")(x)
        x = layers.add([x, block_1_output])
        block_2_output = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(96, 3, activation="relu", padding="same")(block_2_output)
        x = layers.Conv2D(96, 3, activation="relu", padding="same")(x)
        x = layers.add([x, block_2_output])
        block_3_output = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(96, 3, activation="relu")(block_3_output)
        x = layers.Flatten()(x)
        x = layers.Dense(384, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(7, activation='softmax')(x)

        return keras.Model(inputs, outputs, name="face_recognition_res_net")

    def load_model(self, path: str, custom_objects=None):
        self._model = keras.models.load_model(path, custom_objects=custom_objects)

    def load_weights(self, path: str, custom_objects=None):
        self._model.set_weights(keras.models.load_model(path, custom_objects=custom_objects).get_weights())

    def summary(self):
        self._model.summary()

    def compile(self):
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        # opt = keras.optimizers.SGD(learning_rate=0.0001)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        self._model.compile(optimizer=opt, loss=loss, metrics=metrics)

    def fit(self):
        return self._model.fit(self._train[0], self._train[1], batch_size=32, epochs=100,
                               validation_data=self._val, callbacks=self._callbacks)

    def evaluate(self):
        prediction = tf.argmax(self._model.predict(self._test[0]), axis=-1)
        real = tf.argmax(self._test[1], axis=-1)
        precision = precision_score(real, prediction, average='weighted')
        recall = recall_score(real, prediction, average='weighted')
        f1score = f1_score(real, prediction, average='weighted')
        accuracy = accuracy_score(real, prediction)
        bal_accuracy = balanced_accuracy_score(real, prediction)
        return self._model.evaluate(self._test[0], self._test[1]),\
               {'precision': precision, 'recall': recall, 'f1_score': f1score, 'acc': accuracy, 'bal_acc': bal_accuracy}

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

    def plot_model(self):
        plot_model(self._model, to_file='model.png', show_layer_names=False, show_layer_activations=True, show_shapes=True)
