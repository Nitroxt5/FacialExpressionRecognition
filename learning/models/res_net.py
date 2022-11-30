import tensorflow as tf
from tensorflow import keras
from keras import layers, backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ResNet:
    def __init__(self, train_data: tuple, val_data: tuple, test_data: tuple, input_shape=(48, 48, 1)):
        self._train = train_data
        self._val = val_data
        self._test = test_data
        self._input_shape = input_shape
        self._callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, verbose=1),
                           keras.callbacks.ModelCheckpoint(filepath='new2_4/model_{epoch}', save_best_only=True,
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

    @staticmethod
    def f1_score(y_true, y_pred):
        def recall_m():
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            return TP / (Positives + K.epsilon())

        def precision_m():
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            return TP / (Pred_Positives + K.epsilon())

        precision, recall = precision_m(), recall_m()

        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def compile(self):
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        # opt = keras.optimizers.SGD(learning_rate=0.0001)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', self.f1_score, keras.metrics.Precision(), keras.metrics.Recall()]
        self._model.compile(optimizer=opt, loss=loss, metrics=metrics)

    def fit(self):
        return self._model.fit(self._train[0], self._train[1], batch_size=64, epochs=300,
                               validation_data=self._val, callbacks=self._callbacks)

    def evaluate(self):
        return self._model.evaluate(self._test[0], self._test[1])

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
