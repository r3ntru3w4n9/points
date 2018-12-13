import os

import keras
import keras.backend as K
import numpy as np
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical

import models
import provider


def main(epochs=300,
         learning_rate=1e-3,
         num_points=1024,
         batch_size=64):
    K.clear_session()
    if not os.path.exists('weights'):
        os.makedirs('weights')
    train_files = provider.getDataFiles(
        './data/modelnet40_ply_hdf5_2048/train_files.txt')
    test_files = provider.getDataFiles(
        './data/modelnet40_ply_hdf5_2048/test_files.txt')

    model, _ = models.Classifier(points=num_points)

    classifier = Model(inputs=model.inputs,
                       outputs=[model.outputs[0]])

    classifier.compile(optimizer='adam', loss=[
                       'sparse_categorical_crossentropy'], metrics=['accuracy'])

    train_file_num = np.arange(0, len(train_files))

    data = []
    label = []

    for file_num in train_file_num:
        current_data, current_label = provider.loadDataFile(
            train_files[file_num])
        current_data = current_data[:, :num_points, :]

        data.append(current_data)
        label.append(current_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    test_file_num = np.arange(0, len(test_files))

    test_data = []
    test_label = []

    for file_num in test_file_num:
        current_data, current_label = provider.loadDataFile(
            test_files[file_num])
        current_data = current_data[:, :num_points, :]

        test_data.append(current_data)
        test_label.append(current_label)

    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)

    ModelCheckPoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('weights', 'best.{epoch:02d}.hdf5'),
        save_best_only=True)

    TensorBoard = keras.callbacks.TensorBoard()

    EarlyStopping = keras.callbacks.EarlyStopping(
        monitor='val_acc', patience=10)

    classifier.fit(x=data, y=label, batch_size=batch_size,
                   epochs=epochs, callbacks=[
                       ModelCheckPoint, TensorBoard, EarlyStopping
                   ], validation_data=(test_data, test_label))
    K.clear_session()


def generate(generator, num_maps, latent_size, num_cls=40):
    noise = np.random.randn(num_maps, latent_size)
    cls_noise = np.array([i for i in range(0, num_cls)]*10)
    return generator.predict([noise, cls_noise])


if __name__ == "__main__":
    main(300, num_points=1024)
