import os

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical

import models
import loader
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

    ModelCheckPoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('weights', 'best.{epoch:02d}.hdf5'),
        save_best_only=True)

    TensorBoard = keras.callbacks.TensorBoard()

    EarlyStopping = keras.callbacks.EarlyStopping(
        monitor='val_acc', patience=10)

    loss = classifier.fit(x=data,
                             y=label,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=[ModelCheckPoint,
                                        TensorBoard,
                                        EarlyStopping],
                             validation_data=(test_data, test_label))
    
    plt.gcf().clear()
    for item in loss.history.keys():
        plt.plot(loss.history[item],label=item)
    plt.legend()
    plt.savefig('./loss_metrics.jpg')

    K.clear_session()


if __name__ == "__main__":
    main(300, num_points=1024)
