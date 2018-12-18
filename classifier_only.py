import argparse
import os

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical

import loader
import models
import provider

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs, default=300')
    parser.add_argument('--points', type=int, default=1024,
                        help='number of points per sample, default=1024')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate, default=1e-3')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of sets per batch, default=64')
    parser.add_argument('--train_files',
                        default='./data/modelnet40_ply_hdf5_2048/train_files.txt')
    parser.add_argument('--test_files',
                        default='./data/modelnet40_ply_hdf5_2048/test_files.txt')
    parser.add_argument('--plot', type=bool, default=False,
                        help='save training history')
    parser.add_argument('--save_history', type=bool, default=False,
                        help='save history dictionary')

    args = parser.parse_args()

    if not os.path.exists('weights'):
        os.makedirs('weights')

    train_files = provider.getDataFiles(args.train_files)
    test_files = provider.getDataFiles(args.test_files)

    (data, label), (test_data, test_label) = loader.load_data(
        train_files, test_files, args.points)

    model, _ = models.Classifier(points=args.points)

    classifier = Model(inputs=model.inputs,
                       outputs=[model.outputs[0]])

    classifier.compile(optimizer='adam', loss=[
                       'sparse_categorical_crossentropy'], metrics=['accuracy'])

    ModelCheckPoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('weights', 'best.{epoch:03d}.hdf5'),
        save_best_only=True)

    TensorBoard = keras.callbacks.TensorBoard()

    EarlyStopping = keras.callbacks.EarlyStopping(
        monitor='val_acc', patience=10)

    loss = classifier.fit(x=data,
                          y=label,
                          batch_size=args.batch_size,
                          epochs=args.epochs,
                          callbacks=[ModelCheckPoint,
                                     TensorBoard,
                                     EarlyStopping],
                          validation_data=(test_data, test_label))

    if args.save_history:
        np.save(file='./history', arr=loss.history)

    if args.plot:
        for item in loss.history.keys():
            plt.plot(loss.history[item], label=item)
        plt.legend()
        plt.savefig('./loss_metrics.jpg')

    K.clear_session()
