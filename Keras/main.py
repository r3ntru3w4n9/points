import argparse
import os

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical

import loader
import models

parser = argparse.ArgumentParser()
parser.add_argument('epochs', type=int, default=300,
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
parser.add_argument('--plot', action='store_true',
                    help='save training history')
parser.add_argument('--save_history', action='store_true',
                    help='save history dictionary')
rotation = parser.add_argument_group('rotation')
rotation.add_argument('--rotate', action='store_true',
                      help='apply rotation to data')
rotation.add_argument('--rotate_val', action='store_true',
                      help='apply rotation to validation data')
rotation.add_argument('--per_rotation', type=int, default=5,
                      help='how many epochs per rotation')
advanced = parser.add_argument_group('advanced')
advanced.add_argument('--residual', action='store_true',
                      help='whether to use the residual network')
advanced.add_argument('--separable', action='store_true',
                      help='replace `Conv2D` with `SeparableConv2D`')
advanced.add_argument('--ensemble', action='store_true',
                      help='a cluster of predictive models')
parser.add_argument('--cuda', type=str, default='0',
                    help='configure which cuda device to use')
args = parser.parse_args()


# setting GPU usage
config = tf.ConfigProto()
config.gpu_options.visible_device_list = args.cuda
set_session(tf.Session(config=config))

if not os.path.exists('weights'):
    os.makedirs('weights')

(data, label), (test_data, test_label) = loader.load_data(
    args.train_files, args.test_files,
    num_points=args.points,
    shuffle=False,
    rotate=args.rotate,
    rotate_val=args.rotate_val)

if args.residual:
    if args.separable:
        model, _ = models.SeparableResidual(points=args.points)
    else:
        model, _ = models.Residual(points=args.points)
else:
    if args.separable:
        model, _ = models.SeparableClassifier(points=args.points)
    else:
        model, _ = models.Classifier(points=args.points)

classifier = Model(inputs=model.inputs,
                   outputs=[model.outputs[0]])

classifier.compile(optimizer='adam', loss=[
    'sparse_categorical_crossentropy'], metrics=['accuracy'])

del model

ModelCheckPoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('weights', 'best.{epoch:03d}.hdf5'),
    save_best_only=True)

TensorBoard = keras.callbacks.TensorBoard()

EarlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_acc', patience=10)

if args.rotate:
    for epoch in range(1, args.epochs+1, args.per_rotation):
        (data, label), (test_data, test_label) = loader.load_data(
            args.train_files,
            args.test_files,
            args.points,
            rotate=args.rotate,
            rotate_val=args.rotate_val)
        print('epoch: {}/{}'.format(epoch, args.epochs))
        classifier.fit(x=data,
                       y=label,
                       batch_size=args.batch_size,
                       epochs=args.per_rotation,
                       validation_data=(test_data, test_label))
elif args.rotate_val:
    for epoch in range(1, args.epochs+1):
        (x_test, y_test) = loader.rotate_data(args.test_files)
        print('epoch: {}/{}'.format(epoch, args.epochs))
        classifier.fit(x=data,
                       y=label,
                       batch_size=args.batch_size,
                       epochs=args.per_rotation,
                       validation_data=(test_data, test_label))
else:
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

# on original data
(x_train, y_train), (x_test, y_test) = loader.load_data(
    args.train_files, args.test_files, args.points)
(loss, acc) = classifier.evaluate(x=x_train, y=y_train)
print()
print('training loss: {}, training accuracy: {}'.format(loss, acc))
(loss, acc) = classifier.evaluate(x=x_test, y=y_test)
print()
print('testing loss: {}, testing accuracy: {}'.format(loss, acc))

K.clear_session()
