import os
import time

import keras
import keras.backend as K
import numpy as np
from keras.layers import *
from keras.models import *

import models
import provider


def main(epochs=300, num_points=1024, batch_size=64, latent_size=200, num_cls=40):
    K.clear_session()
    if not os.path.exists("weights"):
        os.makedirs("weights")
    train_files = provider.getDataFiles(
        "./data/modelnet40_ply_hdf5_2048/train_files.txt"
    )
    test_files = provider.getDataFiles("./data/modelnet40_ply_hdf5_2048/test_files.txt")

    classifier, biases = models.Classifier(points=num_points)
    generator = models.Generator(latent_size, points=num_points)
    classifier.trainable = False
    combined_output = classifier(generator.outputs + biases)
    combined = Model(inputs=generator.inputs + biases, outputs=combined_output)
    print("Combined model:")
    combined.summary()

    Doptim = keras.optimizers.Adam()

    class_label = K.placeholder()
    real_label = K.placeholder()

    """
    This is simply a work-around, 
    I'm on a legacy system with Keras=2.0.2, tensorflow=1.2.0
    Feel free to simply use the classifier and combined model to train the acgan model
    
    Note that if you're going to use the Model API to train,
    you have to use .train_on_batch instead of .fit
    """

    classifier_training_fn = K.function(
        inputs=[class_label, real_label] + classifier.inputs,
        outputs=[],
        updates=Doptim.get_updates(
            loss=(
                K.categorical_crossentropy(
                    target=class_label, output=classifier.get_output_at(0)[0]
                )
                + K.binary_crossentropy(
                    target=real_label, output=classifier.get_output_at(0)[1]
                )
            ),
            params=classifier.trainable_weights,
        ),
    )

    classless_training_fn = K.function(
        inputs=[real_label] + combined.inputs,
        outputs=[],
        updates=Doptim.get_updates(
            loss=K.binary_crossentropy(target=real_label, output=combined.output[1]),
            params=classifier.trainable_weights,
        ),
    )

    classifier_fn = K.function(inputs=classifier.inputs, outputs=classifier.outputs)

    Goptim = keras.optimizers.Adam()
    combined_training_fn = K.function(
        inputs=[class_label, real_label] + combined.inputs,
        outputs=[],
        updates=Goptim.get_updates(
            loss=(
                K.categorical_crossentropy(
                    target=class_label, output=combined.outputs[0]
                )
                + K.binary_crossentropy(target=real_label, output=combined.output[1])
            ),
            params=generator.trainable_weights,
        ),
    )
    generator_fn = K.function(inputs=generator.inputs, outputs=generator.outputs)

    """
    Arr... I hate legacy systems
    """

    train_file_num = np.arange(len(train_files))

    for epoch in range(1, epochs + 1):
        print("epoch: {}/{}".format(epoch, epochs))
        tm = time.time()

        for file_num in train_file_num:
            current_data, current_label = provider.loadDataFile(train_files[file_num])

            total_len = len(current_label)
            assert len(current_data) == len(current_label)

            current_data = current_data[:, :num_points, :]
            current_data, current_label, _ = provider.shuffle_data(
                current_data, np.squeeze(current_label)
            )
            current_label = np.expand_dims(current_label, axis=-1)

            for index in range(0, total_len - batch_size, batch_size):

                x_train = current_data[index : index + batch_size]
                y_train = current_label[index : index + batch_size]
                aux_y = np.expand_dims(0.95 * np.ones(batch_size), -1)

                classifier_training_fn([y_train, aux_y, x_train])

                noise = np.random.randn(batch_size, latent_size)
                rdn_cls = np.expand_dims(
                    np.random.randint(low=0, high=num_cls - 1, size=batch_size), -1
                )
                x_train = generator_fn([noise, rdn_cls])
                aux_y = np.expand_dims(np.zeros(batch_size), -1)

                classless_training_fn([aux_y, noise, rdn_cls])

                noise = np.random.randn(batch_size, latent_size)
                rdn_cls = np.expand_dims(
                    np.random.randint(low=0, high=num_cls - 1, size=batch_size), -1
                )

                trick = np.expand_dims(0.95 * np.ones(batch_size), -1)

                combined_training_fn([trick, rdn_cls, noise, rdn_cls])
        classifier.save(filepath=os.path.join("weights", "classifier"))
        generator.save(filepath=os.path.join("weights", "generator"))
        combined.save(filepath=os.path.join("weights", "combined"))
        print("took {} seconds".format(time.time() - tm))


def generate(generator_fn, num_maps, latent_size, num_cls=40):
    noise = np.random.randn(num_maps, latent_size)
    cls_noise = np.array([i for i in range(0, num_cls)] * 10)
    return generator.predict([noise, cls_noise])


def test(classifier, test_files, batch_size):
    test_file_num = np.arange(0, len(test_files))

    x_test = []
    y_test = []

    for file_num in test_file_num:
        current_data, current_label = provider.loadDataFile(test_files[file_num])
        current_data = current_data[:, :num_points, :]
        x_test.append(current_data)
        y_test.append(current_label)

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    classifier.evaluate(x=x_test, y=y_test, batch_size=batch_size)


if __name__ == "__main__":
    main(300, num_points=1024)
