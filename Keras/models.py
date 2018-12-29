import keras
import keras.backend as K
import numpy as np
from keras.layers import *
from keras.models import *


def Generator(latent_size, num_cls=40, points=1024):
    '''output format: (N, pints, 3)'''
    ipt = Input(shape=(latent_size,), name='Generator_Noise_Input')
    cls = Input(shape=(1,), dtype='int32', name='Generator_Class')

    h = Multiply()(
        [ipt, Flatten()(
            Embedding(input_dim=num_cls, output_dim=latent_size)(cls))])

    net = Dense(units=3*3*128)(h)
    net = Reshape(target_shape=(3, 3, 128))(net)
    net = BatchNormalization(axis=-1)(net)
    net = Conv2DTranspose(filters=64,
                          kernel_size=(3, 3),
                          strides=(2, 2),
                          activation='relu')(net)
    net = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 activation='relu')(net)
    net = Conv2DTranspose(filters=64,
                          kernel_size=(3, 3),
                          strides=(2, 2),
                          activation='relu')(net)
    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 activation='relu')(net)
    net = BatchNormalization(axis=-1)(net)
    net = Reshape(target_shape=(-1, 1))(net)
    net = Conv1D(filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu')(net)
    net = Conv1D(filters=32,
                 kernel_size=5,
                 strides=2,
                 activation='relu')(net)
    net = Conv1D(filters=64,
                 kernel_size=7,
                 strides=3,
                 activation='relu')(net)
    net = Conv1D(filters=64,
                 kernel_size=9,
                 strides=4,
                 activation='relu')(net)
    net = Flatten()(net)
    net = Dense(units=points*3, activation='relu')(net)
    point_set = Reshape(target_shape=(points, 3), name='Generator_Output')(net)
    model = Model(inputs=[ipt, cls], outputs=point_set)
    print('Generator:')
    model.summary()
    return model


def Classifier(points=1024):
    def InputTransformNet(ipts):
        '''ipts is a keras tensor'''
        ipt = Input(shape=(points, 3), name='InputTransformNet_Input')
        expand = Lambda(function=lambda x: K.expand_dims(x, axis=-1))(ipt)
        net = Conv2D(filters=64,
                     kernel_size=(1, 3),
                     activation='relu')(expand)
        net = Conv2D(filters=128,
                     kernel_size=(1, 1),
                     activation='relu')(net)
        net = Conv2D(filters=1024,
                     kernel_size=(1, 1),
                     activation='relu')(net)

        max_pool = MaxPool2D(pool_size=(points, 1))(net)

        net = Flatten()(max_pool)
        net = Dense(units=512, activation='relu')(net)
        net = Dense(units=256, activation='relu')(net)
        net = Dense(units=3*3)(net)

        bias = Input(tensor=K.eye(3, dtype='float32'),
                     name='InputTransformNet_Bias')

        expand = Lambda(function=lambda x: K.expand_dims(x, axis=0))(bias)
        expand = Flatten()(expand)
        # added = Add()([net, expand])
        added = Lambda(function=lambda t: t[0]+t[1])([net, expand])
        result = Reshape(target_shape=(
            3, 3), name='InputTransformNet_Output')(added)

        model = Model(inputs=[ipt, bias], outputs=[result])
        print('Input transform net:')
        model.summary()
        return model([ipts, bias]), bias

    def FeatureTransformNet(ipts):
        '''ipts is a keras tensor'''
        ipt = Input(shape=(points, 1, 64), name='FeatureTransformNet_Input')
        net = Conv2D(filters=64,
                     kernel_size=(1, 1),
                     activation='relu')(ipt)
        net = Conv2D(filters=128,
                     kernel_size=(1, 1),
                     activation='relu')(net)
        net = Conv2D(filters=1024,
                     kernel_size=(1, 1),
                     activation='relu')(net)

        max_pool = MaxPool2D(pool_size=(points, 1))(net)

        net = Flatten()(max_pool)
        net = Dense(units=512, activation='relu')(net)
        net = Dense(units=256, activation='relu')(net)
        net = Dense(units=64*64)(net)

        bias = Input(tensor=K.eye(64, dtype='float32'),
                     name='FeatureTransformNet_Bias')

        expand = Lambda(function=lambda x: K.expand_dims(x, axis=0))(bias)
        expand = Flatten()(expand)
        # added = Add()([net, expand])
        added = Lambda(function=lambda t: t[0]+t[1])([net, expand])
        result = Reshape(target_shape=(64, 64),
                         name='FeatureTransformNet_Output')(added)

        model = Model(inputs=[ipt, bias], outputs=[result])
        print('Feature transform net:')
        model.summary()
        return model([ipts, bias]), bias

    ipts = Input(shape=(points, 3), name='Classifier_Input')
    biases = []

    transform, bias = InputTransformNet(ipts)

    biases.append(bias)

    net = Lambda(function=lambda t: K.batch_dot(t[0], t[1]))([ipts, transform])
    expand = Lambda(function=lambda x: K.expand_dims(x, axis=-1))(net)
    net = Conv2D(filters=64,
                 kernel_size=(1, 3),
                 activation='relu')(expand)
    net = Conv2D(filters=64,
                 kernel_size=(1, 1),
                 activation='relu')(net)
    net = BatchNormalization(axis=-1)(net)

    transform, bias = FeatureTransformNet(net)

    biases.append(bias)

    net = Lambda(function=lambda x: K.squeeze(x, axis=2))(net)
    net = Lambda(function=lambda t: K.batch_dot(t[0], t[1]))([net, transform])
    net = Lambda(function=lambda x: K.expand_dims(x, axis=2))(net)
    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 activation='relu')(net)
    net = Conv2D(filters=1024,
                 kernel_size=(1, 1),
                 activation='relu')(net)
    net = BatchNormalization(axis=-1)(net)

    max_pool = MaxPool2D(pool_size=(net.shape.as_list()[1:3]))(net)

    flat = Flatten()(max_pool)
    net = Dense(units=512,
                activation='relu')(flat)
    drop = Dropout(rate=0.3)(net)
    net = Dense(units=256,
                activation='relu')(drop)
    drop = Dropout(rate=0.3)(net)
    is_real = Dense(units=1,
                    activation='sigmoid',
                    name='Classifier_Real')(net)
    net = Dense(units=40,
                activation='softmax',
                name='Classifier_Class')(net)

    model = Model(inputs=[ipts]+biases, outputs=[net, is_real])
    print('Classifier model:')
    model.summary()
    return model, biases


def Residual(points=1024):
    def InputTransformNet(ipts):
        '''ipts is a keras tensor'''
        ipt = Input(shape=(points, 3), name='InputTransformNet_Input')
        expand = Lambda(function=lambda x: K.expand_dims(x, axis=-1))(ipt)
        net = Conv2D(filters=64,
                     kernel_size=(1, 3),
                     activation='relu')(expand)
        net = Conv2D(filters=128,
                     kernel_size=(1, 1),
                     activation='relu')(net)
        net = Conv2D(filters=1024,
                     kernel_size=(1, 1),
                     activation='relu')(net)

        max_pool = MaxPool2D(pool_size=(points, 1))(net)

        net = Flatten()(max_pool)
        net = Dense(units=512, activation='relu')(net)
        net = Dense(units=256, activation='relu')(net)
        net = Dense(units=3*3)(net)

        bias = Input(tensor=K.eye(3, dtype='float32'),
                     name='InputTransformNet_Bias')

        expand = Lambda(function=lambda x: K.expand_dims(x, axis=0))(bias)
        expand = Flatten()(expand)
        # added = Add()([net, expand])
        added = Lambda(function=lambda t: t[0]+t[1])([net, expand])
        result = Reshape(target_shape=(
            3, 3), name='InputTransformNet_Output')(added)

        model = Model(inputs=[ipt, bias], outputs=[result])
        print('Input transform net:')
        model.summary()
        return model([ipts, bias]), bias

    def FeatureTransformNet(ipts):
        '''ipts is a keras tensor'''
        ipt = Input(shape=(points, 1, 64), name='FeatureTransformNet_Input')
        net = Conv2D(filters=64,
                     kernel_size=(1, 1),
                     activation='relu')(ipt)
        net = Conv2D(filters=128,
                     kernel_size=(1, 1),
                     activation='relu')(net)
        net = Conv2D(filters=1024,
                     kernel_size=(1, 1),
                     activation='relu')(net)

        max_pool = MaxPool2D(pool_size=(points, 1))(net)

        net = Flatten()(max_pool)
        net = Dense(units=512, activation='relu')(net)
        net = Dense(units=256, activation='relu')(net)
        net = Dense(units=64*64)(net)

        bias = Input(tensor=K.eye(64, dtype='float32'),
                     name='FeatureTransformNet_Bias')

        expand = Lambda(function=lambda x: K.expand_dims(x, axis=0))(bias)
        expand = Flatten()(expand)
        # added = Add()([net, expand])
        added = Lambda(function=lambda t: t[0]+t[1])([net, expand])
        result = Reshape(target_shape=(64, 64),
                         name='FeatureTransformNet_Output')(added)

        model = Model(inputs=[ipt, bias], outputs=[result])
        print('Feature transform net:')
        model.summary()
        return model([ipts, bias]), bias

    ipts = Input(shape=(points, 3), name='Classifier_Input')
    biases = []

    transform, bias = InputTransformNet(ipts)

    biases.append(bias)

    dot_output = Lambda(function=lambda t: K.batch_dot(
        t[0], t[1]))([ipts, transform])
    expand = Lambda(function=lambda x: K.expand_dims(x, axis=-1))(dot_output)
    dot_output = Dense(units=64, activation='relu')(dot_output)
    dot_output = Dropout(rate=.2)(dot_output)
    net = Conv2D(filters=64,
                 kernel_size=(1, 3),
                 activation='relu')(expand)
    net = Conv2D(filters=64,
                 kernel_size=(1, 1),
                 activation='relu')(net)
    dot_output = Lambda(
        function=lambda x: K.expand_dims(x, axis=2))(dot_output)
    net = Add()([net, dot_output])
    net = BatchNormalization(axis=-1)(net)

    transform, bias = FeatureTransformNet(net)

    biases.append(bias)

    net = Lambda(function=lambda x: K.squeeze(x, axis=2))(net)
    dot_output = Lambda(function=lambda t: K.batch_dot(
        t[0], t[1]))([net, transform])
    net = Lambda(function=lambda x: K.expand_dims(x, axis=2))(dot_output)
    dot_output = Flatten()(
        Dense(units=1, activation='relu')(dot_output)
    )
    dot_output = Dropout(rate=.2)(dot_output)
    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 activation='relu')(net)
    net = Conv2D(filters=1024,
                 kernel_size=(1, 1),
                 activation='relu')(net)
    net = BatchNormalization(axis=-1)(net)

    max_pool = MaxPool2D(pool_size=(net.shape.as_list()[1:3]))(net)

    flat = Flatten()(max_pool)
    flat = Add()([flat, dot_output])
    flatten_output = Dense(units=256, activation='relu')(flat)
    flatten_output = Dropout(rate=.3)(flatten_output)
    net = Dense(units=512,
                activation='relu')(flat)
    drop = Dropout(rate=.3)(net)
    net = Dense(units=256,
                activation='relu')(drop)
    net = Add()([flatten_output, net])
    drop = Dropout(rate=.3)(net)
    is_real = Dense(units=1,
                    activation='sigmoid',
                    name='Classifier_Real')(net)
    net = Dense(units=40,
                activation='softmax',
                name='Classifier_Class')(net)

    model = Model(inputs=[ipts]+biases, outputs=[net, is_real])
    print('Classifier model:')
    model.summary()
    return model, biases


if __name__ == "__main__":
    batch_size = 64
    points = 2048
    verbose = False
    np_test = False

    def classifier():
        classifier, biases = Classifier(points=points)
        if np_test:
            out = classifier.predict(
                np.ones(shape=(batch_size*6, points, 3)), batch_size=batch_size)
        if verbose:
            print(out)
            print(type(out))
            print(len(out))
            print(out[0].shape)
            print(out[1].shape)

    def generator():
        latent_size = 107
        batch_size = 10
        num_points = 512
        generator = Generator(latent_size, points=num_points)
        if np_test:
            noise = np.random.randn(batch_size, latent_size)
            cls_noise = np.random.randint(
                low=0,
                high=39,
                size=(batch_size, 1),
                dtype=np.int32)
            output = generator.predict([noise, cls_noise])
        if verbose:
            print(output)
            print(output.shape)

    classifier()
    generator()
