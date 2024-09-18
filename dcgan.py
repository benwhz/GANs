'''
Deep Convolutional GAN (DCGAN) 

- with Keras 3 API

'''
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np

class Generator(keras.Model):
    def __init__(self, image_size = 28, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = 5
        self._filters = [128, 64, 32, 1]
        self.image_size = image_size
        self.start_image_size = image_size // 4

        self.linear = keras.layers.Dense(self.start_image_size*self.start_image_size*self._filters[0])
        self.reshape = keras.layers.Reshape((self.start_image_size, self.start_image_size, self._filters[0]))

        self.activation = keras.layers.Activation('relu')

        self.norms = []
        self.conv2dts = []

        strides = 2
        for i, filters in enumerate(self._filters):
            if i >= 2:
                strides = 1
            self.norms.append(keras.layers.BatchNormalization())
            self.conv2dts.append(keras.layers.Conv2DTranspose(filters=self._filters[i],
                                                     kernel_size=self.kernel_size,
                                                     strides=strides,padding='same'))

        self.activation2 = keras.layers.Activation('sigmoid')

    def call(self, inputs):
        x = self.linear(inputs)
        x = self.reshape(x)
        
        for i, _ in enumerate(self._filters):
            x = self.norms[i](x)
            x = self.activation(x)
            x = self.conv2dts[i](x)

        return self.activation2(x)
    

    def build_graph(self, length):
        input = keras.Input(shape=(length, ))
    
        return keras.Model(inputs=input, 
                          outputs=self.call(input))  

model = Generator()
model.build_graph(100).summary()

'''
input = keras.Input((100,))
generator = Generator()
ouput = generator(input)
model = keras.Model(input, ouput)
model.summary()
'''
class Discriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = 5
        self._filters = [32, 64, 128, 256]

        self.flatten = keras.layers.Flatten()
        self.linear = keras.layers.Dense(1)

        self.activation = keras.layers.LeakyReLU(negative_slope=0.2)

        self.conv2ds = []

        strides = 2
        for i, filters in enumerate(self._filters):
            if i >= 3:
                strides = 1
            self.conv2ds.append(keras.layers.Conv2D(filters=self._filters[i],
                                                     kernel_size=self.kernel_size,
                                                     strides=strides,padding='same'))

        self.activation2 = keras.layers.Activation('sigmoid')

    def call(self, inputs):
        x = inputs
        for i, _ in enumerate(self._filters):
            x = self.activation(x)
            x = self.conv2ds[i](x)

        x = self.flatten(x)
        x = self.linear(x)

        return self.activation2(x)

    def build_graph(self, height, width):
        input = keras.Input(shape=(height, width, 1))
    
        return keras.Model(inputs=input, 
                          outputs=self.call(input))  

model = Discriminator()
model.build_graph(28, 28).summary()

'''
input = keras.Input((28, 28, 1))
d = Discriminator()
ouput = d(input)

model = keras.Model(input, ouput)
model.summary()
'''

# load the dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
# reshape and normalize
img_height, img_width = x_train.shape[1], x_train.shape[2]
x_train = np.reshape(x_train, (-1, img_height, img_width, 1))
x_train = x_train.astype('float32') / 255.0



# discriminator model
discriminator = Discriminator()
discriminator.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy()
    ],
)
discriminator.build(input_shape=(28, 28, 1))
'''
inputs = keras.Input((28, 28, 1))
discriminator(inputs)
discriminator.summary()
'''

# generator model
generator = Generator()
generator.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy()
    ],
)
generator.build(input_shape=(100,))

# adversarial model
#discriminator.trainable = False
inputs = keras.Input(shape=(100,), name='z_input')

adversarial = keras.Model(inputs=inputs, outputs=discriminator(generator(inputs)), name='adversarial')
adversarial.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy()
    ],
)
#adversarial.build(input_shape=(100,))
adversarial.summary()

# network parameters
# the latent or z vector is 100-dim
latent_size = 100
batch_size = 64
train_steps = 100 #40000
lr = 2e-4
decay = 6e-8

noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
train_size = x_train.shape[0]

for i in range(train_steps):
    rand_indexes = np.random.randint(0, train_size, size=batch_size)
    real_images = x_train[rand_indexes]

    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
    fake_images = generator.predict(noise)

    x = np.concatenate((real_images, fake_images))
    y = np.ones([2 * batch_size, 1])
    y[batch_size:, :] = 0.0
    
    # train discriminator network
    loss, acc = discriminator.train_on_batch(x, y)
    log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
    print(log)

    # train the adversarial network
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
    y = np.ones([batch_size, 1])
    print(noise.shape, y.shape)
    #exit()
    loss, acc = adversarial.train_on_batch(noise, y)
    log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
    print(log)