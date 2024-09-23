'''
Conditional GAN (CGAN) 

- with Functional API model

'''

import keras
from keras import layers
from keras import ops
import tensorflow as tf
import numpy as np

# parameters
batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128
generator_dim = latent_dim + num_classes

# We'll use all the available examples from both the training and test
# sets.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, image_size, image_size, 1))
all_labels = keras.utils.to_categorical(all_labels, num_classes)

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

# Create the discriminator.
def create_discriminator(image_size, num_classes):
    input_image = keras.Input(shape=(image_size, image_size, 1))
    input_label = keras.Input(shape=(num_classes,))
    x = layers.Dense(image_size*image_size)(input_label)
    x = layers.Reshape((image_size, image_size, 1))(x)
    x = ops.concatenate([input_image, x], axis=-1)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.GlobalMaxPooling2D()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=[input_image, input_label], outputs=outputs, name="discriminator")
    return model

discriminator = create_discriminator(image_size, num_classes)
discriminator.summary()

# Create the generator.
def create_generator(latent_dim, num_classes):
    input_latent = keras.Input(shape = (latent_dim, ))
    input_label = keras.Input(shape=(num_classes, ))
    x = ops.concatenate([input_latent, input_label], axis=-1)
    # We want to generate 128 + num_classes coefficients to reshape into a
    # 7x7x(128 + num_classes) map.
    x = layers.Dense(7 * 7 * generator_dim)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((7, 7, generator_dim))(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    outputs = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(x)
    model = keras.Model(inputs=[input_latent, input_label], outputs=outputs, name="generator")
    return model

generator = create_generator(latent_dim, num_classes)
generator.summary()

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(1556)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Sample random points in the latent space.
        # This is for the generator.
        batch_size = ops.shape(real_images)[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator([random_latent_vectors, one_hot_labels])

        # Combine them with real images.
        combined_images = ops.concatenate(
            [generated_images, real_images], axis=0
        )
        
        # repeat lables two times.
        combined_labels = ops.repeat(one_hot_labels, 2, axis=0)

        # Assemble labels discriminating real from fake images.
        labels = ops.concatenate(
            [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, combined_labels])
            d_loss = self.loss_fn(labels, predictions)
        
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Assemble labels that say "all real images".
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator([random_latent_vectors, one_hot_labels])
            predictions = self.discriminator([fake_images, one_hot_labels])
            g_loss = self.loss_fn(misleading_labels, predictions)
        
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }
        
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(dataset, epochs=2)