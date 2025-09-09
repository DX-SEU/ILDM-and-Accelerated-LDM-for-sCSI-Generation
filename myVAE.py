import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def sampler(z_mean, z_log_var):
    batch_size = tf.shape(z_mean)[0]
    z_size = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch_size, z_size))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    def __init__(self, latent_dim, activation):
        super().__init__()
        self.conv1 = layers.Conv2D(64, 3, activation=activation, strides=2, padding="same")
        self.conv2 = layers.Conv2D(128, 3, activation=activation, strides=2, padding="same")
        self.flt = layers.Flatten()
        self.den1 = layers.Dense(512, activation=activation)
        self.den2 = layers.Dense(128, activation=activation)
        self.den3 = layers.Dense(16, activation=activation)
        self.mean_layer = layers.Dense(latent_dim, name="z_mean")
        self.var_layer = layers.Dense(latent_dim, name="z_var")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flt(x)
        x = self.den1(x)
        x = self.den2(x)
        x = self.den3(x)
        latent_mean = self.mean_layer(x)
        latent_var = self.var_layer(x)
        return latent_mean, latent_var


class Decoder(layers.Layer):
    def __init__(self, activation):
        super().__init__()
        self.den1 = layers.Dense(128, activation=activation)
        self.den2 = layers.Dense(512, activation=activation)
        self.den3 = layers.Dense(32 * 32 * 64, activation=activation)
        self.resh = layers.Reshape((32, 32, 64))
        self.dec1 = layers.Conv2DTranspose(128, 3, activation=activation, strides=2, padding="same")
        self.dec2 = layers.Conv2DTranspose(64, 3, activation=activation, strides=2, padding="same")
        self.conv1 = layers.Conv2D(16, 3, activation=activation, padding="same")
        self.conv2 = layers.Conv2D(1, 3, activation=activation, padding="same")

    def call(self, inputs):
        x = self.den1(inputs)
        x = self.den2(x)
        x = self.den3(x)
        x = self.resh(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class VAEModel(keras.Model):
    def __init__(self, latent_dim, activation, weight_res, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(latent_dim, activation)
        self.decoder = Decoder(activation)
        self.weight_res = weight_res
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.build(input_shape=(None, 128, 128, 1))

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z_sample = sampler(z_mean, z_log_var)
        reconstruction = self.decoder(z_sample)
        return reconstruction

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(inputs)
            z_sample = sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z_sample)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(inputs, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = self.weight_res * reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }






