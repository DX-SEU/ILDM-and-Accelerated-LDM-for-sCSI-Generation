import os
import numpy as np
import scipy.io as sio
import h5py
from myUNet import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class GaussianDiffusion:
    def __init__(self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=500,
    ):
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32  # sqrt alphat
        )
        self.sqrt_one_minus_alphas_cumprod = tf.constant(  # 1-sqrt alphat
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

    def _extract(self, a, t, x_shape):
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1])

    def q_sample(self, x_start, t, noise):
        x_start_shape = tf.shape(x_start)
        image_t = (
            self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )
        return image_t

class DiffusionModel(keras.Model):
    def __init__(self, network, timesteps, gdf_util):
        super().__init__()
        self.network = network
        self.timesteps = timesteps
        self.gdf_util = gdf_util

    def train_step(self, data):
        images, labels = data
        batch_size1 = tf.shape(images)[0]
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size1,), dtype=tf.int64
        )
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        images_t = self.gdf_util.q_sample(images, t, noise)
        with tf.GradientTape() as tape:
            pred_noise = self.network([images_t, t, labels])
            loss = self.loss(noise, pred_noise)
        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        return {"loss": loss}


class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MyCallBack, self).__init__()
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.model.save_weights(f'ckpt_file/ILDM/model_{epoch + 1}/ildm.ckpt')
        if (epoch + 1) == 30000:
            self.model.optimizer.lr = 1e-4
        elif epoch + 1 == 60000:
            self.model.optimizer.lr = 1e-5
        elif epoch + 1 == 90000:
            self.model.optimizer.lr = 1e-6


trainFile_data = 'data/ILDM_dataset/vae_lat_train.mat'
trainData = sio.loadmat(trainFile_data)
train_data = np.array(trainData['vae_lat_train'])
train_data = train_data / 10
train_data = tf.convert_to_tensor(train_data, dtype='float32')
trainFile_label = 'data/ILDM_dataset/trainLoc.mat'
trainLabel = h5py.File(trainFile_label)
LocTrain = np.array(trainLabel['trainLoc'])
LocTrain = LocTrain.swapaxes(0, 1)
LocTrain = LocTrain + np.array([[-150, 50]])
# LDM configuration
img_size = 128
batch_size = 4096
shuffle_buffer = 8000
num_epochs = 100000
total_timesteps = 500
learning_rate = 1e-3
img_channels = 1
latent_size = 8
first_conv_channels = 8
channel_multiplier = [1, 2, 4]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [True, True, True, True]
num_res_blocks = 2
norm_groups = 8
# Create dataset
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_data, LocTrain))
    .shuffle(buffer_size=shuffle_buffer)
    .batch(batch_size)
)
# Build the unet model
network = build_model(
    img_size=latent_size,
    img_channels=img_channels,
    first_conv_channels = first_conv_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
)
gdf_util = GaussianDiffusion(timesteps=total_timesteps)
model = DiffusionModel(
    network=network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)
checkpoint_callback1 = MyCallBack()
history = model.fit(
    train_dataset,
    shuffle=True,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[checkpoint_callback1],
)
loss = history.history['loss']
sio.savemat('results/loss.mat', {'loss': loss})