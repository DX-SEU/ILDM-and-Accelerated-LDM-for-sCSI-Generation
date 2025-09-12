import os
import numpy as np
from scipy import io
import h5py
from myVAE import *
from myUNet import *
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class GaussianDiffusion:
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=500,
    ):
        self.betas = betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

    def p_sample(self, pred_noise, x, t, cache_timestep):
        delt = self.sqrt_one_minus_alphas_cumprod[t]/self.sqrt_one_minus_alphas_cumprod[cache_timestep] * tf.sqrt(self.betas[cache_timestep])
        sample = self.sqrt_alphas_cumprod[t] * (x - self.sqrt_one_minus_alphas_cumprod[cache_timestep] * pred_noise) / \
                 self.sqrt_alphas_cumprod[cache_timestep] + tf.sqrt(1 - self.alphas_cumprod[t] - delt**2) * pred_noise
        return sample


class DiffusionModel(keras.Model):
    def __init__(self, network, timesteps, delta_step, gdf_util):
        super().__init__()
        self.network = network
        self.timesteps = timesteps
        self.delta_step = delta_step
        self.gdf_util = gdf_util

    def train_step(self, data):
        images, labels = data
        batch_size1 = tf.shape(images)[0]
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size1,), dtype=tf.int64)
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        images_t = self.gdf_util.q_sample(images, t, noise)
        with tf.GradientTape() as tape:
            pred_noise = self.network([images_t, t, labels])
            loss = self.loss(noise, pred_noise)
        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        return {"loss": loss}

    def generate_images(self, loc, num_images=510):
        samples = tf.random.normal(shape=(num_images, latent_size, latent_size), dtype=tf.float32)
        y = loc[0:num_images, :]
        cache_timestep = self.timesteps - 1
        for t in reversed(range(0, self.timesteps, self.delta_step)):
            tt = tf.cast(tf.fill(num_images, cache_timestep), dtype=tf.int64)
            pred_noise = self.network([samples, tt, y])
            samples = self.gdf_util.p_sample(pred_noise, samples, t, cache_timestep)
            cache_timestep = t
        return samples

    def plot_images(self, loc, num_cols=510):
        dm_lat_test = self.generate_images(loc, num_images=num_cols).numpy() * 10
        dm_lat_test1 = dm_lat_test.reshape((num_cols, latent_dim))
        dm_rec_test = vae.decoder(dm_lat_test1).numpy()  # ()
        dm_rec_test = dm_rec_test.reshape(num_cols, img_size, img_size)
        return dm_rec_test


trainFile_label = 'data/ILDM_dataset/testLoc.mat'
trainLabel = h5py.File(trainFile_label)
LocTrain = np.array(trainLabel['testLoc'])
LocTrain = LocTrain.swapaxes(0, 1)
LocTrain = LocTrain + np.array([[-150, 50]])
# DDPM configuration
img_size = 128
total_timesteps = 500
delta_step = 10  # Sampling time space for accelerated LDM
img_channels = 1
latent_size = 8
first_conv_channels = 8
channel_multiplier = [1, 2, 4]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [True, True, True, True]
num_res_blocks = 2
norm_groups = 256
latent_dim = latent_size ** 2
weight_res = 1e3
activation = "ELU"
vae = VAEModel(latent_dim, activation, weight_res)
vae.load_weights("ckpt_file/VAE64/model_1/vae.ckpt")  # Change to the saved model weight path
# Build the unet model
network = build_model(
    img_size=latent_size,
    img_channels=img_channels,
    first_conv_channels=first_conv_channels,
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
    delta_step=delta_step,
)
model.load_weights("ckpt_file/ILDM/model_1/ildm.ckpt")  # Change to the saved model weight path
s1 = time.time()
aldm_rec_test = model.plot_images(LocTrain, num_cols=1020)
e1 = time.time()
print(e1 - s1)
io.savemat('data/results/aldm_rec_test.mat', {'aldm_rec_test': aldm_rec_test})



