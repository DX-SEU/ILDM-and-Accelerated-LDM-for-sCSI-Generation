import os
import numpy as np
import scipy.io as sio
import h5py
from myVAE import *
from myUNet import *
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GaussianDiffusion:
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.betas = tf.constant(betas, dtype=tf.float32)
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32)
        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32)
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32)
        self.posterior_mean_coef1 = tf.constant(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),dtype=tf.float32,)
        self.posterior_mean_coef2 = tf.constant((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),dtype=tf.float32,)

    def _extract(self, a, t, x_shape):
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1])

    def q_sample(self, x_start, t, noise):
        x_start_shape = tf.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        return posterior_mean

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)
        model_mean = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        model_mean = self.p_mean_variance(pred_noise, x=x, t=t, clip_denoised=clip_denoised)
        return model_mean


class DiffusionModel(keras.Model):
    def __init__(self, network, timesteps, gdf_util):
        super().__init__()
        self.network = network
        self.timesteps = timesteps
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

    def generate_images(self, loc, num_images=1020):
        samples = tf.random.normal(shape=(num_images, latent_size, latent_size), dtype=tf.float32)
        y = loc[0:num_images, :]
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.network([samples, tt, y])
            samples = self.gdf_util.p_sample(pred_noise, samples, tt, clip_denoised=True)
        return samples

    def plot_images(self, loc, num_cols=1020):
        dm_lat_test = self.generate_images(loc, num_images=num_cols).numpy() * 10
        dm_lat_test1 = dm_lat_test.reshape((num_cols, latent_dim))
        dm_rec_test = vae.decoder(dm_lat_test1).numpy()
        dm_rec_test = dm_rec_test.reshape(num_cols, img_size, img_size)
        return dm_rec_test


trainFile_label = 'data/ILDM_dataset/testLoc.mat'
trainLabel = h5py.File(trainFile_label)
LocTrain = np.array(trainLabel['testLoc'])
LocTrain = LocTrain.swapaxes(0, 1)
LocTrain = LocTrain + np.array([[-150, 50]])
# ILDM configuration
img_size = 128
total_timesteps = 500
img_channels = 1
latent_size = 8
first_conv_channels = 8
channel_multiplier = [1, 2, 4]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [True, True, True, True]
num_res_blocks = 2
norm_groups = 256
latent_dim = latent_size ** 2
weight_res = 1
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
)
model.load_weights("ckpt_file/ILDM/model_1/ILDM.ckpt")  # Change to the saved model weight path
s1 = time.time()
ildm_rec_test = model.plot_images(LocTrain, num_cols=1020)
e1 = time.time()
print(e1 - s1)
sio.savemat('data/results/ildm_rec_test.mat', {'ildm_rec_test': ildm_rec_test})



